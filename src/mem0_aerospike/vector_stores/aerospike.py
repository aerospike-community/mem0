import json
from datetime import datetime, timezone

try:
    from aerospike_async import BatchWritePolicy, ExpOperation, Operation
    from aerospike_sdk import (
        Behavior,
        DataSet,
        Exp,
        Filter,
        Order,
        OrderByType,
        ResultCode,
        Vector,
    )
    from aerospike_sdk.exceptions import AerospikeError
    from aerospike_sdk.policy.behavior_settings import Mode, OpKind, OpShape
    from aerospike_sdk.policy.policy_mapper import to_batch_policy
    from aerospike_sdk.sync import ClusterDefinition
except ImportError as e:
    raise ImportError(
        "Aerospike vector store support requires the preview 'aerospike-sdk' package. "
        "Install it with: pip install 'aerospike-sdk'."
    ) from e

from mem0.vector_stores.base import VectorStoreBase

from mem0_aerospike.filters import (
    CORE_STRING_FIELDS,
    EMBEDDING_BIN,
    METADATA_BIN,
    SCOPE_FIELDS,
    TIMESTAMP_FIELDS,
    compile_filters,
)
from mem0_aerospike.models import MemoryResult


def _iso_to_epoch_ms(value: str) -> int:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _epoch_ms_to_iso(ms: int) -> str:
    seconds, millis = divmod(int(ms), 1000)
    dt = datetime.fromtimestamp(seconds, tz=timezone.utc).replace(microsecond=millis * 1000)
    return dt.isoformat(timespec="milliseconds")


class AerospikeDB(VectorStoreBase):
    """Aerospike-backed vector store provider."""

    def __init__(
        self,
        namespace: str,
        collection_name: str,
        embedding_model_dims: int,
        host: str = "localhost",
        port: int = 3000,
        allow_scans_with_where: bool = False,
        max_record_bytes: int = 943718,
    ):
        """Connect to Aerospike, derive a sendKey-enabled session, and create the collection.

        Args:
            namespace: Aerospike namespace holding the collection.
            collection_name: Aerospike set name used as the collection.
            embedding_model_dims: Dimension of stored embedding vectors.
            host: Seed host.
            port: Seed port.
            allow_scans_with_where: Permit filtered queries without a
                user_id/agent_id/run_id scope predicate to fall back to a scan.
            max_record_bytes: Client-side record-size guardrail (~0.9 MiB default).
        """
        self.namespace = namespace
        self.collection_name = collection_name
        self.embedding_model_dims = embedding_model_dims
        self.host = host
        self.port = port
        self.allow_scans_with_where = allow_scans_with_where
        self.max_record_bytes = max_record_bytes
        self.dataset = DataSet.of(namespace, collection_name)

        self._cluster = ClusterDefinition(host, port).connect()
        # sendKey = true so memory_id can be recovered from secondary-index and
        # Top-K query results, which otherwise return only digests.
        behavior = Behavior.DEFAULT.derive_with_changes(f"mem0-aerospike-{collection_name}", send_key=True)
        self.session = self._cluster.create_session(behavior)

        batch_write_settings = behavior.get_settings(OpKind.WRITE_RETRYABLE, OpShape.BATCH, Mode.AP)
        self._batch_policy = to_batch_policy(batch_write_settings)
        self._batch_write_policy = BatchWritePolicy()
        self._batch_write_policy.send_key = batch_write_settings.send_key
        self._batch_write_policy.commit_level = batch_write_settings.commit_level
        self._batch_write_policy.durable_delete = batch_write_settings.durable_delete

        self.create_col()

    def close(self) -> None:
        """Close the underlying Aerospike cluster connection, idempotently."""
        cluster = getattr(self, "_cluster", None)
        if cluster is not None:
            cluster.close()
            self._cluster = None

    def __exit__(self, *exc):
        self.close()

    # -- Collection lifecycle --------------------------------------------------

    def create_col(self, name=None, vector_size=None, distance=None):
        """Create the collection's secondary indexes on user_id/agent_id/run_id (idempotent).

        ``vector_size`` and ``distance`` are accepted for API compatibility with
        the mem0 ``VectorStoreBase`` seam but are currently ignored; this
        provider relies on the Aerospike server's vector bin handling rather
        than creating a client-side vector index.
        """
        collection_name = name or self.collection_name
        dataset = DataSet.of(self.namespace, collection_name) if name else self.dataset

        existing = {
            idx.get("name")
            for idx in self.session.list_indexes()
            if idx.get("namespace") == self.namespace and idx.get("set") == collection_name
        }
        for field in SCOPE_FIELDS:
            index_name = f"{collection_name}_{field}_idx"
            if index_name in existing:
                continue
            task = self.session.index(dataset=dataset).on_bin(field).named(index_name).string().create()
            task.wait_till_complete_blocking()

    def list_cols(self):
        """Return the names of all sets (collections) in the namespace."""
        return [s.name for s in self.session.info().sets(self.namespace)]

    def delete_col(self):
        """Truncate the collection's set and wait for the truncate task to finish."""
        task = self.session.truncate(self.dataset)
        if task is not None:
            task.wait_till_complete_blocking()

    def reset(self):
        """Empty the collection and recreate its secondary indexes."""
        self.delete_col()
        self.create_col()

    def col_info(self):
        """Return ``{"name", "count"}`` for this collection (count is the record total)."""
        for set_detail in self.session.info().sets(self.namespace):
            if set_detail.name == self.collection_name:
                return {"name": self.collection_name, "count": set_detail.objects}
        return {"name": self.collection_name, "count": 0}

    # -- Payload <-> bins translation ------------------------------------------

    def _payload_to_bins(self, payload: dict, vector=None) -> dict:
        bins = {}
        metadata = {}
        for key, value in (payload or {}).items():
            if key in TIMESTAMP_FIELDS:
                bins[f"{key}_ms"] = _iso_to_epoch_ms(value)
            elif key in CORE_STRING_FIELDS:
                bins[key] = value
            else:
                metadata[key] = value
        bins[METADATA_BIN] = metadata
        if vector is not None:
            bins[EMBEDDING_BIN] = Vector(vector)
        return bins

    def _bins_to_payload(self, bins: dict) -> dict:
        payload = {}
        for key in CORE_STRING_FIELDS:
            if bins.get(key) is not None:
                payload[key] = bins[key]
        for key in TIMESTAMP_FIELDS:
            ms_value = bins.get(f"{key}_ms")
            if ms_value is not None:
                payload[key] = _epoch_ms_to_iso(ms_value)
        metadata = bins.get(METADATA_BIN) or {}
        payload.update(metadata)
        return payload

    def _check_size(self, payload: dict, vector=None):
        size = len(json.dumps(payload or {}, default=str).encode("utf-8"))
        if vector is not None:
            size += len(vector) * 4
        if size > self.max_record_bytes:
            raise ValueError(f"record size {size} bytes exceeds max_record_bytes ({self.max_record_bytes})")

    # -- Insert, get, update, delete --------------------------------------------

    def insert(self, vectors, payloads=None, ids=None):
        """Batch-write vectors and payloads under the given ids (upsert semantics).

        Raises ValueError if ids is None or lengths mismatch, AerospikeError if
        any record in the batch fails.
        """
        payloads = payloads or [{} for _ in vectors]
        if ids is None:
            raise ValueError("ids is required for insert")
        if len(payloads) != len(vectors):
            raise ValueError(f"payloads length ({len(payloads)}) must match vectors length ({len(vectors)})")
        if len(ids) != len(vectors):
            raise ValueError(f"ids length ({len(ids)}) must match vectors length ({len(vectors)})")
        keys, bins_list = [], []
        for vector, payload, vector_id in zip(vectors, payloads, ids):
            self._check_size(payload, vector)
            keys.append(self.dataset.id(vector_id))
            bins_list.append(self._payload_to_bins(payload, vector))
        results = self.session.client.underlying_client.batch_write_blocking(
            keys, bins_list, batch_policy=self._batch_policy, write_policy=self._batch_write_policy
        )
        failures = [r for r in results if r.result_code != ResultCode.OK]
        if failures:
            first = failures[0]
            raise AerospikeError(
                f"batch insert failed for {len(failures)}/{len(results)} keys "
                f"(first: {first.key.value!r}: {first.server_message or first.result_code})",
                result_code=first.result_code,
                in_doubt=first.in_doubt,
            )

    def get(self, vector_id):
        """Fetch one record by id; returns a MemoryResult or None if not found."""
        key = self.dataset.id(vector_id)
        try:
            record = self.session.get(key)
        except AerospikeError as e:
            if e.result_code == ResultCode.KEY_NOT_FOUND_ERROR:
                return None
            raise
        return MemoryResult(id=vector_id, payload=self._bins_to_payload(record.bins))

    def delete(self, vector_id):
        """Delete the record with the given id."""
        key = self.dataset.id(vector_id)
        self.session.delete(key).execute()

    def update(self, vector_id, vector=None, payload=None):
        """Replace the full payload (and optionally the vector) for an existing id."""
        self._check_size(payload or {}, vector)
        bins = self._payload_to_bins(payload or {}, vector)
        key = self.dataset.id(vector_id)
        self.session.upsert(key).put(bins).execute()

    # -- Filtered access: shared scoping guard ----------------------------------

    def _resolve_scope(self, filters):
        """Return the scope field and value for the secondary-index Filter.

        Returns a tuple ``(scope_field, scope_value)``. ``scope_value`` is ``None``
        when a scope field is present but cannot be expressed as a single
        equality predicate (e.g. ``{"ne": "alice"}``). ``scope_field`` and
        ``scope_value`` are both ``None`` when no scope field is present.

        Raises when `filters` is non-empty but contains none of
        user_id/agent_id/run_id and `allow_scans_with_where` is not set.
        """
        if not filters:
            return None, None
        for field in SCOPE_FIELDS:
            if field not in filters:
                continue
            value = filters[field]
            if not isinstance(value, dict):
                return field, value
            if "eq" in value:
                return field, value["eq"]
        if any(field in filters for field in SCOPE_FIELDS):
            return None, None
        if not self.allow_scans_with_where:
            raise ValueError(
                "Filtered queries must include at least one of user_id, agent_id, or run_id as a "
                "scoping predicate, or set allow_scans_with_where=True to allow a full scan."
            )
        return None, None

    def _read_bin_names(self):
        return list(CORE_STRING_FIELDS) + ["created_at_ms", "updated_at_ms", METADATA_BIN]

    # -- list() ------------------------------------------------------------------

    def list(self, filters: dict = None, top_k: int = None):
        """Return ``[results]`` of MemoryResults matching the filter dict.

        Queries are scoped by a secondary-index predicate on
        user_id/agent_id/run_id when available; a metadata-only filter raises
        unless ``allow_scans_with_where`` is set.
        """
        scope_field, scope_value = self._resolve_scope(filters)
        query = self.session.query(self.dataset)
        if scope_field is not None and scope_value is not None:
            query = query.filter(Filter.equal(scope_field, scope_value))
        remaining_expr = compile_filters(filters, exclude_key=scope_field)
        if remaining_expr is not None:
            query = query.where(remaining_expr)

        query = query.bins(self._read_bin_names())
        if top_k:
            query = query.limit(top_k)

        results = []
        for row in query.execute():
            if not row.is_ok or row.record is None:
                continue
            payload = self._bins_to_payload(row.record.bins)
            results.append(MemoryResult(id=row.record.key.value, payload=payload))
        return [results]

    # -- search() ------------------------------------------------------------------

    def search(self, query, vectors, top_k: int = 5, filters: dict = None):
        """Top-K cosine-similarity search over the filtered candidate set.

        ``query`` is ignored; the search uses ``vectors`` and the Aerospike
        vector bin directly, matching the mem0 ``VectorStoreBase`` seam.
        Returns MemoryResults ordered by descending score, with ``score``
        populated. Same scoping rules as ``list()``.
        """
        scope_field, scope_value = self._resolve_scope(filters)
        qbuilder = self.session.query(self.dataset)
        if scope_field is not None and scope_value is not None:
            qbuilder = qbuilder.filter(Filter.equal(scope_field, scope_value))
        remaining_expr = compile_filters(filters, exclude_key=scope_field)
        if remaining_expr is not None:
            qbuilder = qbuilder.where(remaining_expr)

        score_expr = Exp.cosine_similarity(Vector(vectors), Exp.vector_bin(EMBEDDING_BIN))
        ops = [Operation.get_bin(name) for name in self._read_bin_names()]
        ops.append(ExpOperation.read("score", score_expr))
        qbuilder = qbuilder.with_op_projection(*ops).order_by("score", OrderByType.DOUBLE, Order.DESC).top_k(top_k)

        results = []
        for row in qbuilder.execute():
            if not row.is_ok or row.record is None:
                continue
            bins = dict(row.record.bins)
            score = bins.pop("score", None)
            payload = self._bins_to_payload(bins)
            results.append(MemoryResult(id=row.record.key.value, payload=payload, score=score))
        return results

    def keyword_search(self, query: str, top_k: int = 5, filters: dict = None):
        """Keyword search is not supported; always returns None."""
        return None
