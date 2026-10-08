"""Unit tests for AerospikeDB with a mocked aerospike session (no server)."""

from unittest.mock import MagicMock

import pytest
from aerospike_sdk import ResultCode
from aerospike_sdk.exceptions import AerospikeError

import mem0_aerospike  # noqa: F401  -- registers the provider
from mem0_aerospike.vector_stores.aerospike import AerospikeDB


@pytest.fixture
def db(mocker):
    """AerospikeDB whose cluster/session are mocked; no server required."""
    session = MagicMock()
    session.list_indexes.return_value = []
    cluster = MagicMock()
    cluster.create_session.return_value = session
    mocker.patch(
        "mem0_aerospike.vector_stores.aerospike.ClusterDefinition",
        return_value=MagicMock(connect=MagicMock(return_value=cluster)),
    )
    instance = AerospikeDB(
        namespace="test",
        collection_name="unit_test",
        embedding_model_dims=4,
        host="localhost",
        port=3000,
    )
    return instance


def _payload(**overrides):
    payload = {
        "data": "hello",
        "hash": "h1",
        "user_id": "u1",
        "created_at": "2025-06-01T10:00:00.000+00:00",
        "updated_at": "2025-06-01T10:00:00.000+00:00",
        "category": "docs",
    }
    payload.update(overrides)
    return payload


# -- payload <-> bins ---------------------------------------------------------


def test_payload_to_bins_splits_core_and_metadata(db):
    bins = db._payload_to_bins(_payload(a_very_long_custom_field="v"), vector=[1.0, 0.0])
    assert bins["data"] == "hello"
    assert bins["user_id"] == "u1"
    assert "created_at" not in bins
    assert isinstance(bins["created_at_ms"], int)
    assert bins["metadata"]["category"] == "docs"
    assert bins["metadata"]["a_very_long_custom_field"] == "v"
    assert "embedding" in bins


def test_bins_to_payload_roundtrip(db):
    payload = _payload(extra_field="x")
    bins = db._payload_to_bins(payload, vector=[1.0])
    # Replace the Vector object so only plain bins remain
    bins.pop("embedding")
    assert db._bins_to_payload(bins) == payload


# -- insert validation ---------------------------------------------------------


def test_insert_requires_ids(db):
    with pytest.raises(ValueError, match="ids is required"):
        db.insert(vectors=[[1.0, 0.0]], payloads=[_payload()], ids=None)


def test_insert_rejects_mismatched_lengths(db):
    with pytest.raises(ValueError, match="payloads length"):
        db.insert(vectors=[[1.0], [0.0]], payloads=[_payload()], ids=["a", "b"])
    with pytest.raises(ValueError, match="ids length"):
        db.insert(vectors=[[1.0], [0.0]], payloads=[_payload(), _payload()], ids=["a"])


def test_insert_calls_batch_write(db):
    db.session.client.underlying_client.batch_write_blocking.return_value = [MagicMock(result_code=ResultCode.OK)]
    db.insert(vectors=[[1.0, 0.0]], payloads=[_payload()], ids=["id-1"])
    db.session.client.underlying_client.batch_write_blocking.assert_called_once()
    keys, bins_list = db.session.client.underlying_client.batch_write_blocking.call_args.args[:2]
    assert len(keys) == 1 and len(bins_list) == 1


def test_insert_raises_on_batch_failure(db):
    failure = MagicMock(result_code=ResultCode.KEY_EXISTS_ERROR, server_message="exists")
    failure.key.value = "bad-id"
    db.session.client.underlying_client.batch_write_blocking.return_value = [failure]
    with pytest.raises(AerospikeError, match="batch insert failed"):
        db.insert(vectors=[[1.0]], payloads=[_payload()], ids=["bad-id"])


def test_insert_oversized_record_raises(db):
    with pytest.raises(ValueError, match="max_record_bytes"):
        db.insert(vectors=[[1.0]], payloads=[_payload(data="x" * 2_000_000)], ids=["big"])


def test_update_oversized_record_raises(db):
    with pytest.raises(ValueError, match="max_record_bytes"):
        db.update("id", vector=[1.0], payload=_payload(data="x" * 2_000_000))


# -- get / delete / update ------------------------------------------------------


def test_get_returns_memory_result(db):
    bins = db._payload_to_bins(_payload(), vector=[1.0])
    bins.pop("embedding")
    db.session.get.return_value = MagicMock(bins=bins)
    result = db.get("some-id")
    assert result.id == "some-id"
    assert result.payload["user_id"] == "u1"
    assert result.score is None


def test_get_returns_none_on_key_not_found(db):
    db.session.get.side_effect = AerospikeError("not found", result_code=ResultCode.KEY_NOT_FOUND_ERROR)
    assert db.get("missing") is None


def test_get_reraises_other_errors(db):
    db.session.get.side_effect = AerospikeError("boom", result_code=ResultCode.SERVER_ERROR)
    with pytest.raises(AerospikeError):
        db.get("id")


def test_delete_executes(db):
    db.delete("id-1")
    db.session.delete.return_value.execute.assert_called_once()


def test_update_executes_upsert(db):
    db.update("id-1", vector=[1.0], payload=_payload(data="new"))
    db.session.upsert.return_value.put.return_value.execute.assert_called_once()


# -- collection lifecycle -------------------------------------------------------


def test_create_col_creates_scope_indexes(db):
    # Constructor already called create_col once against the mock session.
    assert db.session.index.call_count == 3  # user_id, agent_id, run_id


def test_create_col_skips_existing_indexes(db):
    db.session.index.reset_mock()
    db.session.list_indexes.return_value = [{"name": "unit_test_user_id_idx", "namespace": "test", "set": "unit_test"}]
    db.create_col()
    assert db.session.index.call_count == 2


def test_delete_col_waits_for_task(db):
    task = MagicMock()
    db.session.truncate.return_value = task
    db.delete_col()
    task.wait_till_complete_blocking.assert_called_once()


def test_delete_col_tolerates_none_task(db):
    db.session.truncate.return_value = None
    db.delete_col()


def test_col_info_reports_count(db):
    db.session.info.return_value.sets.return_value = [
        MagicMock(name="unit_test", objects=7),
    ]
    # MagicMock name kwarg sets mock name, not attribute; fix it
    db.session.info.return_value.sets.return_value[0].name = "unit_test"
    assert db.col_info() == {"name": "unit_test", "count": 7}


def test_col_info_missing_set(db):
    db.session.info.return_value.sets.return_value = []
    assert db.col_info() == {"name": "unit_test", "count": 0}


# -- scope resolution -----------------------------------------------------------


def test_resolve_scope_literal_and_eq(db):
    assert db._resolve_scope({"user_id": "alice"}) == ("user_id", "alice")
    assert db._resolve_scope({"user_id": {"eq": "alice"}}) == ("user_id", "alice")
    assert db._resolve_scope({"agent_id": "a"}) == ("agent_id", "a")
    assert db._resolve_scope({"run_id": "r"}) == ("run_id", "r")


def test_resolve_scope_non_eq_operator_not_indexable(db):
    assert db._resolve_scope({"user_id": {"ne": "alice"}}) == (None, None)


def test_resolve_scope_empty_filters(db):
    assert db._resolve_scope(None) == (None, None)
    assert db._resolve_scope({}) == (None, None)


def test_resolve_scope_rejects_unscoped_by_default(db):
    with pytest.raises(ValueError, match="user_id|scoping"):
        db._resolve_scope({"category": "work"})


def test_resolve_scope_allows_scan_when_configured(db):
    db.allow_scans_with_where = True
    assert db._resolve_scope({"category": "work"}) == (None, None)


def test_keyword_search_returns_none(db):
    assert db.keyword_search("q", top_k=5, filters={"user_id": "u"}) is None


def test_close_calls_cluster_close(db):
    cluster = db._cluster
    db.close()
    cluster.close.assert_called_once()


def test_close_is_idempotent(db):
    db.close()
    db.close()
    # _cluster was nulled after the first close, so a second call must not raise.
    assert db._cluster is None
