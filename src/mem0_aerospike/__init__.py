"""mem0-aerospike: Aerospike vector-store provider for unmodified mem0ai.

Importing this package registers "aerospike" with mem0ai's vector-store
registries, so `MemoryConfig(vector_store={"provider": "aerospike", ...})`
works without modifying mem0ai.
"""

from mem0_aerospike.patch import register_aerospike

register_aerospike()


def create_memory_config(vector_store=None, **kwargs):
    """Build a `MemoryConfig` that uses the Aerospike vector store.

    Performs provider registration and returns a ready-to-use
    `mem0.configs.base.MemoryConfig`, so callers do not need to care about
    import order. `vector_store` is the `AerospikeConfig` field dict
    (without a "provider" key); remaining kwargs are passed through to
    `MemoryConfig` (e.g. `llm=`, `embedder=`, `graph_store=`).
    """
    from mem0.configs.base import MemoryConfig

    # Provider-specific fields must be nested under "config": mem0's
    # VectorStoreConfig only forwards that dict to the provider model, so
    # top-level keys are silently dropped.
    vector_store = {"provider": "aerospike", "config": dict(vector_store or {})}
    return MemoryConfig(vector_store=vector_store, **kwargs)


__all__ = ["create_memory_config", "register_aerospike"]
