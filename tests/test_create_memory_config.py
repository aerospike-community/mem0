"""create_memory_config returns a ready-to-use MemoryConfig on provider aerospike."""

from mem0.configs.base import MemoryConfig

from mem0_aerospike import create_memory_config


def test_helper_returns_memory_config_with_aerospike():
    cfg = create_memory_config(
        vector_store={"namespace": "test", "collection_name": "mem0", "embedding_model_dims": 768, "port": 3100}
    )
    assert isinstance(cfg, MemoryConfig)
    assert cfg.vector_store.provider == "aerospike"

    from mem0.configs.vector_stores.aerospike import AerospikeConfig

    assert isinstance(cfg.vector_store.config, AerospikeConfig)
    assert cfg.vector_store.config.namespace == "test"
    # Non-default values must actually reach the provider config (not be
    # dropped by VectorStoreConfig, which only forwards the "config" dict).
    assert cfg.vector_store.config.port == 3100
    assert cfg.vector_store.config.embedding_model_dims == 768


def test_helper_defaults_vector_store():
    cfg = create_memory_config()
    assert cfg.vector_store.provider == "aerospike"
    assert cfg.vector_store.config.collection_name == "mem0"


def test_helper_forwards_all_provider_fields():
    """Every AerospikeConfig field passed by the caller must reach the provider
    config — VectorStoreConfig drops anything not nested under "config"."""
    fields = {
        "namespace": "demo-ns",
        "collection_name": "demo-set",
        "embedding_model_dims": 768,
        "host": "aerospike.internal",
        "port": 3100,
        "allow_scans_with_where": True,
        "max_record_bytes": 1024,
    }
    cfg = create_memory_config(vector_store=fields)
    provider_cfg = cfg.vector_store.config
    for key, expected in fields.items():
        assert getattr(provider_cfg, key) == expected, f"{key} was not forwarded"


def test_config_survives_trip_into_constructed_store(mocker):
    """The caller's fields must reach the actual AerospikeDB constructor.

    Regression test: create_memory_config previously splatted fields at the top
    level of the vector_store dict, where VectorStoreConfig silently dropped
    them — Memory would then build the store with pure defaults (localhost:3000).
    """
    import sys

    from mem0.utils.factory import VectorStoreFactory

    # The package registers itself under mem0's hardcoded module name; the
    # factory load_class() resolves that module, which is a distinct module
    # object from mem0_aerospike.vector_stores.aerospike.
    store_module = sys.modules["mem0.vector_stores.aerospike"]
    ctor = mocker.patch.object(store_module.AerospikeDB, "__init__", return_value=None)

    fields = {
        "namespace": "demo-ns",
        "collection_name": "demo-set",
        "embedding_model_dims": 768,
        "host": "aerospike.internal",
        "port": 3100,
    }
    cfg = create_memory_config(vector_store=fields)
    # This is exactly what Memory.__init__ does with the validated config.
    VectorStoreFactory.create("aerospike", cfg.vector_store.config)

    ctor.assert_called_once()
    # model_dump() also passes defaulted fields, so assert the caller's
    # values arrived rather than requiring exact kwargs.
    assert fields.items() <= ctor.call_args.kwargs.items()


def test_helper_passes_through_other_sections():
    cfg = create_memory_config(
        vector_store={"namespace": "test"},
        llm={"provider": "openai", "config": {"model": "gpt-4o"}},
        embedder={"provider": "openai", "config": {"model": "text-embedding-3-small"}},
    )
    assert cfg.llm.provider == "openai"
    assert cfg.embedder.provider == "openai"
