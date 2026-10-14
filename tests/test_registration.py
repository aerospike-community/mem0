"""Registration tests: import mem0_aerospike installs fake mem0 modules and
mutates mem0ai registries so provider="aerospike" resolves."""

import sys

import pytest
from mem0.utils.factory import VectorStoreFactory
from mem0.vector_stores.configs import VectorStoreConfig

import mem0_aerospike
from mem0_aerospike import patch


def test_fake_modules_installed():
    assert "mem0.configs.vector_stores.aerospike" in sys.modules
    assert "mem0.vector_stores.aerospike" in sys.modules


def test_provider_config_registry_contains_aerospike():
    registry = patch._config_registry()
    assert registry["aerospike"] == "AerospikeConfig"


def test_factory_registry_contains_aerospike():
    assert VectorStoreFactory.provider_to_class["aerospike"] == "mem0.vector_stores.aerospike.AerospikeDB"


def test_registration_is_idempotent():
    before = dict(patch._config_registry())
    factory_before = dict(VectorStoreFactory.provider_to_class)
    mem0_aerospike.register_aerospike()
    mem0_aerospike.register_aerospike()
    assert patch._config_registry() == before
    assert VectorStoreFactory.provider_to_class == factory_before


def test_config_validation_resolves_aerospike_config():
    cfg = VectorStoreConfig(
        provider="aerospike",
        config={"namespace": "test", "collection_name": "mem0", "embedding_model_dims": 1536},
    )
    from mem0.configs.vector_stores.aerospike import AerospikeConfig

    assert isinstance(cfg.config, AerospikeConfig)


def test_import_path_resolves_via_mem0_namespace():
    # The exact __import__ path mem0ai uses internally.
    module = __import__("mem0.configs.vector_stores.aerospike", fromlist=["AerospikeConfig"])
    assert hasattr(module, "AerospikeConfig")
    store_module = __import__("mem0.vector_stores.aerospike", fromlist=["AerospikeDB"])
    assert hasattr(store_module, "AerospikeDB")


def test_conflict_detection_config_registry():
    registry = patch._config_registry()
    original = registry["aerospike"]
    try:
        registry["aerospike"] = "SomeOtherConfig"
        with pytest.raises(ImportError, match="already maps it"):
            mem0_aerospike.register_aerospike()
    finally:
        registry["aerospike"] = original


def test_conflict_detection_factory_registry():
    original = VectorStoreFactory.provider_to_class["aerospike"]
    try:
        VectorStoreFactory.provider_to_class["aerospike"] = "some.other.Class"
        with pytest.raises(ImportError, match="already maps it"):
            mem0_aerospike.register_aerospike()
    finally:
        VectorStoreFactory.provider_to_class["aerospike"] = original
