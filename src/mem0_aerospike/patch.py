"""Runtime registration of the Aerospike vector-store provider with mem0ai.

mem0ai has no public vector-store plugin API: `VectorStoreConfig` hardcodes
`mem0.configs.vector_stores.<provider>` and `VectorStoreFactory` hardcodes
`mem0.vector_stores.<provider>`. This module installs the package's own
source files into `sys.modules` under those exact names and registers
"aerospike" in both provider registries. No mem0ai source files are modified.
"""

import re
import sys
from importlib.machinery import SourceFileLoader
from importlib.metadata import PackageNotFoundError, version
from importlib.util import module_from_spec, spec_from_loader
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parent

PROVIDER = "aerospike"
CONFIG_MODULE_NAME = "mem0.configs.vector_stores.aerospike"
STORE_MODULE_NAME = "mem0.vector_stores.aerospike"
CONFIG_CLASS_NAME = "AerospikeConfig"
STORE_CLASS_PATH = "mem0.vector_stores.aerospike.AerospikeDB"


def _install_module(module_name: str, path: Path):
    """Execute `path` as `module_name` and bind it in sys.modules and on its parent."""
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing

    loader = SourceFileLoader(module_name, str(path))
    spec = spec_from_loader(module_name, loader)
    module = module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise

    parent_name, _, attr = module_name.rpartition(".")
    if parent_name:
        parent = sys.modules.get(parent_name)
        if parent is None:
            parent = __import__(parent_name, fromlist=["__name__"])
        setattr(parent, attr, module)
    return module


def _ensure_registration(registry: dict, provider: str, expected: str, registry_name: str):
    """Idempotently set registry[provider] = expected; raise on foreign registrations."""
    current = registry.get(provider)
    if current is None:
        registry[provider] = expected
    elif current != expected:
        raise ImportError(
            f"mem0-aerospike cannot register provider {provider!r}: {registry_name} already maps it "
            f"to {current!r} (expected {expected!r}). Another package may have claimed this provider name."
        )


def _config_registry() -> dict:
    """Return the mutable provider->config-class dict behind `VectorStoreConfig._provider_configs`.

    In pydantic v2 the underscore-prefixed class attribute becomes a
    `ModelPrivateAttr`; each instance deep-copies its `default`, so mutating
    the default dict registers the provider for all future instances. On
    versions where it is a plain class attribute, mutate it directly.
    """
    from mem0.vector_stores.configs import VectorStoreConfig

    attr = VectorStoreConfig.__private_attributes__.get("_provider_configs")
    if attr is not None and isinstance(getattr(attr, "default", None), dict):
        return attr.default
    attr = getattr(VectorStoreConfig, "_provider_configs", None)
    if isinstance(attr, dict):
        return attr
    raise ImportError(
        "mem0-aerospike cannot find a mutable VectorStoreConfig._provider_configs registry "
        "on this mem0ai version; mem0ai internals may have changed."
    )


# mem0ai versions this package has been tested against. The registration shim
# depends on mem0ai internals (`VectorStoreConfig._provider_configs`,
# `VectorStoreFactory.provider_to_class`, and the hardcoded
# `mem0.configs.vector_stores.<provider>` / `mem0.vector_stores.<provider>`
# module paths), so out-of-range versions raise instead of failing silently.
MEM0AI_MIN = (2, 0, 0)
MEM0AI_MAX_EXCLUSIVE = (3, 0, 0)
SUPPORTED_MEM0AI_RANGE = ">=2.0.0,<3.0.0"


def _parse_version(value: str) -> tuple:
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", value)
    if not match:
        raise ImportError(
            f"mem0-aerospike could not parse the installed mem0ai version {value!r}; "
            f"supported range is {SUPPORTED_MEM0AI_RANGE}."
        )
    return tuple(int(part) for part in match.groups())


def check_mem0ai_version() -> str:
    """Return the installed mem0ai version, or raise ImportError if unsupported."""
    try:
        installed = version("mem0ai")
    except PackageNotFoundError as e:
        raise ImportError("mem0-aerospike requires the 'mem0ai' package. Install it with: pip install mem0ai") from e

    parsed = _parse_version(installed)
    if not (MEM0AI_MIN <= parsed < MEM0AI_MAX_EXCLUSIVE):
        raise ImportError(
            f"mem0-aerospike supports mem0ai {SUPPORTED_MEM0AI_RANGE} but found {installed}. "
            "mem0ai internals may have changed; refusing to register the Aerospike provider."
        )
    return installed


def register_aerospike():
    """Install the fake mem0 modules and register 'aerospike' in mem0ai's registries."""
    check_mem0ai_version()

    from mem0.utils.factory import VectorStoreFactory

    _install_module(
        CONFIG_MODULE_NAME,
        _PACKAGE_ROOT / "configs" / "vector_stores" / "aerospike.py",
    )
    _install_module(
        STORE_MODULE_NAME,
        _PACKAGE_ROOT / "vector_stores" / "aerospike.py",
    )

    _ensure_registration(
        _config_registry(),
        PROVIDER,
        CONFIG_CLASS_NAME,
        "VectorStoreConfig._provider_configs",
    )
    _ensure_registration(
        VectorStoreFactory.provider_to_class,
        PROVIDER,
        STORE_CLASS_PATH,
        "VectorStoreFactory.provider_to_class",
    )
