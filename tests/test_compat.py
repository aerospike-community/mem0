"""Version guard: unsupported mem0ai versions raise a clear ImportError."""

from importlib.metadata import PackageNotFoundError

import pytest

from mem0_aerospike import patch


def test_current_version_accepted():
    assert patch.check_mem0ai_version()  # returns version string


def test_unsupported_major_version_raises(mocker):
    mocker.patch("mem0_aerospike.patch.version", return_value="3.1.0")
    with pytest.raises(ImportError, match="mem0ai"):
        patch.check_mem0ai_version()


def test_too_old_version_raises(mocker):
    mocker.patch("mem0_aerospike.patch.version", return_value="1.9.9")
    with pytest.raises(ImportError, match="mem0ai"):
        patch.check_mem0ai_version()


def test_missing_mem0ai_raises(mocker):
    mocker.patch(
        "mem0_aerospike.patch.version",
        side_effect=PackageNotFoundError("mem0ai"),
    )
    with pytest.raises(ImportError, match="mem0ai"):
        patch.check_mem0ai_version()


def test_unparseable_version_raises(mocker):
    mocker.patch("mem0_aerospike.patch.version", return_value="not-a-version")
    with pytest.raises(ImportError, match="could not parse"):
        patch.check_mem0ai_version()


def test_prerelease_version_accepted(mocker):
    """Pre-releases inside the supported range must not be rejected."""
    mocker.patch("mem0_aerospike.patch.version", return_value="2.1.0a1")
    assert patch.check_mem0ai_version() == "2.1.0a1"


def test_register_propagates_version_error(mocker):
    mocker.patch("mem0_aerospike.patch.check_mem0ai_version", side_effect=ImportError("bad version"))
    from mem0_aerospike.patch import register_aerospike

    with pytest.raises(ImportError, match="bad version"):
        register_aerospike()
