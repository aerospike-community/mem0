"""Integration tests require a reachable Aerospike server."""

import os
import socket

import pytest

HOST = os.environ.get("AEROSPIKE_HOST", "localhost")
PORT = int(os.environ.get("AEROSPIKE_PORT", "3000"))


def _server_reachable() -> bool:
    try:
        with socket.create_connection((HOST, PORT), timeout=2):
            return True
    except OSError:
        return False


if not _server_reachable():
    collect_ignore_glob = ["*"]
    pytestmark = pytest.mark.skip(reason=f"no Aerospike server at {HOST}:{PORT}")
