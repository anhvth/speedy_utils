"""Process-local SSH forwards for bare ``HOST:PORT`` LLM endpoints."""

from __future__ import annotations

import atexit
import os
import socket
import subprocess
import threading
import time
from urllib.parse import urlsplit


_STARTUP_TIMEOUT = 15.0
_tunnels: dict[tuple[str, int], tuple[int, subprocess.Popen[bytes]]] = {}
_tunnels_lock = threading.Lock()
_tunnel_owner_pid = os.getpid()


def resolve_ssh_endpoint(endpoint: str) -> str:
    """Resolve a bare SSH ``HOST:PORT`` endpoint to a shared local URL.

    HTTP(S) URLs are returned unchanged. A forward is retained for this Python
    process and is reused by every later request for the same SSH target.
    """
    target = _parse_ssh_endpoint(endpoint)
    if target is None:
        return endpoint
    host, remote_port = target
    local_port = _ensure_tunnel(host, remote_port)
    return f"http://127.0.0.1:{local_port}/v1"


def _parse_ssh_endpoint(endpoint: str) -> tuple[str, int] | None:
    parsed_url = urlsplit(endpoint)
    if parsed_url.scheme in {"http", "https"}:
        return None
    if "://" in endpoint:
        raise ValueError(
            "client URLs must use http(s); bare SSH endpoints use HOST:PORT."
        )

    parsed = urlsplit(f"ssh://{endpoint}")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"Invalid SSH endpoint {endpoint!r}") from exc
    host = parsed.hostname
    if not host or port is None or parsed.path or parsed.query or parsed.fragment:
        raise ValueError(
            "client must be an http(s) URL or a bare SSH endpoint HOST:PORT."
        )
    return host, port


def _ensure_tunnel(host: str, remote_port: int) -> int:
    key = (host, remote_port)
    with _tunnels_lock:
        existing = _tunnels.get(key)
        if existing is not None and existing[1].poll() is None:
            return existing[0]
        if existing is not None:
            _tunnels.pop(key, None)

        local_port = _reserve_local_port()
        process = subprocess.Popen(
            _ssh_command(host, remote_port, local_port),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            _wait_for_listener(process, local_port)
        except Exception:
            _stop_process(process)
            raise
        _tunnels[key] = (local_port, process)
        return local_port


def _reserve_local_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_listener(process: subprocess.Popen[bytes], port: int) -> None:
    deadline = time.monotonic() + _STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        status = process.poll()
        if status is not None:
            raise RuntimeError(
                f"ssh exited before forwarding local port {port} (status {status})"
            )
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                return
        except OSError:
            time.sleep(0.05)
    raise RuntimeError(f"Timed out waiting for SSH to forward local port {port}")


def _ssh_command(host: str, remote_port: int, local_port: int) -> list[str]:
    return [
        "ssh",
        "-N",
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "ControlMaster=no",
        "-o",
        "ControlPath=none",
        "-o",
        "ControlPersist=no",
        "-o",
        "ConnectTimeout=10",
        "-L",
        f"127.0.0.1:{local_port}:127.0.0.1:{remote_port}",
        host,
    ]


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        process.terminate()


def _close_tunnels() -> None:
    # Forked workers inherit the registry, but the forwarding processes still
    # belong to the parent.  A child exiting must never terminate them.
    if os.getpid() != _tunnel_owner_pid:
        return
    with _tunnels_lock:
        tunnels = list(_tunnels.values())
        _tunnels.clear()
    for _, process in tunnels:
        _stop_process(process)


atexit.register(_close_tunnels)
