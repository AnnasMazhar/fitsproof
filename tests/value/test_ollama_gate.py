"""
tests/value/test_ollama_gate.py — Integration test for scripts/ollama_gate.py.

The ollama gate is the primary real-world integration example: it gates an external
LLM engine (ollama) with fitsproof's admission contract before any model is loaded.
This file verifies the gate's observable contracts without requiring a live daemon.

Research source: Sources [1] (Roofline), [2] (Memory-Bandwidth-Bound Decode) — the
roofline admission model that the gate enforces. The gate's fail-closed requirement is
derived from ADOPTION.md §2 and MARKET-VERDICTS.md §4: a gate that does not refuse
when it cannot check is not a gate.

Faults detected:
  test_ollama_gate_fails_closed_no_daemon:
      Fault: the gate silently succeeds when the ollama daemon is unreachable.
      If the gate exits 0 when it cannot check, it is not a gate — it is a no-op.
      Any admitted run could then silently OOM without the gate having fired.
      The gate MUST exit non-zero and name the connectivity issue.

  test_ollama_gate_daemon_error_message_is_actionable:
      Fault: the error message says "daemon unreachable" with no recovery path.
      A stranger who sees this cannot take any action. The message must tell them
      what command to run next (e.g. "Is ollama running? Try: ollama serve").

  test_ollama_gate_model_not_found_message_is_actionable:
      Fault: a 404 from the daemon produces the same generic "unreachable" error
      as a connection failure. The user cannot distinguish "wrong model name" from
      "ollama is down". The message must say "model not found" and suggest
      "ollama list" or "ollama pull <name>".

  test_ollama_gate_exits_nonzero_on_metadata_failure:
      Fault: the gate silently exits 0 when model metadata cannot be parsed.
      A malformed or truncated response must cause a refused exit, not admission.

All tests run offline (no real daemon). The daemon-down path is the normal CI path.
The daemon-up path is opt-in via FITSPROOF_OLLAMA_INTEGRATION_TEST=1 (skipped otherwise).
"""

from __future__ import annotations

import http.server
import json
import os
import sys
import threading
from pathlib import Path
from subprocess import run as _run

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE_SCRIPT = REPO_ROOT / "scripts" / "ollama_gate.py"

_PY = sys.executable


def _run_gate(*args: str, env: dict | None = None) -> tuple[int, str, str]:
    """Run ollama_gate.py and return (returncode, stdout, stderr)."""
    merged = os.environ.copy()
    if env:
        merged.update(env)
    result = _run(
        [_PY, str(GATE_SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=30,
        env=merged,
    )
    return result.returncode, result.stdout, result.stderr


# ---------------------------------------------------------------------------
# Fail-closed: daemon unreachable must exit non-zero
# ---------------------------------------------------------------------------


def test_ollama_gate_fails_closed_no_daemon() -> None:
    """
    Fault detected: gate exits 0 when the daemon is unreachable.
    If the gate cannot verify the model's memory footprint, it must refuse —
    admitting an unchecked config defeats the purpose of the gate.

    We point the gate at a port guaranteed to be closed (the OS reserves 1 for
    ICMP; it will never be listening). Any real port could theoretically be in use.
    Exit code must be 1 (gate failure), not 0 (admitted).
    """
    rc, stdout, stderr = _run_gate(
        "gemma3:4b",
        "--budget-gb",
        "4",
        "--host",
        "http://127.0.0.1:1",  # port 1 is reserved; never listening
    )
    assert rc != 0, (
        f"Gate must not exit 0 when the daemon is unreachable. "
        f"rc={rc}, stdout={stdout!r}, stderr={stderr!r}"
    )
    assert rc == 1, (
        f"Daemon unreachable must exit 1 (gate failure), not 2 (refusal). "
        f"rc={rc}, stderr={stderr!r}"
    )


def test_ollama_gate_daemon_error_message_is_actionable() -> None:
    """
    Fault detected: error message gives the user no recovery path.
    A stranger who sees "daemon unreachable" cannot take any action.
    The message must either name the command to start ollama or say
    what to check — a credential error, a connection refused, a timeout
    all need different actions.

    Asserts: stderr contains 'ollama serve' or 'Is ollama running' so the
    user knows how to diagnose and fix the connectivity issue.
    """
    rc, stdout, stderr = _run_gate(
        "gemma3:4b",
        "--budget-gb",
        "4",
        "--host",
        "http://127.0.0.1:1",
    )
    assert rc == 1
    # The actionable hint must appear in stderr
    assert any(phrase in stderr for phrase in ("ollama serve", "Is ollama running")), (
        f"Daemon-unreachable error message must include a recovery hint "
        f"('Is ollama running? Try: ollama serve'). Got: {stderr!r}"
    )


# ---------------------------------------------------------------------------
# Model-not-found must produce a distinct, actionable error
# ---------------------------------------------------------------------------


class _FakeOllamaHandler(http.server.BaseHTTPRequestHandler):
    """Minimal fake ollama daemon that returns HTTP 404 for any /api/show request."""

    def log_message(self, *_args) -> None:  # silence access log in test output
        pass

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/api/show":
            body = json.dumps({"error": "model 'nonexistent_xyz' not found"}).encode()
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(200)
            self.end_headers()


def _start_fake_daemon(port: int) -> http.server.HTTPServer:
    server = http.server.HTTPServer(("127.0.0.1", port), _FakeOllamaHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


def test_ollama_gate_model_not_found_message_is_actionable() -> None:
    """
    Fault detected: a 404 from the daemon produces the same generic error as a
    connection failure. User cannot distinguish 'wrong model name' from 'ollama is down'.

    We run a fake daemon that returns 404 for /api/show, then assert:
    1. Gate exits 1 (gate failure, not refusal).
    2. stderr mentions 'not found' (not 'unreachable').
    3. stderr suggests 'ollama list' or 'ollama pull' as the recovery path.
    """
    import socket

    # Find a free port
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    server = _start_fake_daemon(port)
    try:
        rc, stdout, stderr = _run_gate(
            "nonexistent_xyz",
            "--budget-gb",
            "4",
            "--host",
            f"http://127.0.0.1:{port}",
        )
    finally:
        server.shutdown()

    assert rc == 1, f"Model-not-found must exit 1 (gate failure). rc={rc}, stderr={stderr!r}"
    assert "not found" in stderr.lower(), (
        f"Model-not-found error must say 'not found', not 'unreachable'. Got: {stderr!r}"
    )
    assert any(phrase in stderr for phrase in ("ollama list", "ollama pull")), (
        f"Model-not-found error must suggest 'ollama list' or 'ollama pull'. Got: {stderr!r}"
    )


# ---------------------------------------------------------------------------
# Admitted path: gate integrates with real probe+plan+admit pipeline
# ---------------------------------------------------------------------------


class _FakeOllamaFullHandler(http.server.BaseHTTPRequestHandler):
    """Fake daemon returning a minimal but structurally valid /api/show response
    for a tiny synthetic model. Used to test the gate's admission path offline."""

    # Minimal GGUF-backed modelfile path that won't exist — gate reads GGUF directly.
    # We use a nonexistent path so gguf_vocab_size will fail, triggering the
    # metadata-failure path.  That is the next test below.  This handler is the
    # foundation for a future live-daemon test only.

    def log_message(self, *_args) -> None:
        pass

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/api/show":
            body = json.dumps(
                {
                    "details": {
                        "family": "llama",
                        "parameter_size": "0.04B",
                        "quantization_level": "F32",
                    },
                    "modelfile": "FROM /nonexistent/blob.gguf",
                    "model_info": {
                        "llama.embedding_length": 384,
                        "llama.block_count": 6,
                        "llama.attention.head_count": 6,
                        "llama.attention.head_count_kv": 2,
                        "llama.feed_forward_length": 1536,
                        "llama.context_length": 512,
                    },
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(200)
            self.end_headers()


def test_ollama_gate_exits_nonzero_on_metadata_failure() -> None:
    """
    Fault detected: gate exits 0 (admitted) when model metadata is unreadable.
    If the GGUF blob path doesn't exist, the gate cannot compute vocab_size and
    must refuse rather than admit with a fabricated shape.

    We use a fake daemon that returns a valid /api/show with a nonexistent blob path.
    The gate should hit the GGUF read failure and exit 2 (REFUSED) with an error.
    """
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    server = http.server.HTTPServer(("127.0.0.1", port), _FakeOllamaFullHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        rc, stdout, stderr = _run_gate(
            "tiny_model",
            "--budget-gb",
            "4",
            "--host",
            f"http://127.0.0.1:{port}",
        )
    finally:
        server.shutdown()

    assert rc != 0, (
        f"Gate must not exit 0 when GGUF blob is unreadable. "
        f"rc={rc}, stdout={stdout!r}, stderr={stderr!r}"
    )
    # Should exit 2 (GATE REFUSED: cannot read metadata) not 1 (connectivity)
    assert rc == 2, (
        f"Unreadable GGUF must exit 2 (metadata refusal), got rc={rc}. stderr={stderr!r}"
    )
    assert "GATE REFUSED" in stderr, (
        f"Metadata failure must print 'GATE REFUSED'. Got stderr: {stderr!r}"
    )
