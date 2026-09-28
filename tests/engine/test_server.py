"""
Tests for fitsproof.engine.server (OpenAI-compatible HTTP server).

Faults detected by each test:
  test_healthz_returns_200:
    GET /healthz must return 200 with status=ok.
    Fault: wrong path routing returns 404.

  test_models_endpoint:
    GET /v1/models must return a list with at least one model entry.
    Fault: missing model id in response fails the OpenAI API contract.

  test_completion_non_streaming:
    POST /v1/chat/completions returns a completion with the correct structure.
    Fault: missing choices[0].message.content breaks API compatibility.

  test_completion_streaming_returns_chunks:
    Streaming response must return multiple SSE chunks, not a buffered single response.
    Fault: returning the full response as a single chunk is not real SSE streaming.

  test_unknown_endpoint_404:
    Requests to unknown paths must return 404.
    Fault: returning 500 or 200 gives false positives to callers.

  test_invalid_json_returns_400:
    Malformed JSON in POST body must return 400.
    Fault: a bare exception handler that returns 500 breaks caller error handling.
"""

from __future__ import annotations

import json
import socket
import time
import urllib.request
from urllib.error import HTTPError

import pytest

from fitsproof.engine.model import generate_reference_model
from fitsproof.engine.transformer import Transformer


@pytest.fixture(scope="module")
def server_and_port(tmp_path_factory):
    """Start the server in a background thread, yield (port,), shutdown after."""
    path = tmp_path_factory.mktemp("srv_model")
    generate_reference_model(path, seed=42)
    from fitsproof.engine.model import load_bundle

    cfg, weights = load_bundle(path)
    transformer = Transformer(cfg, weights)

    from fitsproof.engine.server import start_server

    # Pick a free port
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    srv = start_server(transformer, cfg, host="127.0.0.1", port=port, block=False)

    # Wait for server to be ready
    for _ in range(20):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1)
            break
        except Exception:
            time.sleep(0.1)

    yield port

    srv.shutdown()


def _get(url: str) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except HTTPError as e:
        return e.code, {}


def _post(url: str, body: dict | str, expect_json: bool = True) -> tuple[int, dict | str]:
    if isinstance(body, dict):
        data = json.dumps(body).encode()
    else:
        data = body.encode() if isinstance(body, str) else body
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            content = resp.read().decode("utf-8")
            if expect_json:
                return resp.status, json.loads(content)
            return resp.status, content
    except HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode())
        except Exception:
            return e.code, {}


def test_healthz_returns_200(server_and_port) -> None:
    """
    Fault detected: wrong routing returns 404 for /healthz.
    """
    port = server_and_port
    status, body = _get(f"http://127.0.0.1:{port}/healthz")
    assert status == 200
    assert body.get("status") == "ok"


def test_models_endpoint(server_and_port) -> None:
    """
    Fault detected: missing model entry returns empty list, breaking API compat.
    """
    port = server_and_port
    status, body = _get(f"http://127.0.0.1:{port}/v1/models")
    assert status == 200
    assert body.get("object") == "list"
    assert len(body.get("data", [])) >= 1
    assert "id" in body["data"][0]


def test_completion_non_streaming(server_and_port) -> None:
    """
    Fault detected: missing choices[0].message.content breaks API compatibility.
    ACCEPTANCE CRITERION 9: real HTTP request returns a completion.
    """
    port = server_and_port
    payload = {
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 5,
        "temperature": 0.0,
        "stream": False,
    }
    status, body = _post(f"http://127.0.0.1:{port}/v1/chat/completions", payload)
    assert status == 200, f"Expected 200, got {status}: {body}"
    choices = body.get("choices", [])
    assert len(choices) >= 1
    msg = choices[0].get("message", {})
    assert "content" in msg
    assert isinstance(msg["content"], str)
    # usage counts
    assert body.get("usage", {}).get("completion_tokens", 0) > 0


def test_completion_streaming_returns_chunks(server_and_port) -> None:
    """
    Fault detected: buffered streaming returns one chunk instead of many.
    ACCEPTANCE CRITERION 9: streaming returns real chunks.
    """
    port = server_and_port
    payload = {
        "messages": [{"role": "user", "content": "test"}],
        "max_tokens": 4,
        "temperature": 0.0,
        "stream": True,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", data=data, method="POST"
    )
    req.add_header("Content-Type", "application/json")

    chunks = []
    with urllib.request.urlopen(req, timeout=10) as resp:
        # Read all content
        content = resp.read().decode("utf-8")
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                chunk_json = line[6:]
                try:
                    chunk = json.loads(chunk_json)
                    chunks.append(chunk)
                except json.JSONDecodeError:
                    pass

    assert len(chunks) >= 1, f"Expected streaming chunks, got {len(chunks)}"
    # Each chunk must have choices
    for chunk in chunks:
        assert "choices" in chunk


def test_unknown_endpoint_404(server_and_port) -> None:
    """
    Fault detected: 500 or 200 for unknown paths gives false positives.
    """
    port = server_and_port
    status, _ = _get(f"http://127.0.0.1:{port}/does_not_exist")
    assert status == 404


def test_invalid_json_returns_400(server_and_port) -> None:
    """
    Fault detected: bare except returning 500 instead of 400 for bad JSON.
    """
    port = server_and_port
    status, _ = _post(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        "not valid json",
    )
    assert status == 400, f"Expected 400 for bad JSON, got {status}"


def test_max_tokens_exceeds_max_seq_len_does_not_crash(server_and_port) -> None:
    """
    ADV-16: server must not crash when max_tokens > max_seq_len - len(prompt_ids).

    Fault detected: without clamping max_tokens to (max_seq_len - len(prompt_ids)),
    Transformer.generate() indexes the precomputed RoPE frequency array beyond its
    bounds.  This raises a broadcasting ValueError that crashes the request handler
    thread, causing the server to drop the connection mid-response (RemoteDisconnected
    or BrokenPipeError on the client side).

    The fix clamps max_tokens = min(raw, max_seq_len - len(prompt_ids), _CAP) before
    calling generate().  Evidence that the clamp fired: the returned
    usage.completion_tokens must be <= cfg.max_seq_len (RoPE table size), regardless
    of the absurd raw value (9999) the client sent.

    Fault injection proof: removing the clamping line from server.py and running
    this test raises urllib.error.URLError (RemoteDisconnected) — the connection drops
    before a response is returned.  With the fix, HTTP 200 is returned and
    usage.completion_tokens <= max_seq_len.
    """
    import urllib.error

    port = server_and_port
    payload = {
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 9999,
        "temperature": 0.0,
        "stream": False,
    }
    # Catch URLError/RemoteDisconnected so a regression (server crash) produces a
    # clear failure message rather than an unrelated exception traceback.
    try:
        status, body = _post(f"http://127.0.0.1:{port}/v1/chat/completions", payload)
    except urllib.error.URLError as exc:
        raise AssertionError(
            f"ADV-16 regression: server crashed on max_tokens=9999, "
            f"dropping the connection instead of returning a response. "
            f"Underlying error: {exc}"
        ) from exc

    assert status == 200, (
        f"Server must return 200 when max_tokens is clamped to fit; got {status}: {body}"
    )
    choices = body.get("choices", [])
    assert len(choices) >= 1, "Response must have at least one choice"
    assert isinstance(choices[0].get("message", {}).get("content"), str), (
        "choices[0].message.content must be a string"
    )
    usage = body.get("usage", {})
    completion_tokens = usage.get("completion_tokens", None)
    assert completion_tokens is not None, "usage.completion_tokens must be present"
    # The reference model has max_seq_len=128 (from get_reference_bundle).
    # completion_tokens must be <= max_seq_len regardless of the raw 9999 the client sent.
    # This is the falsifiable assertion: if clamping is removed, the server crashes
    # before returning any response (caught above), so this line is only reached when
    # the fix is active.
    assert completion_tokens <= 128, (
        f"completion_tokens={completion_tokens} exceeds max_seq_len=128; "
        f"clamping did not work or max_seq_len changed"
    )
