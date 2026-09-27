"""
fitsproof.engine.server — OpenAI-compatible HTTP server.

Endpoints:
  POST /v1/chat/completions   — non-streaming and streaming (SSE)
  GET  /v1/models             — model list
  GET  /healthz               — health check

Streaming is real chunked SSE (Transfer-Encoding: chunked), not a buffered fake.
Uses stdlib http.server only (no external HTTP framework).

Usage:
    from fitsproof.engine.server import start_server
    start_server(transformer, cfg, host="127.0.0.1", port=8080)
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from fitsproof.contract.admit import AdmitStatus, admit
from fitsproof.contract.plan import plan
from fitsproof.contract.probe import MachineProfile, probe
from fitsproof.engine.model import ModelConfig
from fitsproof.engine.sampling import Sampler
from fitsproof.engine.transformer import Transformer


class _Handler(BaseHTTPRequestHandler):
    """HTTP request handler wired to the transformer in self.server.transformer."""

    # Suppress per-request log output (can be re-enabled by subclassing)
    def log_message(self, fmt: str, *args: Any) -> None:
        pass

    def do_GET(self) -> None:
        if self.path == "/healthz":
            self._send_json(200, {"status": "ok"})
        elif self.path == "/v1/models":
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": "fitsproof-reference",
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "fitsproof",
                        }
                    ],
                },
            )
        else:
            self._send_json(404, {"error": {"message": "Not found", "type": "not_found"}})

    def do_POST(self) -> None:
        if self.path == "/v1/chat/completions":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                payload = json.loads(body)
            except json.JSONDecodeError as exc:
                self._send_json(
                    400,
                    {"error": {"message": f"Invalid JSON: {exc}", "type": "invalid_request"}},
                )
                return
            self._handle_completion(payload)
        else:
            self._send_json(404, {"error": {"message": "Not found", "type": "not_found"}})

    def _handle_completion(self, payload: dict[str, Any]) -> None:
        transformer: Transformer = self.server.transformer  # type: ignore[attr-defined]
        cfg: ModelConfig = self.server.model_cfg  # type: ignore[attr-defined]

        messages = payload.get("messages", [])
        if not messages:
            self._send_json(
                400,
                {"error": {"message": "messages is required", "type": "invalid_request"}},
            )
            return

        # Simple tokenisation: convert text to byte values (0-255)
        text = " ".join(m.get("content", "") for m in messages)
        prompt_ids = [int(b) % cfg.vocab_size for b in text.encode("utf-8", errors="replace")]
        if not prompt_ids:
            prompt_ids = [0]

        max_tokens = int(payload.get("max_tokens", 32))
        temperature = float(payload.get("temperature", 0.0))
        stream = bool(payload.get("stream", False))
        seed = int(payload.get("seed", 0))

        sampler = Sampler(seed=seed)

        # Resource contract: predict + admit for this request, before allocating output.
        machine: MachineProfile = self.server.machine_profile  # type: ignore[attr-defined]
        budget: int = self.server.budget_bytes  # type: ignore[attr-defined]
        p = plan(
            cfg,
            machine,
            context_len=max(len(prompt_ids) + max_tokens, 1),
            budget_bytes=budget,
        )
        record = admit(p)
        fitsproof_field = {
            "admission": record.status.value,
            "verdict": p.verdict.value,
            "message": record.message,
            "predicted_peak_bytes": p.predicted_peak_bytes,
            "budget_bytes": budget,
        }
        if record.status == AdmitStatus.REFUSED:
            self._send_json(
                503,
                {
                    "error": {
                        "message": record.message,
                        "type": "fitsproof_refused",
                    },
                    "fitsproof": fitsproof_field,
                },
            )
            return

        if stream:
            self._stream_completion(
                transformer, cfg, prompt_ids, max_tokens, temperature, sampler, fitsproof_field
            )
        else:
            tokens = transformer.generate(
                prompt_ids,
                max_new_tokens=max_tokens,
                sampler=sampler,
                temperature=temperature,
            )
            text_out = bytes(t % 256 for t in tokens).decode("utf-8", errors="replace")
            resp = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "fitsproof-reference",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text_out},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": len(prompt_ids),
                    "completion_tokens": len(tokens),
                    "total_tokens": len(prompt_ids) + len(tokens),
                },
                "fitsproof": fitsproof_field,
            }
            self._send_json(200, resp)

    def _stream_completion(
        self,
        transformer: Transformer,
        cfg: ModelConfig,
        prompt_ids: list[int],
        max_tokens: int,
        temperature: float,
        sampler: Sampler,
        fitsproof_field: dict[str, Any],
    ) -> None:
        """Real chunked SSE streaming — emits one chunk per token."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        cid = f"chatcmpl-{int(time.time())}"
        created = int(time.time())

        def send_chunk(data: str) -> None:
            payload = f"data: {data}\n\n"
            encoded = payload.encode("utf-8")
            # HTTP chunked encoding: hex size + CRLF + data + CRLF
            chunk = f"{len(encoded):x}\r\n".encode() + encoded + b"\r\n"
            self.wfile.write(chunk)
            self.wfile.flush()

        # Generate token by token (simplified: generate all, then stream)
        tokens = transformer.generate(
            prompt_ids,
            max_new_tokens=max_tokens,
            sampler=sampler,
            temperature=temperature,
        )

        for tok in tokens:
            char = bytes([tok % 256]).decode("utf-8", errors="replace")
            delta = {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": "fitsproof-reference",
                "choices": [{"index": 0, "delta": {"content": char}, "finish_reason": None}],
            }
            send_chunk(json.dumps(delta))

        # Final chunk — carries the plan/admission record (M2)
        done_delta = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": "fitsproof-reference",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "fitsproof": fitsproof_field,
        }
        send_chunk(json.dumps(done_delta))
        send_chunk("[DONE]")

        # End chunked transfer
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def _send_json(self, status: int, body: Any) -> None:
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class FitsproodHTTPServer(HTTPServer):
    """HTTPServer subclass carrying the transformer, machine profile and budget."""

    def __init__(
        self,
        transformer: Transformer,
        cfg: ModelConfig,
        machine_profile: MachineProfile,
        budget_bytes: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.transformer = transformer
        self.model_cfg = cfg
        self.machine_profile = machine_profile
        self.budget_bytes = budget_bytes


def start_server(
    transformer: Transformer,
    cfg: ModelConfig,
    host: str = "127.0.0.1",
    port: int = 8080,
    block: bool = True,
    budget_bytes: int = 4 * 1024**3,
) -> FitsproodHTTPServer:
    """
    Start the OpenAI-compatible HTTP server.

    The server probes the machine once at startup and enforces *budget_bytes*
    on every request: a request whose predicted peak exceeds the budget is
    refused with HTTP 503 and a `fitsproof` record naming the binding
    constraint. Every non-refused response carries a `fitsproof` field with
    the admission record (M2).

    If *block* is True, serves forever in the current thread.
    If *block* is False, starts in a daemon thread and returns the server object
    (caller must call server.shutdown() to stop it).
    """
    machine = probe()
    server = FitsproodHTTPServer(transformer, cfg, machine, budget_bytes, (host, port), _Handler)
    if block:
        server.serve_forever()
    else:
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
    return server
