"""
fitsproof.mcp — Minimal MCP (Model Context Protocol) server exposing the resource contract.

Exposes three tools that an agent can call before deciding to load a model:
  - probe:  characterise the current machine, return JSON metrics.
  - plan:   predict resource usage for a given config; return verdict + predictions.
  - admit:  enforce the plan; return ADMITTED / DEGRADED / REFUSED with binding constraint.

Transport: stdout/stdin JSONRPC (MCP stdio transport) — the `fitsproof mcp` CLI subcommand
starts this server and the calling agent connects to the process stdio.

Protocol reference: https://modelcontextprotocol.io/specification/2025-03-26/ (Research source 23)
This implements the minimal subset required for tools: initialize, tools/list, tools/call.

Fault detected: a server that returns a tools/call response without checking whether the
plan was refused would allow an agent to proceed past a refusal silently. The admit tool
sets is_error=True and returns the refusal message on REFUSED status so the agent cannot
mistake a refusal for an admission.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from fitsproof.client import FitsproofClient
from fitsproof.contract.admit import AdmitStatus
from fitsproof.contract.admit import admit as _admit

# MCP protocol version we speak
_PROTOCOL_VERSION = "2025-03-26"

# Tool definitions for tools/list
_TOOLS: list[dict[str, Any]] = [
    {
        "name": "probe",
        "description": (
            "Characterise the current machine: memory bandwidth, GEMM throughput, "
            "available RAM, available VRAM. Returns a JSON object with metric keys."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "plan",
        "description": (
            "Predict peak memory usage and decode throughput for a given configuration. "
            "Returns verdict (fits/fits_with_degradation/does_not_fit), "
            "predicted_peak_gb, predicted_tok_s, and available degradation steps."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "context_len": {
                    "type": "integer",
                    "description": "Prompt + generation context length in tokens.",
                    "default": 512,
                },
                "budget": {
                    "type": "string",
                    "description": "Memory budget, e.g. '4GiB', '8GB', or integer bytes.",
                    "default": "4GiB",
                },
                "quant": {
                    "type": "string",
                    "description": "Quantisation: 'none', 'int8_sym', or 'int4_sym'.",
                    "default": "none",
                },
            },
            "required": [],
        },
    },
    {
        "name": "admit",
        "description": (
            "Enforce the resource contract. Returns ADMITTED, DEGRADED (with the exact "
            "degradation applied), or REFUSED (with the binding constraint named). "
            "Use this before loading a model to avoid silent OOM."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "context_len": {
                    "type": "integer",
                    "description": "Context length in tokens.",
                    "default": 512,
                },
                "budget": {
                    "type": "string",
                    "description": "Memory budget, e.g. '4GiB'.",
                    "default": "4GiB",
                },
                "quant": {
                    "type": "string",
                    "description": "Quantisation level.",
                    "default": "none",
                },
            },
            "required": [],
        },
    },
]


def _handle_request(req: dict[str, Any], client: FitsproofClient) -> dict[str, Any]:
    """
    Route one JSON-RPC request to the appropriate handler.
    Returns the JSON-RPC response dict.
    """
    method = req.get("method", "")
    req_id = req.get("id")
    params = req.get("params", {})

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": _PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "fitsproof-mcp", "version": "0.1.0"},
            },
        }

    elif method == "notifications/initialized":
        # Client acknowledges initialisation; no response required for notifications.
        return {}  # caller must skip empty responses

    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": _TOOLS},
        }

    elif method == "tools/call":
        tool_name = params.get("name")
        args = params.get("arguments", {})
        return _handle_tool_call(req_id, tool_name, args, client)

    else:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }


def _handle_tool_call(
    req_id: Any,
    tool_name: str | None,
    args: dict[str, Any],
    client: FitsproofClient,
) -> dict[str, Any]:
    """Dispatch tool calls and return a tools/call response."""

    def ok(text: str) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"content": [{"type": "text", "text": text}], "isError": False},
        }

    def err(text: str) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"content": [{"type": "text", "text": text}], "isError": True},
        }

    if tool_name == "probe":
        metrics = client.metrics()
        return ok(json.dumps(metrics, indent=2))

    elif tool_name == "plan":
        context_len = int(args.get("context_len", 512))
        budget = args.get("budget", "4GiB")
        quant = args.get("quant", "none")
        p = client.plan(context_len=context_len, budget_bytes=budget, quant=quant)
        result = {
            "verdict": p.verdict.value,
            "predicted_peak_gb": round(p.predicted_peak_bytes / 1e9, 4),
            "predicted_peak_ci_gb": [
                round(p.predicted_peak_ci[0] / 1e9, 4),
                round(p.predicted_peak_ci[1] / 1e9, 4),
            ],
            "predicted_tok_s": round(p.predicted_tok_s, 2),
            "budget_gb": round(p.budget_bytes / 1e9, 3),
            "degradations": [
                {
                    "kind": d.kind,
                    "description": d.description,
                    "predicted_peak_gb": round(d.predicted_peak_bytes / 1e9, 4),
                    "fits_budget": d.fits_budget,
                }
                for d in p.degradations
            ],
        }
        return ok(json.dumps(result, indent=2))

    elif tool_name == "admit":
        context_len = int(args.get("context_len", 512))
        budget = args.get("budget", "4GiB")
        quant = args.get("quant", "none")
        p = client.plan(context_len=context_len, budget_bytes=budget, quant=quant)
        record = _admit(p)
        result = {
            "status": record.status.value,
            "message": record.message,
        }
        if record.applied_degradation is not None:
            result["applied_degradation"] = record.applied_degradation.description
        if record.refusal_reason:
            result["refusal_reason"] = record.refusal_reason
        if record.status == AdmitStatus.REFUSED:
            return err(json.dumps(result, indent=2))
        return ok(json.dumps(result, indent=2))

    else:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32602, "message": f"Unknown tool: {tool_name}"},
        }


def run_mcp_server(
    stdin: Any = None,
    stdout: Any = None,
) -> None:
    """
    Run the MCP stdio server loop.

    Reads newline-delimited JSON-RPC from stdin, writes responses to stdout.
    Exits on EOF.

    In tests, pass synthetic stdin/stdout file-like objects for in-process testing.
    """
    if stdin is None:
        stdin = sys.stdin
    if stdout is None:
        stdout = sys.stdout

    client = FitsproofClient()

    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {exc}"},
            }
            stdout.write(json.dumps(resp) + "\n")
            stdout.flush()
            continue

        resp = _handle_request(req, client)
        if resp:  # skip empty (notifications)
            stdout.write(json.dumps(resp) + "\n")
            stdout.flush()
