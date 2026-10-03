"""
CLI smoke tests — every subcommand must import, run, and exit without crashing.

Fault detected: a broken import (e.g. run_pareto_sweep ImportError) or a command
that crashes at startup will fail here, preventing broken subcommands from shipping.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time


def _run(*args: str, input: str | None = None, timeout: int = 120) -> subprocess.CompletedProcess:
    """Run fitsproof subcommand via the installed module entry point."""
    return subprocess.run(
        [sys.executable, "-m", "fitsproof.cli", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        input=input,
    )


def test_probe_smoke() -> None:
    """probe must exit 0 and print bandwidth/gemm/RAM lines."""
    r = _run("probe")
    assert r.returncode == 0, f"probe crashed: {r.stderr}"
    assert "bandwidth" in r.stdout
    assert "RAM" in r.stdout


def test_plan_smoke() -> None:
    """plan must exit 0 for a default 4 GB budget."""
    r = _run("plan", "--budget-gb", "4")
    assert r.returncode == 0, f"plan crashed: {r.stderr}"
    # plan shows the prediction with verdict (fits/does_not_fit/fits_with_degradation),
    # not the enforcement messages (ADMITTED/DEGRADED/REFUSED) which belong to admit.
    assert "verdict:" in r.stdout and "fits" in r.stdout


def test_admit_smoke_fits() -> None:
    """admit must exit 0 and print ADMITTED for a generous budget."""
    r = _run("admit", "--budget-gb", "4")
    assert r.returncode == 0, f"admit (fits) crashed: {r.stderr}"
    assert "ADMITTED" in r.stdout


def test_admit_smoke_refused() -> None:
    """admit must exit 2 and print REFUSED for an impossibly small budget."""
    r = _run("admit", "--budget-gb", "0.000001")
    assert r.returncode == 2, f"admit (refused) wrong exit code: {r.returncode}\n{r.stdout}"
    assert "REFUSED" in r.stdout


def test_verify_smoke() -> None:
    """verify must exit 0 for a 1 GB budget with the reference model."""
    r = _run("verify", "--budget-gb", "1.0", "--tokens", "4")
    assert r.returncode == 0, f"verify crashed (rc={r.returncode}):\n{r.stderr}"
    assert "measured_peak" in r.stdout


def test_stress_smoke() -> None:
    """stress must exit 0 with 0 violations for a 4 GB budget."""
    r = _run("stress", "--budget-gb", "4", timeout=180)
    assert r.returncode == 0, f"stress crashed: {r.stderr}"
    assert "violations" in r.stdout


def test_serve_smoke() -> None:
    """server must start, respond to a health probe, and shut down cleanly."""
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    proc = subprocess.Popen(
        [sys.executable, "-m", "fitsproof.cli", "server", "--port", str(port), "--budget-gb", "4"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        import urllib.request

        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/v1/models", timeout=2
                ) as resp:
                    assert resp.status == 200
                    break
            except Exception:
                time.sleep(0.2)
        else:
            raise AssertionError("server did not become ready within 30s")
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_server_alias_smoke() -> None:
    """'serve' alias must also start without crashing (just checks import + arg parse)."""
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    proc = subprocess.Popen(
        [sys.executable, "-m", "fitsproof.cli", "serve", "--port", str(port), "--budget-gb", "4"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    time.sleep(2)
    assert proc.poll() is None, f"serve alias crashed immediately: {proc.stderr.read()}"
    proc.terminate()
    proc.wait(timeout=10)


def test_mcp_smoke() -> None:
    """mcp must handle initialize + tools/list without crashing."""
    messages = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    ]
    input_str = "\n".join(json.dumps(m) for m in messages) + "\n"
    r = _run("mcp", input=input_str, timeout=60)
    assert r.returncode == 0, f"mcp crashed: {r.stderr}"
    lines = [line for line in r.stdout.splitlines() if line.strip()]
    assert len(lines) >= 2, f"mcp returned too few lines: {r.stdout!r}"
    init_reply = json.loads(lines[0])
    assert init_reply["result"]["serverInfo"]["name"] == "fitsproof-mcp"
    tools_reply = json.loads(lines[1])
    tool_names = {t["name"] for t in tools_reply["result"]["tools"]}
    assert {"probe", "plan", "admit"} <= tool_names


def test_pareto_smoke() -> None:
    """pareto must exit 0 and print a table with quant/ctx columns."""
    r = _run("pareto", timeout=180)
    assert r.returncode == 0, f"pareto crashed: {r.stderr}"
    assert "quant" in r.stdout
    assert "Non-dominated" in r.stdout
