# fitsproof

[![CI](https://github.com/AnnasMazhar/fitsproof/actions/workflows/ci.yml/badge.svg)](https://github.com/AnnasMazhar/fitsproof/actions/workflows/ci.yml)

**Prove your local LLM fits in memory — or get a loud refusal instead of a silent OOM.**

fitsproof is for people running LLMs on consumer hardware (4–8 GB VRAM / 16–32 GB RAM): a class
of machine every mainstream engine either ignores or silently falls back from. It **predicts**
peak memory for your machine from an on-device calibration, **enforces** a declared budget
(admit / degrade loudly / refuse), and **proves** it with a measured stress harness.

## Quickstart (clean machine, no GPU, no CUDA toolkit, no model download)

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install "git+https://github.com/AnnasMazhar/fitsproof.git"
fitsproof admit --budget-gb 4
fitsproof admit --budget-gb 0.001   # REFUSED — names the binding constraint, exit code 2
```

Python 3.11+. Everything runs offline after install. Once the PyPI release is published
(pending — `pip install fitsproof` is not yet available), use the Git-URL form above.

If this is useful, star the repo.

## Headline evidence

`fitsproof stress` runs 25 configurations against a declared budget and fails the build on any
violation or undocumented mode change. Real output (budget: 4 GB, reference model ~70 MB):

```
$ fitsproof stress
ADMITTED: 0.070 GB predicted peak <= 4.000 GB budget (margin: 3930.5 MB)
Stress harness: 25 configs, 0 violations, 0 silent mode changes. Margin: min=3912.7 MB, median=3912.9 MB, max=3916.2 MB.
```

## Plug it in — four surfaces, every snippet below is executed by the test suite

### 1. Python client (`fitsproof.client`)

```python
# SURFACE: client
from fitsproof.client import DoesNotFit, FitsproofClient

client = FitsproofClient()
plan = client.plan(context_len=512, budget_bytes="4GiB")
record = client.admit(plan)
print(record.message)

try:
    client.admit(client.plan(context_len=512, budget_bytes="1MiB"))
except DoesNotFit as e:
    print("refused:", e)

assert client.metrics()["ram_gb"] > 0
```

### 2. Guard decorator — raises **before** the caller allocates

```python
# SURFACE: guard
from fitsproof.client import DoesNotFit, guard

loaded = []

@guard(budget="1MiB")  # the reference model needs ~40 MB: this config cannot fit
def load_model():
    loaded.append("allocated")

try:
    load_model()
except DoesNotFit as e:
    print(e)

assert loaded == []  # the wrapped callable was provably never invoked
```

### 3. OpenAI-compatible server — existing SDK code works by changing `base_url` only

```python
# SURFACE: server
import json
import socket
import time
import urllib.request

from fitsproof.engine.model import get_reference_bundle
from fitsproof.engine.server import start_server
from fitsproof.engine.transformer import Transformer

cfg, weights = get_reference_bundle()
with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
srv = start_server(Transformer(cfg, weights), cfg, host="127.0.0.1", port=port, block=False)

# Your existing client, pointed at fitsproof:
#   from openai import OpenAI
#   client = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="unused")
payload = {
    "messages": [{"role": "user", "content": "hi"}],
    "max_tokens": 5,
    "temperature": 0.0,
    "stream": False,
}
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
for _ in range(30):
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read())
        break
    except Exception:
        time.sleep(0.2)
else:
    raise RuntimeError("server did not start")

assert body["choices"][0]["message"]["content"]
assert body["fitsproof"]["admission"] in ("admitted", "degraded")
srv.shutdown()
```

Every response carries a `fitsproof` field with the admission record. A request whose predicted
peak exceeds the server's budget is refused with HTTP 503 and the binding constraint named —
never a silent OOM.

### 4. MCP server — an *agent* consults the contract before it loads a model

```python
# SURFACE: mcp
import json
import subprocess
import sys

messages = [
    {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
    {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {"name": "admit", "arguments": {"budget": "4GiB", "context_len": 512}},
    },
]
proc = subprocess.run(
    [sys.executable, "-m", "fitsproof.cli", "mcp"],
    input="\n".join(json.dumps(m) for m in messages) + "\n",
    capture_output=True,
    text=True,
    timeout=120,
)
replies = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
assert replies[0]["result"]["serverInfo"]["name"] == "fitsproof-mcp"
tool_names = {t["name"] for t in replies[1]["result"]["tools"]}
assert {"probe", "plan", "admit"} <= tool_names
assert replies[2]["result"]["isError"] is False
```

Register it with any MCP host as `fitsproof mcp` (stdio).

## What this is not

fitsproof is not a new CUDA kernel. It does not claim speed superiority over
llama.cpp, vLLM, or KTransformers. Those engines are faster, more mature, cover
more hardware, and support more models. `COMPARISONS.md` names each one and states
where it beats us.

The engine exists to make the contract real and testable: a NumPy-only
correctness-first runtime that runs offline with no GPU and no CUDA toolkit.

## CLI

```
fitsproof probe    # measure this machine (bandwidth, GEMM, RAM/VRAM)
fitsproof plan     # predict peak memory for a budget (--model <bundle> for a saved model)
fitsproof admit    # admit / degrade loudly / refuse (exit 2 on refusal)
fitsproof verify   # measure peak RSS during generation, assert <= budget
fitsproof stress   # >=20 configs: zero violations, zero silent mode changes
fitsproof serve    # OpenAI-compatible HTTP server (alias: fitsproof server)
fitsproof mcp      # MCP server (stdio): plan / admit / probe tools
fitsproof pareto   # measured Pareto frontier over (quant, context)
```

## Architecture

```
src/fitsproof/
  engine/          NumPy-only transformer runtime
    model.py       Bundle format + in-repo reference model (6L/384H/GQA, ~38MB fp32)
    attention.py   MHA + GQA + RoPE + KV cache
    quant.py       int8_sym, int8_asym, int4_sym, int4_asym weight quantisation
    sampling.py    greedy, temperature, top-k, top-p (seeded RNG)
    transformer.py RMSNorm + SwiGLU FFN + full forward pass
    speculative.py speculative decoding (Leviathan et al. 2023)
    server.py      OpenAI-compatible HTTP server (stdlib only)
  contract/        Resource contract enforcement
    probe.py       Machine characterisation from measurement (STREAM triad, GEMM)
    cost.py        Analytical roofline cost model
    calibrate.py   Fit model constants from on-device measurements; MAPE + CI
    plan.py        Compute verdict: fits / fits_with_degradation / does_not_fit
    admit.py       Enforcement point — emits a record on every mode change
    verify.py      Proof harness — measures peak RSS vs declared budget
    pareto.py      Pareto frontier sweep over (quant, context) configurations
  client.py        FitsproofClient, DoesNotFit, @guard decorator
  mcp.py           MCP stdio server (probe / plan / admit tools)
  cli.py           fitsproof command
```

## Limitations

These are honest. A repo with no stated limitations is not credible.

- **No CUDA kernels.** The engine is NumPy-only. It is correctness-first and slow
  vs llama.cpp or vLLM. That is intentional; the product is the contract, not speed.

- **Calibration is CPU-only.** The cost model is calibrated from DRAM bandwidth
  and NumPy GEMM measurements. It does not account for GPU memory hierarchy or
  compute rooflines.

- **Contract covers memory, not latency SLOs.** fitsproof enforces a peak RSS budget;
  it does not guarantee latency targets (tok/s predictions are estimates).

- **KV cache bandwidth not included in decode formula.** At long contexts, KV cache
  streaming adds to the bandwidth cost. The current formula is accurate for the
  reference model and short contexts; it underpredicts decode time at very long contexts.

- **RSS measurement is coarse.** Peak RSS on Linux is the high-water mark since
  process start. Allocations freed before the post-call sample may not be captured.

- **Boundary safety margin.** `admit` returns NEAR_BOUNDARY (exit 1) when the
  predicted peak is within 50 MiB (52 MB) of the declared budget. This covers a ~25%
  prediction overrun on a small model (~70 MB weights × 1.25 = ~17 MB overrun, well under
  the margin). Configurations flagged as NEAR_BOUNDARY should be validated with
  `fitsproof verify` before use. Admission at a tight budget does not guarantee the proof
  harness will pass.

- **Speculative decoding equality only holds at temperature=0.** Probabilistic
  acceptance (temperature > 0) requires rejection sampling (Algorithm 1 of
  Leviathan et al. 2023), not implemented.

- **No NTK-aware RoPE scaling.** Context lengths beyond `max_seq_len` will degrade.

- **In-repo reference model is randomly initialised.** It generates valid token
  sequences but not coherent text. It exists to run all contract tests offline.

## Comparisons

See `COMPARISONS.md` for the full table with star counts and release dates. Short version:

| Tool | What it does better than fitsproof |
|---|---|
| llama.cpp | Mature, broad model support, fast CPU kernels, broad quant support |
| vLLM | GPU serving, high throughput, PagedAttention, broad model support |
| KTransformers | CPU/GPU hybrid MoE, AMX kernels, runs 671B on ~14 GB VRAM |
| ridgepoint | Calibrated VRAM/roofline prediction for GPU (A100/H100) |
| Strata | Consumer packaging, one-click install |

fitsproof's position: none of the above enforces a *resource contract* with a measured
proof of compliance and explicit degradation on any configuration. That is the claim.

## Demo

`docs/demo.sh` is a 15-second end-to-end demo. Recording instructions are in its header.

## Contributing

See `CONTRIBUTING.md`. Every test file must name the fault it detects, and core tests
must cite their research source.

## License

MIT — see `LICENSE`.
