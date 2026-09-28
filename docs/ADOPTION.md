# ADOPTION — how a real team adopts fitsproof on a Tuesday

Pass 3 (real-world applicability), cycle 1. Every command below was run on
2026-09-27 on the actual target machine (ThinkStation P500, Quadro M2000,
31 GB RAM, Python 3.11.15). Raw output only.

**The named ecosystem tool is ollama 0.20.3** with a locally pulled
`gemma3:4b` (4.3 GB, Q4_K_M). It was chosen because it is what the team
already runs: this box serves models through ollama today, and ollama is
the most widely deployed local-model runtime. fitsproof does not replace
it. fitsproof gates it.

---

## 0. The team and the moment

A two-person team shares this workstation. They run ollama for coding
assistants and small eval jobs. Twice in the last quarter a model load
pushed the box into swap and took an unrelated job down. Nobody logs an
OOM as a bug — it just "reboots itself". The Tuesday adoption is: put a
pre-flight gate in front of `ollama run` so a load that will not fit is
refused *before* it is attempted, with the binding constraint named.

What they get: one script, ~1.7 s per check, offline, no daemon, no
config file. What they give up: nothing at runtime — the gate runs
before ollama and exits.

---

## 1. Install (5 minutes, offline after the clone)

```
$ git clone <repo> && cd fitsproof
$ uv venv && uv pip install -e '.[dev]'
$ .venv/bin/fitsproof probe
Probing machine...
  bandwidth:  3.94 GB/s
  gemm:       302.41 GFLOPS
  RAM:        33.55 GB
  VRAM:       0.00 GB
```

`probe` is the machine characterisation: measured DRAM bandwidth, not a
spec-sheet number. It takes **0.64 s**:

```
$ time .venv/bin/fitsproof probe
...
real	0m0.636s
```

Note `VRAM: 0.00 GB`: the v0.1 probe does not enumerate GPUs. The
contract on this host is a **RAM** contract (see failure mode F-5).

---

## 2. The integration recipe: gate ollama behind fitsproof

The gate is `scripts/ollama_gate.py`. It reads the model's real
architecture from the running ollama daemon (`POST /api/show`), parses
the vocabulary size out of the GGUF blob itself, maps the quantisation
class, and runs `plan → admit` against your declared budget. If it cannot
read the metadata it **refuses** (fail-closed — it never guesses a model
shape).

Chained before an ollama invocation:

```bash
./scripts/ollama_gate.py gemma3:4b --budget-gb 12 --context 4096 \
    && ollama run gemma3:4b
```

### 2a. The admit path (the one you want)

```
$ .venv/bin/python scripts/ollama_gate.py gemma3:4b --budget-gb 12 --context 4096
model:   gemma3:4b (4.3B, Q4_K_M -> int4_sym)
shape:   34L x 2560h, heads 8/4, vocab 262145, ctx 4096
budget:  12 GB
ADMITTED: 7.219 GB predicted peak <= 12.000 GB budget (margin: 4781.1 MB)
EXIT:0
```

Cost of one gate run, measured:

```
$ /usr/bin/time -v .venv/bin/python scripts/ollama_gate.py gemma3:4b --budget-gb 12 --context 4096
ADMITTED: 7.219 GB predicted peak <= 12.000 GB budget (margin: 4781.1 MB)
	Elapsed (wall clock) time (h:mm:ss or m:ss): 0:01.68
	Maximum resident set size (kbytes): 303604
```

### 2b. The refusal path (the point of the product)

```
$ .venv/bin/python scripts/ollama_gate.py gemma3:4b --budget-gb 3 --context 4096
model:   gemma3:4b (4.3B, Q4_K_M -> int4_sym)
shape:   34L x 2560h, heads 8/4, vocab 262145, ctx 4096
budget:  3 GB
REFUSED: needs 7.219 GB, budget 3.000 GB; no listed option fits — nearest is "Offload ~50% of layers to system RAM (CPU fallback for those layers)" at 3.699 GB (0.699 GB above budget)
EXIT:2
```

Exit 2 stops the `&&` chain — ollama never loads. And this refusal is
*correct*: ollama actually holds 4.4 GB for this model (see 2d), which
is above the 3 GB budget.

### 2c. The degradation path — strict by default (a finding, see F-2)

```
$ .venv/bin/python scripts/ollama_gate.py gemma3:4b --budget-gb 6 --context 4096
model:   gemma3:4b (4.3B, Q4_K_M -> int4_sym)
shape:   34L x 2560h, heads 8/4, vocab 262145, ctx 4096
budget:  6 GB
DEGRADED: base config needs 7.219 GB > budget 6.000 GB. Applying: Offload ~50% of layers to system RAM (CPU fallback for those layers). New predicted peak: 3.699 GB.
DEGRADATION RECORD: Offload ~50% of layers to system RAM (CPU fallback for those layers)
GATE: config only fits after a declared degradation, and this gate cannot apply it to an external engine. Not chaining. Re-run with --allow-degrade only if you will wire the degradation yourself.
EXIT:2
```

A fitsproof degradation is a promise about fitsproof's own engine. The
gate cannot apply "offload 50% of layers" to ollama, so a DEGRADED
verdict must **not** green-light the chained command — that would be
exactly the silent contract violation this repo exists to prevent.
`--allow-degrade` opts in explicitly:

```
$ .venv/bin/python scripts/ollama_gate.py gemma3:4b --budget-gb 6 --context 4096 --allow-degrade
...
DEGRADED: ... New predicted peak: 3.699 GB.
EXIT:0
```

### 2d. Post-run verification (closing the loop)

After the run, compare prediction against reality:

```
$ ollama ps
NAME         ID              SIZE      PROCESSOR          CONTEXT    UNTIL
gemma3:4b    a2af6cc3eb7f    4.4 GB    44%/56% CPU/GPU    4096       2 minutes from now
```

- predicted peak: **7.219 GB** — observed resident footprint: **4.4 GB**.
  The prediction is conservative by +64% on this model (why, in F-1).
- For a refusal gate, error in the safe direction: an ADMITTED verdict
  (predicted ≤ budget) implies the real load fits with margin.

Measured decode on this box, for the cost record:

```
$ curl -s http://localhost:11434/api/generate -d '{"model":"gemma3:4b","prompt":"The capital of France is","stream":false,"options":{"num_ctx":4096,"temperature":0}}'
{'eval_count': 23, 'eval_duration': 1857110766, 'prompt_eval_count': 14, ...}
decode tok/s: 12.38
```

12.38 tok/s measured, with ollama splitting the model 44% CPU / 56% GPU
— on a machine whose probe reports 0 GB VRAM. fitsproof's CPU roofline
would predict <1 tok/s for a 3.3 GB weight set on 4 GB/s DRAM. The
memory prediction is portable to ollama; **the throughput prediction is
not** (failure mode F-4).

---

## 3. Production failure modes (all observed on this machine)

### F-1 — Shape fidelity: +64% over-prediction on real GGUF models

The gate plans gemma3:4b at 7.219 GB; ollama holds 4.4 GB. Two causes,
both in the cost model's assumptions, not in the gate:

1. `ModelConfig.head_dim` is `hidden/num_heads` (= 320 for gemma3), but
   gemma3's real head_dim is 256 (from `gemma3.attention.key_length`).
   Every attention projection and the KV term are ~25% high.
2. `weight_bytes` stores the embedding and the output head in fp32
   unconditionally (2 × 262145 × 2560 × 4 = 5.37 GB of the 7.22 GB
   predicted). GGUF Q4_K_M ships those at reduced precision and gemma
   ties the output to the input embeddings — we effectively pay for the
   unembedding twice.

Direction: conservative. A budget of 12 GB is admitted honestly; a
budget of 3 GB is refused correctly. The cost is **false degradations**
for budgets between the true footprint (4.4 GB) and the predicted peak
(7.22 GB). Fix belongs to the cost model (implement/improve phase), with
the KAT being exactly this observed 4.4 GB.

### F-2 — Refusal wording names a config that does not fit (**FIXED**, improve pass c1-p09-improve-2)

Before (this pass's original transcript, §2b above as first run):

```
REFUSED: needs 7.22 GB, budget 3.00 GB; nearest fitting config is Offload ~50% of layers ...
```

At 3 GB no degradation fits (the offload option predicts 3.699 GB), yet the message
offered it as the "nearest *fitting* config" (`plan.py` appended `degradations[-1]`
unconditionally on the DOES_NOT_FIT path) — and it was not even the nearest option.
The decision was right; the message misled.

Fixed in improve pass 2: `plan.no_fit_reason()` (src/fitsproof/contract/plan.py) is
now used on the DOES_NOT_FIT path. It names the option with the smallest predicted
peak and states the gap above budget, so the message can never contradict the
`[does not fit]` tags the CLI prints underneath:

```
$ .venv/bin/fitsproof admit --budget-gb 0.001
REFUSED: needs 0.042 GB, budget 0.001 GB; no listed option fits — nearest is "Use int4_sym quantisation instead of none" at 0.006 GB (0.005 GB above budget)
```

Pinned by `tests/contract/test_plan_admit_verify.py::
test_refusal_never_names_a_non_fitting_config`, which fails on all three old faults
(claiming a fitting config; naming the last option instead of the nearest; naming an
option above budget without saying so).

### F-3 — Calibration drifts with machine state

Probe bandwidth readings for the same box:

```
2026-09-26 (EVIDENCE.md §6):   7.18 GB/s
2026-09-27 (this pass):        3.94, 3.96, 4.22 GB/s (three runs)
```

Run-to-run spread today is ~7% (fine); day-to-day it is ~1.8× (not
fine). A profile saved on a loaded box is wrong when the box goes idle,
and vice versa. Operational rule: re-run `fitsproof probe` when the
sustained load profile changes, and store the profile with the job that
consumes it — probe records what was measured, when, and where.

### F-4 — CPU roofline vs hybrid CPU/GPU engines

The throughput model is calibrated from DRAM bandwidth and GEMM
throughput on the CPU. ollama offloads 56% of gemma3:4b to the GPU;
measured 12.38 tok/s vs a sub-1 tok/s CPU prediction. The predicted
`tok/s` must never be published for a GPU-backed engine from a CPU
profile. The memory prediction (F-1 aside) is architecture arithmetic
and does port.

### F-5 — GPU memory is unmeasured on v0.1

`probe` reports `VRAM: 0.00 GB` on a box with a Quadro M2000. The
contract covers host RAM only until the probe enumerates GPUs. On this
class of machine that is mostly honest (the model never fully fits the
4 GB device anyway — hence ollama's 44/56 split), but any claim of a
*VRAM* budget is out of scope for v0.1 and must not appear in adoption
material.

### F-6 — The gate fails closed when the daemon is down

If ollama is unreachable the gate exits 1 with
`GATE ERROR: ollama daemon unreachable` and nothing is admitted. In CI
this means the job must start the daemon or skip the gate explicitly —
a skipped gate is visible in the log, a swallowed one would not be.

---

## 4. Operational cost

| Item | Measured cost |
|---|---|
| `fitsproof probe` | 0.64 s wall, offline |
| Full gate run (probe + /api/show + GGUF parse + plan) | 1.68 s wall, 297 MB RSS |
| Runtime overhead of the gated model | **zero** — the gate exits before ollama runs |
| Services to run | none (no daemon, no config, no DB) |
| Network | none after install (daemon is localhost) |
| CI | one script, CPU-only, seconds; no GPU |
| Maintenance | re-probe when host load profile changes (F-3) |

The recurring cost is one line in front of every `ollama run` and the
discipline of re-probing. Everything else is one-time.

---

## 5. The single most likely reason someone would NOT adopt it

**The first time it disagrees with reality, it is wrong in the
annoying direction — and the operator can prove it.**

A team that gates ollama at a 5 GB budget gets `DEGRADED` (predicted
7.22 GB) for a model that ollama then loads at 4.4 GB — the gate was
wrong by +64%, and to make it green you must either accept a
degradation record you cannot apply, or keep the gate from talking at
all. Operators who catch a tool being confidently wrong once do the
rational thing: they bypass it. Every extra `--no-gate` flag erodes the
only thing fitsproof sells — trust in the refusal.

This is a prediction-accuracy problem, not a positioning problem
(MARKET-VERDICTS.md's gap still holds — see RESEARCH.md Pass 3 for the
aura finding and the narrowed claim). It is fixable with exactly what
the spec already demands: per-machine calibration with published held-out
MAPE (currently 46.1% fresh / 50.3% on 2026-09-27 morning, n_held_out=1), embedding and
head-dimension terms corrected against GGUF reality, and the observed
4.4 GB footprint used as a known-answer test. Until that lands, the
honest adoption advice is: **run the gate on ADMITTED-only paths
(budgets ≥ predicted peak), treat DEGRADED as refuse (the gate's
default), and read the refusal's "nearest" option as the closest
*listed* option — since the F-2 fix it is never claimed to fit.**

---

## 6. What adoption looks like at each maturity level

| Level | What the team does | Status in this repo |
|---|---|---|
| L0 — try it | clone, `probe`, `plan` against a budget | works today (this document) |
| L1 — gate the box | chain `ollama_gate.py && ollama run` | works today, strict mode |
| L2 — CI admission | gate in CI before any model pull | works today (CPU-only, seconds) |
| L3 — in-process guard | `@guard(budget=...)` in Python services | works today (`fitsproof.client.guard`, tested) |
| L4 — agent-facing | MCP server so an agent asks before loading | works today (`fitsproof mcp`, tested) |
| L5 — drop-in binary | single executable, SHA256 release | v0.2 MANDATE M1 (not built yet) |

L0–L4 are adopted from source on a Tuesday (L3/L4 landed in the v0.2 mandate pass and
are exercised by the test suite). L5 is the remaining mandate item.

---

## 7. v0.2 surface reality check — cycle 2

*2026-09-27T22:34Z. All four v0.2 plugin surfaces are now implemented and
tested. This section documents them against raw output from this machine.*

*Updated 2026-09-28 (c2-p09-improve-2): `fitsproof plan` and `fitsproof admit` are now
distinct commands. `plan` shows the prediction (peak, CI, tok/s, verdict) and exits 0
even on `does_not_fit`; `admit` enforces and exits 2 on refusal. The README CLI section
reflects this with a worked example.*

### The four surfaces, one at a time

**L2 — CLI gate (plan to inspect, admit to enforce)**

```
$ .venv/bin/fitsproof plan --budget-gb 4
predicted peak:  0.042 GB  (95% CI: [0.033, 0.050] GB)
predicted tok/s: 98.6  (95% CI: [69.0, 128.2])
budget:          4.000 GB
verdict:         fits

$ .venv/bin/fitsproof admit --budget-gb 4
ADMITTED: 0.042 GB predicted peak <= 4.000 GB budget (margin: 3958.3 MB)
$ echo $?
0

$ .venv/bin/fitsproof plan --budget-gb 0.001
predicted peak:  0.042 GB  (95% CI: [0.033, 0.050] GB)
predicted tok/s: 103.8  (95% CI: [72.6, 134.9])
budget:          0.001 GB
verdict:         does_not_fit
degradation options:
  [does not fit] Use int4_sym quantisation instead of none -> 0.006 GB  (724.2 tok/s)
  ...
$ echo $?
0   # plan exits 0 — it describes; it does not enforce

$ .venv/bin/fitsproof admit --budget-gb 0.001
REFUSED: needs 0.042 GB, budget 0.001 GB; no listed option fits — nearest is "Use int4_sym quantisation instead of none" at 0.006 GB (0.005 GB above budget)
$ echo $?
2   # admit exits 2 on refusal
```

**L3 — in-process Python guard**

```python
from fitsproof.client import DoesNotFit, FitsproofClient, guard

# Direct client usage
client = FitsproofClient()
plan = client.plan(context_len=512, budget_bytes="4GiB")
record = client.admit(plan)
print(record.message)
# → ADMITTED: 0.042 GB predicted peak <= 4.295 GB budget (margin: 4253.3 MB)

print(client.metrics()["ram_gb"])
# → 33.548316672

# Guard decorator — raises before the caller allocates
loaded = []

@guard(budget="1MiB")   # reference model needs ~40 MB; this will refuse
def load_model():
    loaded.append("allocated")

try:
    load_model()
except DoesNotFit as e:
    print("refused:", e)

assert loaded == []     # the callable was provably never invoked
```

The `@guard` surface is the right tool for a Python inference service that
wants to enforce a budget before calling any allocation. The check happens
in-process, before the caller reaches the model loading code.

**L4 — MCP server (for agents)**

```python
import json, subprocess, sys

messages = [
    {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
    {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    {
        "jsonrpc": "2.0", "id": 3, "method": "tools/call",
        "params": {"name": "admit", "arguments": {"budget": "4GiB", "context_len": 512}},
    },
]
proc = subprocess.run(
    [sys.executable, "-m", "fitsproof.cli", "mcp"],
    input="\n".join(json.dumps(m) for m in messages) + "\n",
    capture_output=True, text=True, timeout=60,
)
replies = [json.loads(l) for l in proc.stdout.splitlines() if l.strip()]
# replies[0]["result"]["serverInfo"]["name"] == "fitsproof-mcp"
# sorted(t["name"] for t in replies[1]["result"]["tools"]) == ["admit","plan","probe"]
# replies[2]["result"]["isError"] == False
```

Register with any MCP host as `fitsproof mcp` (stdio transport). An agent that
calls the `admit` tool and gets `isError: false` can proceed; `isError: true`
means the contract refused and the agent should not attempt the load.

**L5 — Binary release (v0.2 MANDATE M1, not yet built)**

```bash
# When M1 lands, the adoption story becomes zero-install:
curl -L https://github.com/AnnasMazhar/fitsproof/releases/download/v0.2.0/fitsproof-linux-x86_64 \
  -o fitsproof && chmod +x fitsproof
sha256sum fitsproof   # verify against the release page value
./fitsproof probe
./fitsproof admit --budget-gb 4
./fitsproof mcp &     # start MCP server
```

L0–L4 work today from source. L5 is the remaining M1 deliverable. Until
the binary release exists, the install requires Python (§1 recipe above).

### What changed from cycle 1 to cycle 2

| Item | Cycle 1 (pass 3) | Cycle 2 (pass 3) |
|---|---|---|
| L3 Python client | Not yet implemented | Works; `FitsproofClient`, `@guard`, `DoesNotFit` |
| L4 MCP server | Not yet implemented | Works; `tools=['admit','plan','probe']`, refused config → `isError: true` |
| Refusal wording | Named a non-fitting config as "nearest fitting" (F-2) | Fixed (c1-p09-improve-2); names smallest-predicted-peak option + gap above budget |
| MAPE reported | 46.1% (first run) | 50.3–60.1% across multiple sessions; run-to-run variation confirmed |
| Test count | 88 | 155 |
| Stress harness | 25 configs, 0 violations | 25 configs, 0 violations (confirmed fresh this pass) |

### Adoption maturity table — updated

| Level | What the team does | Status |
|---|---|---|
| L0 — try it | clone, `probe`, `plan` against a budget | works today |
| L1 — gate the box | `ollama_gate.py && ollama run` | works today (§2) |
| L2 — CLI gate | `fitsproof admit` in shell scripts / CI | works today |
| L3 — in-process guard | `@guard(budget=...)` in Python services | **works today** (new this cycle) |
| L4 — agent-facing | `fitsproof mcp`, agent calls `admit` before loading | **works today** (new this cycle) |
| L5 — drop-in binary | single executable, SHA256 release | v0.2 M1 pending |

---

## 8. Cycle 3 — Pass 3 — State as of 2026-09-28T06:00Z

*All commands run fresh on the ThinkStation P500, Python 3.11.15, branch feat/v0.1.*

### 8.1 What changed since cycle 2

| Item | Cycle 2 (c2-p3, 2026-09-27T22:34Z) | Cycle 3 (c3-p3, 2026-09-28T06:00Z) |
|---|---|---|
| Test count | 155 passed | **169 passed** (+14) |
| MAPE (held-out) | 50.3–60.1% across sessions | **49.1%** this session (n_held_out=1) |
| Stress harness | 25 configs, 0 violations | 25 configs, 0 violations (confirmed) |
| Stress margin | min=3698.0 MB | min=3909.2 MB (run-to-run variation in bandwidth measurement) |
| CLI plan/admit | Distinct commands (c2-p09) | Unchanged |
| Python client (L3) | Working | Confirmed working |
| MCP server (L4) | Working | Confirmed working: admit 4GiB → `isError:false`, admit 1MiB → `isError:true` |
| Binary release (L5) | Not built | **Still not built** — v0.2 MANDATE M1 pending |

The MAPE variation (46.1% → 60.1% → 49.1% across three cycles' sessions) is within
the expected run-to-run spread documented as F-3 (bandwidth drift with machine state).
The underlying instrument has not changed; the number fluctuates because the bandwidth
probe is sensitive to load at measurement time.

### 8.2 Raw output — 2026-09-28T06:00Z

```
$ source .venv/bin/activate && python scripts/calibration_demo.py
=== Calibration demo ===
bandwidth: 6.62 GB/s
gemm:      89.64 GFLOPS
RAM:       33.5 GB
bandwidth_utilisation: 0.0209
MAPE (held-out):       49.1%
CI (95%):              [49.1%, 49.1%]
n_train=2, n_held_out=1
```

```
$ fitsproof probe
Probing machine...
  bandwidth:  6.57 GB/s
  gemm:       44.96 GFLOPS
  RAM:        33.55 GB
  VRAM:       0.00 GB
```

```
$ fitsproof stress
ADMITTED: 0.039 GB predicted peak <= 4.000 GB budget (margin: 3961.0 MB)
Stress harness: 25 configs, 0 violations, 0 silent mode changes. Margin: min=3909.2 MB, median=3909.4 MB, max=3912.8 MB.
```

```
$ fitsproof admit --budget-gb 4
ADMITTED: 0.042 GB predicted peak <= 4.000 GB budget (margin: 3958.3 MB)

$ fitsproof admit --budget-gb 0.001
REFUSED: needs 0.042 GB, budget 0.001 GB; no listed option fits — nearest is "Use int4_sym quantisation instead of none" at 0.006 GB (0.005 GB above budget)
Degradation options:
  [does not fit] Use int8_sym quantisation instead of none -> 0.011 GB
  [does not fit] Use int4_sym quantisation instead of none -> 0.006 GB
  [does not fit] Reduce context to 256 tokens (1/2 of 512) -> 0.040 GB
  [does not fit] Reduce context to 128 tokens (1/4 of 512) -> 0.039 GB
  [does not fit] Reduce context to 64 tokens (1/8 of 512) -> 0.039 GB
  [does not fit] Offload ~50% of layers to system RAM (CPU fallback for those layers) -> 0.022 GB
(exit: 2)
```

```
$ fitsproof plan --budget-gb 4
predicted peak:  0.042 GB  (95% CI: [0.033, 0.050] GB)
predicted tok/s: 107.5  (95% CI: [75.3, 139.8])
budget:          4.000 GB
verdict:         fits
```

Python client and guard:

```
$ python -c "
from fitsproof.client import DoesNotFit, FitsproofClient, guard
client = FitsproofClient()
record = client.admit(client.plan(context_len=512, budget_bytes='4GiB'))
print(record.message)
print('ram_gb:', client.metrics()['ram_gb'])
loaded = []
@guard(budget='1MiB')
def load_model():
    loaded.append('allocated')
try:
    load_model()
except DoesNotFit as e:
    print('refused:', str(e)[:80])
print('loaded ==', loaded)
"
ADMITTED: 0.042 GB predicted peak <= 4.295 GB budget (margin: 4253.3 MB)
ram_gb: 33.548316672
refused: REFUSED: needs 0.042 GB, budget 0.001 GB; no listed option fits — nearest is "Us
loaded == []
```

MCP server (admitted and refused):

```
$ python -c "
import json, subprocess, sys
messages = [
    {'jsonrpc':'2.0','id':1,'method':'initialize','params':{}},
    {'jsonrpc':'2.0','id':2,'method':'tools/list','params':{}},
    {'jsonrpc':'2.0','id':3,'method':'tools/call',
     'params':{'name':'admit','arguments':{'budget':'4GiB','context_len':512}}},
    {'jsonrpc':'2.0','id':4,'method':'tools/call',
     'params':{'name':'admit','arguments':{'budget':'1MiB','context_len':512}}},
]
proc = subprocess.run([sys.executable,'-m','fitsproof.cli','mcp'],
    input='\n'.join(json.dumps(m) for m in messages)+'\n',
    capture_output=True, text=True, timeout=60)
replies = [json.loads(l) for l in proc.stdout.splitlines() if l.strip()]
print('server:', replies[0]['result']['serverInfo']['name'])
print('tools:', sorted(t['name'] for t in replies[1]['result']['tools']))
print('admit 4GiB isError:', replies[2]['result']['isError'])
print('admit 1MiB isError:', replies[3]['result']['isError'])
"
server: fitsproof-mcp
tools: ['admit', 'plan', 'probe']
admit 4GiB isError: False
admit 1MiB isError: True
```

Test suite:

```
$ python -m pytest tests/ -q --tb=no
169 passed in 107.65s (0:01:47)
```

### 8.3 Failure modes — cycle 3 status

No new failure modes were observed in cycle 3. The five documented failure modes from
ADOPTION §3 (F-1 through F-6) remain open or closed per the prior cycle's state:

| Finding | Status | C3 observation |
|---|---|---|
| F-1: +64% over-prediction on real GGUF models | **Open** (cost model fix in implement pass) | Not re-tested against ollama this cycle; MAPE at 49.1% on fixture (within prior range) |
| F-2: Refusal names a non-fitting config | **Fixed** (c1-p09-improve-2) | Confirmed fixed: "no listed option fits — nearest is..." correctly named in CLI output above |
| F-3: Bandwidth drift with machine state | **Open / documented** | Today's probe: 6.57 GB/s vs 3.94–7.18 GB/s across sessions. Operational rule (re-probe on load change) documented in §3 |
| F-4: CPU roofline tok/s invalid for GPU-backed engines | **Open / documented** | Not re-tested; not changed since cycle 2 |
| F-5: VRAM unmeasured (probe shows 0.00 GB) | **Open / documented** | Confirmed: `VRAM: 0.00 GB` in probe output above |
| F-6: Gate fails closed when ollama daemon is down | **Open / documented** | Not re-tested; gate code unchanged |

The most adoption-critical finding remains **F-1**: the +64% over-prediction on real
GGUF models produces false DEGRADED verdicts in the 4.4–7.2 GB budget range. This is
the fix target for the cycle 3 implement pass. Until the cost model is corrected, the
operational guidance from §5 applies: treat DEGRADED as refuse-by-default, and confirm
any ADMITTED verdict is at a budget above the predicted peak, not merely above the true
footprint.

### 8.4 Adoption maturity table — final state for cycle 3

| Level | What the team does | Status |
|---|---|---|
| L0 — try it | clone, `probe`, `plan` against a budget | Works today |
| L1 — gate the box | `ollama_gate.py && ollama run` | Works today (strict mode) |
| L2 — CLI gate | `fitsproof admit` in shell scripts / CI | Works today; exit 0 / exit 2 |
| L3 — in-process guard | `@guard(budget=...)` in Python services | Works today; DoesNotFit raised before callable invoked |
| L4 — agent-facing | `fitsproof mcp`, agent calls `admit` before loading | Works today; `isError:false` / `isError:true` |
| L5 — drop-in binary | single executable, SHA256 release | v0.2 MANDATE M1 — not built yet |

### 8.5 The single most likely reason someone would NOT adopt it — cycle 3 update

No change from §5. The prediction accuracy (MAPE 46–62% across sessions, +64% on the
one real-model test) remains the adoption blocker. The contract surfaces (L2–L4) work
correctly; the number the contract enforces is the weak link.

The path to closing this is narrow and clear: correct the cost model's head_dim default
(use the model config's `attention.key_length`, not `hidden_size / num_heads`) and the
embedding/unembedding fp32 double-count, then re-measure against gemma3:4b. If the
resulting MAPE drops below 20% on the one real-model calibration point, the F-1 finding
is resolved and the adoption-blocker analysis changes.

### 8.6 Changes in c3-p09-improve-2 (2026-09-28T09:00Z)

| Item | Before | After |
|---|---|---|
| README Limitations MAPE range | "~46–50%" | "~46–62%" (matches observed range across sessions including today's 61.9% run) |
| README calibration section | No note on why example shows 46.1% | Explicit note: low bandwidth (1.92 GB/s) = loaded box; MAPE varies 46–62% across sessions |
| README stress output | Slightly stale median (3909.5 MB) | Updated to match fresh run (3909.7 MB) |
| `test_calibration_demo_runs` | Missing | Added to `tests/test_packaging.py`; asserts the README-referenced script exits 0 and emits all required output fields |
| Test count | 177 | **178** (+1) |
| `ruff check .` | clean | clean |
