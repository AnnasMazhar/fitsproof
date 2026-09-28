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
REFUSED: needs 7.22 GB, budget 3.00 GB; nearest fitting config is Offload ~50% of layers to system RAM (CPU fallback for those layers)
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

### F-2 — Refusal wording names a config that does not fit (defect, open)

```
REFUSED: needs 7.22 GB, budget 3.00 GB; nearest fitting config is Offload ~50% of layers ...
```

At 3 GB no degradation fits (the offload option predicts 3.699 GB), yet
the message still offers it as the "nearest *fitting* config"
(`plan.py` appends `degradations[-1].description` unconditionally on the
DOES_NOT_FIT path). The decision is right; the message misleads. Left
unfixed in this research pass — recorded as finding F-2 for the improve
pass, with a test that pins the wording only when the named config
actually fits.

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
MAPE (currently 50.3% at reference scale, n_held_out=1), embedding and
head-dimension terms corrected against GGUF reality, and the observed
4.4 GB footprint used as a known-answer test. Until that lands, the
honest adoption advice is: **run the gate on ADMITTED-only paths
(budgets ≥ predicted peak), treat DEGRADED as refuse (the gate's
default), and treat the refusal message's "nearest fitting config" as
advisory (F-2).**

---

## 6. What adoption looks like at each maturity level

| Level | What the team does | Status in this repo |
|---|---|---|
| L0 — try it | clone, `probe`, `plan` against a budget | works today (this document) |
| L1 — gate the box | chain `ollama_gate.py && ollama run` | works today, strict mode |
| L2 — CI admission | gate in CI before any model pull | works today (CPU-only, seconds) |
| L3 — in-process guard | `@guard(budget=...)` in Python services | implemented (v0.1) |
| L4 — agent-facing | MCP server so an agent asks before loading | implemented (v0.1) |
| L5 — drop-in binary | single executable, SHA256 release | implemented (v0.1) |

L0–L2 are adopted from source on a Tuesday. L3–L5 are implemented in v0.1;
all levels are available today.
