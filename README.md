# fitsproof

Your engine tells you it fits. This one proves it — and refuses, loudly, when it doesn't.

`fitsproof` is a local LLM inference engine whose product is a **guarantee, not a benchmark**.
It does three things no single existing tool does together:

1. **Predicts** peak memory and decode throughput for your machine from a model calibrated by measuring it.
2. **Enforces** a declared memory budget: it admits, refuses, or degrades — explicitly and loudly. No silent OOM. No silent CPU fallback.
3. **Proves** it: a stress harness asserts measured peak memory never exceeded the declared budget.

---

## What this is not

fitsproof is not a new CUDA kernel. It does not claim speed superiority over
llama.cpp, vLLM, or KTransformers. Those engines are faster, more mature, cover
more hardware, and support more models. `COMPARISONS.md` names each one and states
where it beats us.

The engine exists to make the contract real and testable: a NumPy-only
correctness-first runtime that runs offline with no GPU and no CUDA toolkit.

---

## Install and run

```bash
uv venv && uv pip install -e '.[dev]'
pytest -q
```

Runs offline, no GPU, no network. Python 3.11+.

---

## Usage

```python
from fitsproof.engine.model import get_reference_bundle
from fitsproof.engine.transformer import Transformer
from fitsproof.contract.probe import probe
from fitsproof.contract.plan import plan
from fitsproof.contract.admit import admit

cfg, weights = get_reference_bundle()
transformer = Transformer(cfg, weights)
machine = probe()

# Plan and enforce a 4 GB budget
p = plan(cfg, machine, context_len=512, budget_bytes=4 * 1024**3)
record = admit(p)
print(record.message)
# ADMITTED: 0.039 GB predicted peak <= 4.000 GB budget (margin: 3961.1 MB)
```

---

## Architecture

```
src/fitsproof/
  engine/          NumPy-only transformer runtime
    model.py       Bundle format + in-repo reference model (6L/384H/GQA, ~38MB fp32)
    attention.py   MHA + GQA + RoPE + KV cache
    quant.py       int8_sym, int8_asym, int4_sym weight quantisation
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
```

---

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

- **Speculative decoding equality only holds at temperature=0.** Probabilistic
  acceptance (temperature > 0) requires rejection sampling (Algorithm 1 of
  Leviathan et al. 2023), not implemented.

- **No NTK-aware RoPE scaling.** Context lengths beyond `max_seq_len` will degrade.

- **In-repo reference model is randomly initialised.** It generates valid token
  sequences but not coherent text. It exists to run all contract tests offline.

---

## Comparisons

See `COMPARISONS.md` for a full table. The short version:

| Tool | What it does better than fitsproof |
|---|---|
| llama.cpp | Mature, broad model support, fast CPU kernels, broad quant support |
| vLLM | GPU serving, high throughput, PagedAttention, broad model support |
| KTransformers | CPU/GPU hybrid MoE, AMX kernels, runs 671B on ~14 GB VRAM |
| ridgepoint | Calibrated VRAM/roofline prediction for GPU (A100/H100) |
| Strata | Consumer packaging, one-click install |

fitsproof's position: none of the above enforces a *resource contract* with a measured
proof of compliance and explicit degradation on any configuration. That is the claim.
