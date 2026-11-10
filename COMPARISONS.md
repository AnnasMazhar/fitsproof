# COMPARISONS

fitsproof is a resource contract engine. It does not compete on speed.
The tools below beat fitsproof on speed, model coverage, and hardware breadth.
We name them here so reviewers do not have to look them up.

Star counts and versions retrieved from the GitHub REST API and PyPI at
2026-10-27T20:08Z (cycle 8 pass 2 refresh). Confirmed against c8-p2 searches —
RAMLLM (Mayank3613/RAMLLM, v0.2.1) is a new entrant added this refresh; aura
has 0 commits since 2026-09-03; llama.cpp +2,683, vLLM +2,761, KTransformers +97
vs the c7-p2 snapshot.

## Inference engines

| Tool | Stars | Latest | What it does better than fitsproof | What fitsproof adds |
|---|---|---|---|---|
| **llama.cpp** (ggml-org/llama.cpp) | 132,551 | v0.7.0 (2026-10-15) | Mature (3+ years), broad model support (GGUF), fast CPU kernels (GGML), broad quant support, GPU offload | No enforced resource contract; silent OOM documented; no calibrated prediction interval; no `--budget` flag in v0.7.0 |
| **vLLM** (vllm-project/vllm) | 95,701 | v0.32.0 (2026-10-18) | GPU serving, high throughput, PagedAttention, continuous batching, 100+ model support | No enforced memory budget; targets A100/H100, not 4–8 GB VRAM class |
| **KTransformers** (kvcache-ai/ktransformers) | 19,641 | v0.7.2 (2026-10-15) | CPU/GPU hybrid MoE, Intel AMX kernels, runs DeepSeek-671B on ~14 GB VRAM. *Requires 128 GB RAM, AMX, CUDA/ROCm.* | Does not serve the 16–32 GB RAM class; no calibrated contract; no stress harness |
| **Strata** | N/A | N/A | Consumer packaging, one-click install for a single model on a consumer GPU. Designed for non-technical users. | Requires 12 GB+ VRAM and 64 GB RAM; no enforced memory contract; targets higher-end consumer hardware |

## Prediction, sizing, and planning tools

| Tool | Stars | Last push | What it does better than fitsproof | What fitsproof adds |
|---|---|---|---|---|
| **RAMLLM** (Mayank3613/RAMLLM, PyPI v0.2.1) | 0 | 2026-10-18 | Per-machine empirical calibration with ±2–4% tok/s accuracy on real hardware; live KV-cache quantisation governor; OpenAI-compatible server (`/v1/chat/completions` + `/ramllm/metrics`). Correctly targets consumer hardware (4–16 GB VRAM class). | No held-out train/test split with published MAPE; no stress harness asserting `measured ≤ budget` over ≥20 configs (the CI gate); no embeddable in-process `plan()`/`admit()` API; requires llama.cpp backend |
| **ridgepoint** (Isk4R1oT/ridgepoint, PyPI v0.1.2) | 2 | 2026-09-08 | Calibrated VRAM prediction for GPU (A100/H100), KV cache formula correct for GQA and MLA, ~1% accuracy vs real vLLM, per-field `calibrated` flag | GPU-only calibration; no enforcement gate; no RSS proof harness |
| **llm-inference-calculator** (pochenai) | 21 | 2026-10-06 | Two-phase roofline (prefill compute-bound TTFT + decode bandwidth-bound TPOT), MoE sparsity, spec-decoding modelling | No calibration; no enforcement; targets data-centre multi-GPU |
| **llm-roofline** (Pluenet-Killian) | 0 | 2026-06-20 | Decode throughput floor = bytes_per_token / bandwidth, per GPU | Throughput floor only; no memory prediction; no enforcement; inactive |
| **hardware-aware-llm-runtime** (JohnScheuer) | 0 | 2026-06-25 | Hardware-calibrated roofline with empirical fitting; predicts optimal batch within ~1 | Throughput focus; no enforcement; no stress harness; inactive |
| **llm-vram-calculator** (Shun-Calvin) | 3 | 2026-09-26 | Formula-based VRAM/TTFT/tok/s across 100+ models × 70+ GPUs | Formula-based, not calibrated; GPU-only; no enforcement |

## Enforcement / orchestration

| Tool | Stars | Last push | What it does better than fitsproof | What fitsproof adds |
|---|---|---|---|---|
| **aura** (Grevix/aura, Rust, MIT/Apache-2.0) | 4 | 2026-09-03 | Kernel-level budget enforcement (cgroup v2 / Win32 Job Object), context-ladder degradation, ollama model discovery, NVMe/GPU/SIMD diagnostics — enforcement is more aggressive than fitsproof's in-process gate | No on-device calibration or held-out MAPE. BENCHMARK.md (c2-p2 verified, confirmed through c8-p2; 0 commits since 2026-09-03): `qwen3:8b` with a 4.00 GB Win32 Job Object budget reporting `Peak Working Set: 4.92 GB` — 23% over declared budget — with no violation flag and no failing assertion. No embeddable `plan`/`admit` API, no OpenAI/MCP plugin surfaces |

## Determinism checking

| Tool | Stars | Last push | What it does better than fitsproof | What fitsproof adds |
|---|---|---|---|---|
| **detllm** (tommasocerruti) | 22 | 2026-08-20 | Capability-gated determinism tier reporting; repro packs | Determinism checking only; no memory prediction or enforcement |

## The claim

fitsproof's specific claim: **predict from on-device measurement + enforce with explicit
degradation + prove with a stress harness**. No tool in the table above does all three.

Narrowed after the pass-3 scan (docs/RESEARCH.md): aura already does enforcement
(kernel-level). RAMLLM (new in c8-p2) does calibrated prediction with ±2–4% accuracy
on real hardware. What remains unclaimed by anyone — including aura and RAMLLM — is
the combination of on-device calibration with a *published held-out MAPE*, a
*zero-violation stress harness as a repo test*, and an *embeddable admission API*.

Each ingredient exists: ridgepoint predicts (for other hardware). llama.cpp runs inference.
aura enforces (at the OS level). RAMLLM plans and governs (with empirical calibration).
detllm checks determinism. fitsproof assembles them into a single enforced contract:
`probe → calibrate → plan → admit (enforce) → verify (prove)`.

Where each tool beats us (honest accounting):
- **Speed:** llama.cpp, vLLM, KTransformers are orders of magnitude faster.
- **Prediction accuracy:** RAMLLM achieves ±2–4% tok/s error on real hardware;
  fitsproof's MAPE on held-out configs is ~46–65% (published in EVIDENCE.md and
  README). The error is one-sided (over-predicts), which keeps the refusal gate safe,
  but the false-degradation rate is high.
- **Model coverage:** llama.cpp supports hundreds of GGUF models; fitsproof ships one
  synthetic reference model.
- **GPU breadth:** vLLM and ridgepoint are calibrated to real A100/H100 hardware.
- **Production readiness:** vLLM has a full serving API and continuous batching;
  fitsproof has a correctness-first NumPy engine.

The claim is the contract, not the speed.
