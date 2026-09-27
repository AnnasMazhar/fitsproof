# COMPARISONS

fitsproof is a resource contract engine. It does not compete on speed.
The tools below beat fitsproof on speed, model coverage, and hardware breadth.
We name them here so reviewers do not have to look them up.

Star counts and versions retrieved from the GitHub REST API and PyPI at
2026-09-27T13:00 UTC.

## Inference engines

| Tool | Stars | Latest | What it does better than fitsproof | What fitsproof adds |
|---|---|---|---|---|
| **llama.cpp** (ggml-org/llama.cpp) | 129,661 | v0.5.0 (2026-09-23) | Mature (3+ years), broad model support (GGUF), fast CPU kernels (GGML), broad quant support, GPU offload | No enforced resource contract; silent OOM documented; no calibrated prediction interval |
| **vLLM** (vllm-project/vllm) | 92,766 | v0.30.0 (2026-09-22) | GPU serving, high throughput, PagedAttention, continuous batching, 100+ model support | No enforced memory budget; targets A100/H100, not 4–8 GB VRAM class |
| **KTransformers** (kvcache-ai/ktransformers) | 19,538 | v0.7.1 (2026-09-15) | CPU/GPU hybrid MoE, Intel AMX kernels, runs DeepSeek-671B on ~14 GB VRAM. *Requires 128 GB RAM, AMX, CUDA/ROCm.* | Does not serve the 16–32 GB RAM class; no calibrated contract; no stress harness |

## Prediction and sizing tools

| Tool | Stars | Last push | What it does better than fitsproof | What fitsproof adds |
|---|---|---|---|---|
| **ridgepoint** (Isk4R1oT/ridgepoint, PyPI v0.1.2) | 1 | 2026-09-08 | Calibrated VRAM prediction for GPU (A100/H100), KV cache formula correct for GQA and MLA, ~1% accuracy vs real vLLM, per-field `calibrated` flag | GPU-only calibration; no enforcement gate; no RSS proof harness |
| **llm-inference-calculator** (pochenai) | 20 | 2026-09-09 | Two-phase roofline (prefill compute-bound TTFT + decode bandwidth-bound TPOT), MoE sparsity, spec-decoding modelling | No calibration; no enforcement; targets data-centre multi-GPU |
| **llm-roofline** (Pluenet-Killian) | 0 | 2026-06-20 | Decode throughput floor = bytes_per_token / bandwidth, per GPU | Throughput floor only; no memory prediction; no enforcement; inactive |
| **hardware-aware-llm-runtime** (JohnScheuer) | 0 | 2026-06-25 | Hardware-calibrated roofline with empirical fitting; predicts optimal batch within ~1 | Throughput focus; no enforcement; no stress harness; inactive |
| **llm-vram-calculator** (Shun-Calvin) | 1 | 2026-09-26 | Formula-based VRAM/TTFT/tok/s across 100+ models × 70+ GPUs | Formula-based, not calibrated; GPU-only; no enforcement |

## Determinism checking

| Tool | Stars | Last push | What it does better than fitsproof | What fitsproof adds |
|---|---|---|---|---|
| **detllm** (tommasocerruti) | 20 | 2026-08-20 | Capability-gated determinism tier reporting; repro packs | Determinism checking only; no memory prediction or enforcement |

## The claim

fitsproof's specific claim: **predict from on-device measurement + enforce with explicit
degradation + prove with a stress harness**. No tool in the table above does all three.

Each ingredient exists: ridgepoint predicts (for other hardware). llama.cpp runs inference.
detllm checks determinism. fitsproof assembles them into a single enforced contract:
`probe → calibrate → plan → admit (enforce) → verify (prove)`.

Where each tool beats us (honest accounting):
- **Speed:** llama.cpp, vLLM, KTransformers are orders of magnitude faster.
- **Model coverage:** llama.cpp supports hundreds of GGUF models; fitsproof ships one
  synthetic reference model.
- **GPU breadth:** vLLM and ridgepoint are calibrated to real A100/H100 hardware.
- **Production readiness:** vLLM has a full serving API and continuous batching;
  fitsproof has a correctness-first NumPy engine.

The claim is the contract, not the speed.
