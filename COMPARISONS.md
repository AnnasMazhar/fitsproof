# COMPARISONS

fitsproof is a resource contract engine. It does not compete on speed.
The tools below beat fitsproof on speed, model coverage, and hardware breadth.
We name them here so reviewers do not have to look them up.

Star counts and versions retrieved from the GitHub REST API and PyPI at
2026-09-29T00:31:26Z (cycle 5 pass 2 refresh; +42 llama.cpp, +26 vLLM, +2 KTransformers
vs the c4-p2 12:30Z snapshot; all other repos unchanged).
Six new search queries run; three new candidates evaluated (Emmimal/context-engine 197★ —
token-budget pipeline tool, different layer; CryptoGuy1/BoundedEdge — empty repo;
ashcakeancient7671/aura — README-only push, confirmed consumer Windows wrapper);
no new tool entered the comparison table.

## Inference engines

| Tool | Stars | Latest | What it does better than fitsproof | What fitsproof adds |
|---|---|---|---|---|
| **llama.cpp** (ggml-org/llama.cpp) | 129,804 | v0.5.0 (2026-09-23) | Mature (3+ years), broad model support (GGUF), fast CPU kernels (GGML), broad quant support, GPU offload | No enforced resource contract; silent OOM documented; no calibrated prediction interval |
| **vLLM** (vllm-project/vllm) | 92,887 | v0.30.0 (2026-09-22) | GPU serving, high throughput, PagedAttention, continuous batching, 100+ model support | No enforced memory budget; targets A100/H100, not 4–8 GB VRAM class |
| **KTransformers** (kvcache-ai/ktransformers) | 19,546 | v0.7.1 (2026-09-15) | CPU/GPU hybrid MoE, Intel AMX kernels, runs DeepSeek-671B on ~14 GB VRAM. *Requires 128 GB RAM, AMX, CUDA/ROCm.* | Does not serve the 16–32 GB RAM class; no calibrated contract; no stress harness |

## Prediction and sizing tools

| Tool | Stars | Last push | What it does better than fitsproof | What fitsproof adds |
|---|---|---|---|---|
| **ridgepoint** (Isk4R1oT/ridgepoint, PyPI v0.1.2) | 1 | 2026-09-08 | Calibrated VRAM prediction for GPU (A100/H100), KV cache formula correct for GQA and MLA, ~1% accuracy vs real vLLM, per-field `calibrated` flag | GPU-only calibration; no enforcement gate; no RSS proof harness |
| **llm-inference-calculator** (pochenai) | 21 | 2026-09-09 | Two-phase roofline (prefill compute-bound TTFT + decode bandwidth-bound TPOT), MoE sparsity, spec-decoding modelling | No calibration; no enforcement; targets data-centre multi-GPU |
| **llm-roofline** (Pluenet-Killian) | 0 | 2026-06-20 | Decode throughput floor = bytes_per_token / bandwidth, per GPU | Throughput floor only; no memory prediction; no enforcement; inactive |
| **hardware-aware-llm-runtime** (JohnScheuer) | 0 | 2026-06-25 | Hardware-calibrated roofline with empirical fitting; predicts optimal batch within ~1 | Throughput focus; no enforcement; no stress harness; inactive |
| **llm-vram-calculator** (Shun-Calvin) | 1 | 2026-09-26 | Formula-based VRAM/TTFT/tok/s across 100+ models × 70+ GPUs | Formula-based, not calibrated; GPU-only; no enforcement |

## Enforcement / orchestration

| Tool | Stars | Last push | What it does better than fitsproof | What fitsproof adds |
|---|---|---|---|---|
| **aura** (Grevix/aura, Rust, MIT/Apache-2.0) | 4 | 2026-09-03 | Kernel-level budget enforcement (cgroup v2 / Win32 Job Object), context-ladder degradation, ollama model discovery, NVMe/GPU/SIMD diagnostics — enforcement is more aggressive than fitsproof's in-process gate | No on-device calibration or held-out MAPE. BENCHMARK.md (c2-p2 verified, c3-p2/c4-p2/c5-p2 confirmed; 0 commits since 2026-09-03): `qwen3:8b` with a 4.00 GB Win32 Job Object budget reporting `Peak Working Set: 4.92 GB` — 23% over declared budget — with no violation flag and no failing assertion. No embeddable `plan`/`admit` API, no OpenAI/MCP plugin surfaces |

## Determinism checking

| Tool | Stars | Last push | What it does better than fitsproof | What fitsproof adds |
|---|---|---|---|---|
| **detllm** (tommasocerruti) | 20 | 2026-08-20 | Capability-gated determinism tier reporting; repro packs | Determinism checking only; no memory prediction or enforcement |

## The claim

fitsproof's specific claim: **predict from on-device measurement + enforce with explicit
degradation + prove with a stress harness**. No tool in the table above does all three.

Narrowed after the pass-3 scan (docs/RESEARCH.md): aura already does enforcement
(kernel-level). What remains unclaimed by anyone — including aura — is the combination of
on-device calibration with a published held-out MAPE, a zero-violation stress harness as a
repo test, and an embeddable admission API.

Each ingredient exists: ridgepoint predicts (for other hardware). llama.cpp runs inference.
aura enforces (at the OS level). detllm checks determinism. fitsproof assembles them into a
single enforced contract: `probe → calibrate → plan → admit (enforce) → verify (prove)`.

Where each tool beats us (honest accounting):
- **Speed:** llama.cpp, vLLM, KTransformers are orders of magnitude faster.
- **Model coverage:** llama.cpp supports hundreds of GGUF models; fitsproof ships one
  synthetic reference model.
- **GPU breadth:** vLLM and ridgepoint are calibrated to real A100/H100 hardware.
- **Production readiness:** vLLM has a full serving API and continuous batching;
  fitsproof has a correctness-first NumPy engine.

The claim is the contract, not the speed.
