# COMPARISONS

fitsproof is a resource contract engine. It does not compete on speed.
The tools below beat fitsproof on speed, model coverage, and hardware breadth.
We name them here so reviewers do not have to look them up.

| Tool | What it does better than fitsproof | What fitsproof adds |
|---|---|---|
| **llama.cpp** | Mature (3+ years), broad model support (GGUF), fast CPU kernels (GGML), broad quant support, GPU offload | No enforced resource contract; silent OOM documented; no calibrated prediction interval |
| **vLLM** | GPU serving, high throughput, PagedAttention, continuous batching, 100+ model support | No enforced memory budget; targets A100/H100, not 4–8 GB VRAM class |
| **KTransformers** | CPU/GPU hybrid MoE, Intel AMX kernels, runs DeepSeek-671B on ~14 GB VRAM. *Requires 128 GB RAM, AMX, CUDA/ROCm.* | Does not serve the 16–32 GB RAM class; no calibrated contract |
| **ridgepoint** (PyPI) | Calibrated VRAM prediction for GPU (A100/H100), KV cache formula correct for GQA and MLA, ~1% accuracy vs real vLLM | GPU-only, no enforcement, no stress harness |
| **Strata** | Consumer packaging, one-click install | Requires 12 GB+ VRAM and 64 GB RAM |
| **detllm** | Determinism tier verification | Does not predict or enforce memory budgets |

fitsproof's specific claim: **predict from on-device measurement + enforce with explicit
degradation + prove with a stress harness**. None of the above does all three.
