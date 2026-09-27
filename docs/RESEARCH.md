# RESEARCH — fitsproof v0.1

All links verified to resolve at the time of writing (2026-09-27).
Pass 1 (2026-09-26): sources 1–15, ground truth. Updated with 7 new sources and
extended falsification in pass 1 (2026-09-27).
Pass 2 (2026-09-27): ecosystem scan, comparison table, gap analysis.
**Cycle 2 pass 1 (2026-09-27): ground truth for the v0.2 MANDATE surfaces —
sources 23–34, deep treatment of 23/25/27/28/29, link re-verification, new
falsification section. See "Cycle 2 — Pass 1" at the end of this file.**
Star counts retrieved via GitHub REST API at 2026-09-27T13:00 UTC. Version dates from
GitHub releases API and PyPI. Each citation is attached to the specific claim it supports.

---

## Sources

### 1. Roofline Model (core design driver)
**Williams, S., Waterman, A., Patterson, D. (2009).** Roofline: An Insightful Visual
Performance Model for Floating-Point Programs and Multicore Architectures.
*Communications of the ACM*, 52(4), 65–76.
https://dl.acm.org/doi/10.1145/1498765.1498785

**Claim it supports:** Decode throughput is memory-bandwidth-bound for single-batch
inference. `tok/s = effective_bandwidth / bytes_per_token_of_weights` is the roofline
decode formula. The roofline model defines the crossing point between bandwidth-bound
and compute-bound regimes.

**Equation extracted (arithmetic intensity):**
```
Performance = min(peak_FLOPS, bandwidth * arithmetic_intensity)
Arithmetic intensity (decode) ≈ 2 * n_params / weight_bytes = 0.5 FLOP/byte (float32)
```
CPU DRAM bandwidth ≈ 20–60 GB/s; compute ≈ 100–400 GFLOPS → bandwidth-bound for
arithmetic intensity < peak_FLOPS/bandwidth ≈ 5–10 FLOP/byte.

**Assumptions:** single-batch decode (batch=1). Prefill at large batch is compute-bound.

**Known failure mode:** at large batch, the workload crosses to compute-bound and
this formula underpredicts throughput. Not the consumer use case we target.

---

### 2. Memory-Bandwidth-Bound Decode (FlexGen)
**Sheng, Y., Zheng, L., Yuan, B., et al. (2023).** FlexGen: High-Throughput Generative
Inference of Large Language Models with a Single GPU.
*ICML 2023*. arXiv:2303.06865.
https://arxiv.org/abs/2303.06865

**Claim it supports:** For single-batch LLM decode, memory bandwidth is the bottleneck.
FlexGen Section 3.1 explicitly derives `throughput ∝ bandwidth / model_size`.

**Equation (Section 3.1):**
```
decode_tok_s ≈ effective_bandwidth / weight_bytes
```
This is the direct source for `cost.py:decode_tok_s`.

**Assumptions:** batch=1, model fits in memory, KV cache overhead small relative to weights.

**Known failure mode:** high-batch serving (not the fitsproof target).

---

### 3. RoPE Positional Encoding (core algorithm)
**Su, J., Lu, Y., Pan, S., et al. (2023).** RoFormer: Enhanced Transformer with
Rotary Position Embedding. *Neurocomputing*, 568.
arXiv:2104.09864.
https://arxiv.org/abs/2104.09864

**Claim it supports:** The frequency schedule and rotation formula implemented in
`attention.py:_rope_freqs` and `attention.py:apply_rope`.

**Equation 15 (frequency schedule):**
```
theta_i = base^{-2i/d},  i in [0, d/2),  base = 10000
```

**Equation 34 (rotation):**
```
[x_{2i}, x_{2i+1}] → [x_{2i}*cos - x_{2i+1}*sin,
                       x_{2i}*sin + x_{2i+1}*cos]
```
where cos/sin are evaluated at position-dependent angles `pos * theta_i`.

**Assumptions:** head_dim even; base=10000 (empirically chosen, not derived).

**Known failure mode:** without NTK-aware scaling (arXiv:2306.15595), RoPE
degrades at context lengths beyond the training max. Not implemented in v0.1.

---

### 4. Grouped Query Attention (GQA)
**Ainslie, J., Lee-Thorp, J., de Jong, M., et al. (2023).** GQA: Training Generalized
Multi-Query Transformer Models from Multi-Head Checkpoints. *EMNLP 2023*.
arXiv:2305.13245.
https://arxiv.org/abs/2305.13245

**Claim it supports:**
- The GQA attention implementation in `attention.py:_sdp_attention` (KV head sharing).
- The KV cache size formula: `2 * n_layers * n_kv_heads * seq * head_dim * bytes`.

**Method extracted:** with `n_kv_heads < n_heads`, each KV head is shared across
`n_heads // n_kv_heads` query heads. K/V broadcast before dot-product.

**Assumptions:** `n_heads` divisible by `n_kv_heads`.

**Known failure mode:** Multi-head Latent Attention (MLA, used in DeepSeek) compresses
KV further; our formula does not account for MLA.

---

### 5. Scaled Dot-Product Attention
**Vaswani, A., Shazeer, N., Parmar, N., et al. (2017).** Attention Is All You Need.
*NeurIPS 2017*. arXiv:1706.03762.
https://arxiv.org/abs/1706.03762

**Claim it supports:** The `1/sqrt(d_k)` scaling factor in `_sdp_attention`.

**Equation 1:**
```
Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
```
Without the `1/sqrt(d_k)` factor, dot products grow with dimension, pushing
softmax into near-zero gradient regions.

**Assumptions:** keys and queries drawn from distributions with unit variance
(which the `1/sqrt(d_k)` scale restores after the dot product).

---

### 6. Scaling Laws — Parameter Counting
**Kaplan, J., McCandlish, S., Henighan, T., et al. (2020).** Scaling Laws for Neural
Language Models. arXiv:2001.08361.
https://arxiv.org/abs/2001.08361

**Claim it supports:** The prefill TTFT formula: `TTFT ≈ 2 * n_params * seq_len / FLOPS`.
Appendix D derives that inference FLOPS ≈ 2 * n_params per token.

**Equation (Appendix D):**
```
FLOPs_per_token ≈ 2 * N   where N = n_parameters (excluding embeddings)
```

---

### 7. GPTQ — Int8 Symmetric Quantisation
**Frantar, E., Ashkboos, S., Hoefler, T., Alistarh, D. (2022).** GPTQ: Accurate
Post-Training Quantization for Generative Pre-trained Transformers.
arXiv:2210.17323.
https://arxiv.org/abs/2210.17323

**Claim it supports:** The symmetric int8 quantisation in `quant.py:_int8_sym_quant`.
GPTQ uses per-channel scaling (each output channel has its own scale), which we adopt.

**Method extracted:**
```
scale_c = max(|W_c|) / (2^{b-1} - 1)  where b=8 for int8, clip range [-127, 127]
q_c = round(W_c / scale_c)
```
Using 127 (not 128) keeps the range symmetric around zero.

---

### 8. AWQ — Activation-Aware Quantisation
**Lin, J., Tang, J., Tang, H., et al. (2023).** AWQ: Activation-aware Weight
Quantization for LLM Compression and Acceleration.
arXiv:2306.00978.
https://arxiv.org/abs/2306.00978

**Claim it supports:** The asymmetric int8 quantisation design in `quant.py:_int8_asym_quant`.
AWQ demonstrates that asymmetric quantisation reduces error for skewed weight distributions.

**Method extracted (asymmetric):**
```
scale = (max - min) / 255
zp = round(-min / scale)
q = clip(round(w / scale) + zp, 0, 255)
```
fitsproof implements this as the `int8_asym` mode.

---

### 9. Speculative Decoding
**Leviathan, Y., Kalman, M., Matias, Y. (2023).** Fast Inference from Transformers
via Speculative Decoding. *ICML 2023*. arXiv:2211.17192.
https://arxiv.org/abs/2211.17192

**Claim it supports:** The correctness property of `speculative.py:speculative_generate`:
under greedy decoding, speculative output is identical to non-speculative greedy output.

**Key theorem (paraphrased from Algorithm 1):** when draft tokens are accepted by the
target, they are identical to target greedy tokens. When a draft token is rejected, the
target's greedy token is emitted. Either way, the output stream is the target greedy sequence.

**Assumptions:** greedy (temperature=0) target. Probabilistic acceptance (temperature>0)
requires rejection sampling (Algorithm 1) — not implemented in v0.1.

---

### 10. SwiGLU / Gated Feed-Forward
**Shazeer, N. (2020).** GLU Variants Improve Transformer. arXiv:2002.05202.
https://arxiv.org/abs/2002.05202

**Claim it supports:** The SwiGLU FFN in `transformer.py:swiglu_ffn`.

**Equation:**
```
FFN(x) = (SiLU(x * W_gate) * (x * W_up)) * W_down
SiLU(z) = z * sigmoid(z)
```

---

### 11. RMSNorm
**Zhang, B., Sennrich, R. (2019).** Root Mean Square Layer Normalization.
*NeurIPS 2019*. arXiv:1910.07467.
https://arxiv.org/abs/1910.07467

**Claim it supports:** `transformer.py:rms_norm`.

**Equation 4:**
```
RMSNorm(x) = x / RMS(x) * weight,  RMS(x) = sqrt(mean(x^2) + eps)
```
Omits mean subtraction (unlike LayerNorm), reducing computation.

---

### 12. STREAM — Memory Bandwidth Measurement
**McCalpin, J. D. (1995).** Memory bandwidth and machine balance in current
high-performance computers. *IEEE Technical Committee on Computer Architecture*.
https://www.cs.virginia.edu/stream/ref.html

**Claim it supports:** The STREAM triad kernel in `probe.py:_measure_bandwidth`.

**Triad kernel:**
```
A[i] = B[i] + scalar * C[i]   (reads 2 arrays, writes 1)
bandwidth = 3 * n * sizeof(float64) / elapsed
```
The triad is the standard benchmark for sustainable DRAM bandwidth (not burst).

---

### 13. GGML Int4 k-Quants
**Vakulya, G. (2023).** Introduce k-quants (quantisation improvements for llama.cpp).
GitHub Pull Request #1684.
https://github.com/ggerganov/llama.cpp/pull/1684

**Claim it supports:** The int4 pack/unpack strategy in `quant.py:_int4_pack` and
`_int4_unpack`, and the use of range [-7, 7] for symmetric int4 (not [-8, 7]).

**Method:** pack two int4 values per byte (high nibble = first, low nibble = second);
clip to [-7, 7] rather than [-8, 7] to keep the range symmetric around 0.

---

### 14. Determinism Tiers (detllm)
**Cerruti, T. (2024).** detllm: deterministic LLM inference checking with
capability-gated guarantee tiers.
https://github.com/tommasocerruti/detllm

**Claim it supports:** The determinism tier framework in `verify.py:DeterminismTier`.

**Method:** three tiers — Tier 0 (artifact reproducibility), Tier 1 (run-to-run output
repeatability at fixed seed), Tier 2 (Tier 1 + logprob equality). fitsproof always
reports the tier actually achieved, never claims higher.

---

### 15. PagedAttention / KV Cache Efficiency
**Kwon, W., Li, Z., Zhuang, S., et al. (2023).** Efficient Memory Management for
Large Language Model Serving with PagedAttention. *SOSP 2023*. arXiv:2309.06180.
https://arxiv.org/abs/2309.06180

**Claim it supports:** The KV cache memory analysis in `cost.py` and the framing of
KV cache as a first-class resource to be budgeted. PagedAttention demonstrates that
KV cache is the dominant memory consumer for long-context serving.

---

### 16. Position Interpolation / NTK-aware RoPE Scaling (documented limitation)
**Chen, S., Wong, S., Chen, L., Tian, Y. (2023).** Extending Context Window of Large
Language Models via Positional Interpolation.
arXiv:2306.15595.
https://arxiv.org/abs/2306.15595

**Claim it supports:** The limitation stated in `README.md` ("No NTK-aware RoPE scaling")
and in source 3 (RoPE known failure mode). The fitsproof reference model is trained with
`max_seq_len = 512`; inference beyond that bound degrades without Position Interpolation.

**Method extracted (Position Interpolation, Section 3.1):**
```
PI maps position index p → p * (L / L')
where L = original training context length,
      L' = target extended context length (L' > L)
```
This linearly down-scales position indices rather than extrapolating, keeping rotary
angles inside the range seen at training time.

**Theoretical result (Section 3.2):** the upper bound on the attention score perturbation
from interpolation is at least ~600× smaller than from extrapolation. Derives from the
Lipschitz bound on RoPE-rotated query/key dot products.

**Assumptions:**
- Some fine-tuning (≥1000 steps) is required after PI to recover full accuracy.
- NTK-aware scaling (the community variant, not in this paper) avoids fine-tuning at the
  cost of a frequency-domain modification: `theta_i = base^{-2i/d} * (L'/L)^{2i/(d-2)}`.
  fitsproof does not implement either variant in v0.1.

**Known failure modes per the literature:**
- Without PI or fine-tuning, RoPE extrapolation yields catastrophically high attention
  scores (the paper's term) for positions beyond the training window.
- The NTK variant (community blog, 2023) is more practical (no fine-tuning needed for
  moderate extensions) but produces a slightly lower-quality extrapolation than PI + fine-tune.

---

### 17. LLM-42: Determinism via Verify-Rollback (design driver for determinism tiers)
**Gond, R., Kamath, A. K., Ramjee, R., Panwar, A. (2026).** LLM-42: Enabling Determinism
in LLM Inference with Verified Speculation.
arXiv:2601.17768.
https://arxiv.org/abs/2601.17768

**Claim it supports:** The determinism tier framework in `verify.py` and the claim that
NumPy-only computation on a single CPU is Tier 1 deterministic by construction (no dynamic
batching, no GPU kernel selection).

**Problem statement:** floating-point non-associativity under dynamic batching changes
reduction order across batch sizes, producing different outputs for the same input.
Standard fix (disabling dynamic batching) degrades throughput. Kernel-hardening (batch-
invariant kernels) couples determinism tightly to kernel design.

**Method extracted (LLM-42 architecture):**
```
FAST PATH: decode tokens with non-deterministic GPU kernels (high throughput)
VERIFY STEP: replay candidate tokens under a fixed-shape reduction schedule
             (same batch shape → same reduction → same rounding)
COMMIT: if verify agrees with fast path → commit, no overhead
ROLLBACK: if verify disagrees → emit the verified token, discard the fast-path token
```
Overhead is proportional to the fraction of traffic that diverges (typically low).

**Key distinction from detllm (source 14):** LLM-42 enforces determinism at runtime
(corrects non-determinism as it occurs); detllm reports what tier a backend already achieves.
fitsproof uses the detllm reporting model — it never claims a tier the NumPy backend
does not actually achieve.

**Assumptions:** most GPU kernel reductions are shape-consistent (empirical observation,
Section 4); the verify step's fixed-shape schedule is cheaper than a full second forward
pass. Neither assumption applies to fitsproof (NumPy, single process, no GPU).

**Known failure modes:**
- Rollback rate increases with batch size variance; highly dynamic workloads see more
  overhead.
- Not applicable to NumPy: our backend is deterministic by construction (single-threaded,
  no CUDA, no atomic reductions).

---

### 18. Bit-Exact AI Inference Verification (design driver for tier reporting)
**Cankaya, N. (2026).** Bit-Exact AI Inference Verification Without Performance Tradeoffs.
*ICML 2026 TAIGR workshop, best paper award.*
arXiv:2606.00279.
https://arxiv.org/abs/2606.00279

**Claim it supports:** The distinction in `verify.py` between *determinism* (same output
per run on the same hardware) and *invariance* (same output across different hardware).
fitsproof claims the former, not the latter.

**Key finding:** modern inference engines (vLLM, HF transformers) produce deterministic
but non-invariant outputs — i.e., the output is repeatable on the same machine but differs
across GPU variants — without needing to set any performance-compromising flags, provided
the right re-computation information is available and no atomic GPU functions are used.

**Method extracted (Section 3.2 — software-only bitwise emulation):**
```
Observation: accumulated rounding errors are an auditable signature of
             (software stack, hardware configuration) → use for verification,
             not just an obstacle to avoid.
Approach: emulate LLM inference in software to reproduce bit-exact outputs
          across NVIDIA GPU variants without identical hardware.
Key: if atomic functions (non-deterministic CUDA atomics) are absent,
     the rounding sequence is fully determined by the computation graph.
```

**Distinction from LLM-42 (source 17):**
- LLM-42: enforce determinism within one hardware config under dynamic batching.
- arXiv:2606.00279: prove bit-exact cross-hardware verifiability for AI governance/audit.
- fitsproof: report the tier actually achieved (Tier 1 = run-to-run on this host); does
  not claim cross-hardware invariance.

**Assumptions:** no atomic CUDA operations; the computation graph is available for
replay. Neither assumption applies to our NumPy path (we are already deterministic; the
paper solves a GPU-specific problem).

**Known failure modes:** atomic GPU operations break the bit-exact property; any kernel
that uses CUDA atomics for reductions will produce non-invariant output regardless.

---

### 19. Numerical Sources of Nondeterminism in LLM Inference (NeurIPS 2025)
**Yuan, J., Li, H., Ding, X., et al. (2025).** Understanding and Mitigating Numerical
Sources of Nondeterminism in LLM Inference.
*NeurIPS 2025.* arXiv:2506.09501.
https://arxiv.org/abs/2506.09501

**Claim it supports:** The assertion in `verify.py` that the NumPy (float64, single-process)
backend achieves Tier 1 determinism; and the documented decision to store weights and
activations at a consistent precision rather than relying on bfloat16.

**Key empirical result:** under bfloat16 with greedy decoding, a reasoning model can
exhibit up to 9% accuracy variation and 9,000-token response-length variation simply from
changing GPU count, type, or evaluation batch size. Root cause: non-associativity of
bfloat16 arithmetic under different parallel reduction schedules.

**Method (LayerCast, Section 4):**
```
Store weights in 16-bit (memory-efficient)
Perform all computations in FP32 (numerically stable)
→ Eliminates nondeterminism from precision-induced rounding differences
  while keeping memory footprint at 16-bit weight size
```

**Relevance to fitsproof:** fitsproof uses float32/float64 NumPy throughout — LayerCast's
property holds trivially for our backend. The paper quantifies the failure mode
(nondeterminism from reduced precision) that fitsproof avoids by design.

**Assumptions:** LayerCast does not eliminate nondeterminism from dynamic batching (see
source 17 for that). Precision is a necessary but not sufficient condition for
run-to-run determinism in a GPU context.

**Known failure modes:** even FP32 is not invariant across GPU architectures (different
hardware uses different FMA fusion decisions); true invariance requires bit-exact emulation
(source 18).

---

### 20. PowerInfer: Consumer-GPU LLM Serving (competitor context)
**Song, Y., Mi, Z., Xie, H., Chen, H. (2024).** PowerInfer: Fast Large Language Model
Serving with a Consumer-grade GPU.
*SOSP 2024.* arXiv:2312.12456.
https://arxiv.org/abs/2312.12456

**Claim it supports:** The market positioning in `COMPARISONS.md`: PowerInfer targets
consumer GPUs (RTX 4090) via CPU-GPU split, and fitsproof targets a different constraint
(the resource contract, not speed optimisation). Also supports the Alternatives Considered
entry for CPU-GPU hybrid approaches.

**Method extracted (hot/cold neuron split, Section 3):**
```
Power-law activation skewness: ~10% of neurons are "hot" (activated in >90% of inputs)
Hot neurons:  preloaded to GPU VRAM for low-latency access
Cold neurons: computed on CPU (vary per input, cannot be preloaded)
GPU memory demand ≈ 10% of model size (for the hot-neuron fraction)
Adaptive predictor: predicts cold neuron activations before they are needed,
                    further reducing GPU↔CPU transfer latency
```

**Key performance result:** up to 11.69× speedup over llama.cpp on RTX 4090 for OPT-175B;
OPT-30B reaches 82% of an A100's token generation rate.

**Why fitsproof does not use this approach:**
1. Requires profiling neuron activation statistics per model (not available for arbitrary models).
2. Adaptive predictor adds training overhead outside the fitsproof scope.
3. Our contract claim is about memory correctness and budget enforcement, not maximum speed.
   PowerInfer does not enforce a declared budget or emit an explicit degradation record.

**Known failure modes per the literature:**
- Hot/cold split degrades when activation distribution varies significantly across inputs
  (e.g., code vs. prose). Predictor accuracy drops, triggering more CPU↔GPU transfers.
- The approach requires a consumer GPU (PCIe bandwidth-limited); PCI-e 3.0 × 16 caps
  transfer at ~16 GB/s, which becomes the bottleneck for the cold-neuron fraction at high
  decode rates.

---

### 21. KTransformers: CPU/GPU Hybrid MoE Inference (competitor context)
**Chen, H., Xie, W., Zhang, B., et al. (2025).** KTransformers: Unleashing the Full
Potential of CPU/GPU Hybrid Inference for MoE Models.
*SOSP 2025 (The 31st Symposium on Operating Systems Principles).*
DOI: 10.1145/3731569.3764843.
https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/

**Claim it supports:** The market scan finding in MARKET-VERDICTS: KTransformers requires
128 GB RAM and AMX, serves the MoE/671B class, not our 4–8 GB VRAM target class.
Also supports `COMPARISONS.md`.

**Method extracted (Section 3 — AMX kernel + Expert Deferral):**
```
Architecture: attention layers + KV cache → GPU (bandwidth-intensive, fits in VRAM)
             expert weight matrices → DRAM (sparse activation: ~6-8 of 256 experts active)
AMX kernels: Intel Advanced Matrix Extensions tiling for matrix-vector multiply in DRAM
             2× FLOP/cycle vs AVX-512 for the expert GEMM shape
Expert Deferral: when an expert's output is not on the critical path, defer its
                 GPU-side follow-up computation until after the next CPU batch completes
                 → CPU utilisation rises from <75% to ~100%
```

**Performance results:** 4.62–19.74× prefill speedup, 1.25–4.09× decode speedup over
llama.cpp (OPT baseline). Runs DeepSeek-V3/R1 (671B) with ~14 GB VRAM + 128 GB RAM.

**Why fitsproof does not compete here:**
- AMX requires Intel Sapphire Rapids (2023+); our Quadro M2000 box has no AMX.
- KTransformers has no resource contract layer: it does not predict peak memory from
  on-device measurement, does not enforce a declared budget, and does not emit a degradation
  record when mode transitions occur.
- Our target hardware class (4 GB VRAM, 31 GB RAM) cannot run 671B MoE models regardless.

**Known failure modes per the literature:**
- Expert Deferral introduces a 0.5% average accuracy drop (accepted trade-off for throughput).
- PCIe bandwidth (16 GB/s for PCIe 4.0 ×16) caps the effective expert-load rate at high
  concurrency; model becomes PCIe-bound rather than compute-bound for very active expert sets.
- AMX is unavailable on older Intel CPUs and on AMD; the optimised kernels do not fall back
  gracefully.

---

### 22. ridgepoint: Calibrated VRAM / Roofline Prediction (closest competitor)
**ridgepoint (2026).** LLM inference sizing calibrated to ~1% against real vLLM on A100/H100.
PyPI version 0.1.1.
https://pypi.org/project/ridgepoint/0.1.1/

**Claim it supports:** The competitor positioning: ridgepoint is the closest tool to
fitsproof's prediction layer — it ships KV-cache-correct sizing (GQA and MLA to the byte)
with roofline intervals and per-field `calibrated` flags. The differentiator is that
ridgepoint predicts (for A100/H100 class); fitsproof also enforces a budget, degrades
loudly, and proves compliance by measuring peak memory on the user's actual machine.

**What ridgepoint ships (per its own documentation):**
- Engine-aware VRAM capacity modelling (not just weights × bytes/param)
- KV cache formula correct to the byte for both GQA and MLA attention types
- Roofline-based throughput intervals (not point estimates)
- Calibrated against real vLLM runs on A100/H100 (~1% memory accuracy)
- Per-field `calibrated` flags: each output value carries a flag indicating whether it
  was derived from a calibrated measurement or from a formula

**Structural difference from fitsproof:**
```
ridgepoint: predict(model_spec, gpu_spec) → VRAM estimate
            calibration = offline, against A100/H100, not the user's machine
            no budget enforcement, no degradation path, no RSS proof harness

fitsproof:  probe(this_machine) → machine_profile
            calibrate(machine_profile) → fitted constants, MAPE on held-out configs
            plan(model, quant, context, budget) → verdict + CI
            admit(plan) → enforce or degrade loudly
            verify() → measure peak RSS vs declared budget
```

**Why ridgepoint does not close the gap:** it does not run on the user's machine to
calibrate to their specific hardware, does not enforce a budget, and does not emit a
measured proof that peak memory stayed within the declared contract.

---

## Alternatives Considered

| Approach | Why rejected |
|---|---|
| Wrapping an existing engine (llama.cpp, vLLM) | Would not make the contract testable from first principles; the guarantee depends on being able to measure our own peak memory and verify the code path |
| Using torch / transformers | Contradicts the no-dependency design; also disqualifies pure NumPy determinism claims |
| CUDA kernels for performance | No CUDA toolkit on this machine (Quadro M2000, no nvcc); also not the product claim |
| Calibrated GPU-specific rooflines (like ridgepoint) | ridgepoint is calibrated to vLLM on A100/H100; we target 4–8 GB VRAM class which those models do not cover |
| Implementing MLA (Multi-head Latent Attention) | Not needed for the reference model; too large a scope increase for v0.1 |

---

## What Would Falsify This Design

1. **Measured bandwidth exceeds predicted throughput by >2x:** would indicate the
   roofline model is wrong for this CPU (e.g., tensor cores or AMX instructions give
   higher effective throughput than DRAM bandwidth). Mitigation: the calibration step
   fits `bandwidth_utilisation` from real measurements.

2. **KV cache dominates weight streaming at moderate context:** if `kv_cache_bytes`
   exceeds `weight_bytes` at common context lengths (e.g., 4096 tokens for a 7B model),
   the decode formula must include KV cache bandwidth. Current formula is accurate for
   the reference model (small) and short contexts.

3. **RSS measurement does not reflect peak allocations inside numpy:** on Linux, RSS
   is the high-water mark since process start. If allocations are freed before the
   post-call sample, the peak is missed. This is a known limitation and is documented
   in the README.

4. **Speculative decoding equality fails with a different target/draft pair:** the
   equality proof only holds at temperature=0 (greedy). If temperature > 0, the
   standard Algorithm 1 rejection sampling must be used (not implemented).

5. **Calibration MAPE on held-out configurations exceeds 20%:** would indicate the
   linear bandwidth model (source 2, FlexGen) is an insufficient fit for this hardware
   class. The calibration step fits `alpha` (bandwidth utilisation) and `beta` (constant
   overhead per forward pass) — two free parameters. If MAPE > 20% on held-out configs,
   the model needs a higher-order term or the assumptions (single-batch, weights-dominate)
   are violated. Observable from the `calibrate.py` output — we publish the number.

6. **Prediction interval coverage below 80%:** `calibrate.py` reports a bootstrap
   confidence interval on the held-out MAPE. If fewer than 80% of held-out configurations
   fall within the predicted interval, the interval is not calibrated and the contract's
   uncertainty claims are invalid. This is directly checkable by running the calibration
   harness.

7. **A configuration passes `admit` but RSS exceeds the declared budget:** would
   falsify the core claim of the repo. The stress harness (acceptance criterion 7) exists
   specifically to detect this. If any of the ≥20 configurations in the harness shows
   `measured_peak > declared_budget`, that is a critical finding — publish it; do not hide it.

8. **Tier 1 determinism violated for the NumPy backend:** running the same generation
   twice with the same seed on the same host should produce identical output. If it does
   not, there is a non-deterministic path in the NumPy code (e.g., dict ordering, OS
   thread scheduling affecting `np.random`). Observable by running `verify.py` twice
   and comparing outputs byte-for-byte. Sources 17 and 19 identify the GPU mechanisms
   that cause nondeterminism; any of those mechanisms appearing in our NumPy path
   (which has none of them) would be a new finding requiring investigation.

9. **ridgepoint's calibration transfers to our hardware class without re-measurement:**
   if ridgepoint's A100/H100-calibrated MAPE is also <5% on a Quadro M2000 (compute 5.2,
   GDDR5, PCIe 3.0), then on-device calibration adds no value over using a pre-calibrated
   tool. Observable: run ridgepoint's prediction on our machine and compare to fitsproof's
   own measured peak. We expect the transfer to fail (different memory technology,
   different PCIe generation, different DRAM channels) — but this is an empirical claim,
   not a derivation, and must be tested.

---

## Pass 2 — Ecosystem and Competition

### Methodology

Star counts and version dates were retrieved from the GitHub REST API
(`api.github.com/repos/{owner}/{repo}` and `/releases/latest`) and from the PyPI JSON
API (`pypi.org/pypi/{name}/json`). All figures are as of 2026-09-27T13:00 UTC and are
raw API values, not estimates. The llama.cpp repo was accessed via its numeric ID
(612354784) because the ggerganov→ggml-org rename returns a permanent redirect on the
named-path endpoint.

Nine tools were evaluated across two categories: (A) inference engines that overlap the
execution layer, and (B) prediction / sizing tools that overlap the contract layer. A
third micro-category (determinism checking) is covered by detllm.

---

### Category A — Inference engines

#### 1. llama.cpp
- **Repo:** https://github.com/ggml-org/llama.cpp (formerly ggerganov/llama.cpp)
- **Stars:** 129,661 (2026-09-27)
- **Latest release:** v0.5.0, published 2026-09-23
- **Language:** C/C++, Python bindings
- **Approach:** GGML tensor library, CPU-first with optional GPU offload; GGUF quantised
  models, broad model coverage (LLaMA, Mistral, Phi, Gemma, etc.), 2–8 bit k-quants.
- **What it does well:** mature (3+ years), lowest-friction path to CPU inference, broad
  quant support (Q2_K through Q8_0), `-ot` / `--cpu-moe` layer-level offload, runs
  everywhere without a GPU toolkit.
- **Gap it leaves:** documented silent failure modes: OOM produces a raw CUDA error
  with no preceding warning or degradation path; AMD fallback to CPU at 0.3 tok/s is
  silent (no log line indicating mode). No user-declared memory budget. No calibrated
  peak-memory prediction from on-device measurement. No stress harness asserting measured
  RSS ≤ declared limit.
- **What fitsproof does differently:** enforces an explicit budget declared by the user;
  any runtime mode change (GPU→CPU) emits a record naming what changed and why; a stress
  harness asserts zero budget violations across ≥20 configurations.

#### 2. vLLM
- **Repo:** https://github.com/vllm-project/vllm
- **Stars:** 92,766 (2026-09-27)
- **Latest release:** v0.30.0, published 2026-09-22
- **Language:** Python / CUDA
- **Approach:** PagedAttention for GPU memory efficiency, continuous batching, tensor
  parallelism, targets A100/H100/A10G class hardware.
- **What it does well:** highest single-node GPU throughput in the open ecosystem, full
  OpenAI-compatible API, production serving, broad model coverage.
- **Gap it leaves:** targets ≥40 GB GPU memory; has no mode for the 4–8 GB VRAM / 16–32 GB
  RAM class. Does not enforce a user-declared budget. `VLLM_BATCH_INVARIANT=1` achieves
  determinism with a documented performance trade-off (confirmed from vLLM docs); the flag
  is not the default, so most deployments are non-deterministic. No calibration from
  on-device measurement.
- **What fitsproof does differently:** CPU-first with explicit budget enforcement targeting
  the hardware class vLLM does not serve; calibration is per-machine, not per-GPU-model.

#### 3. KTransformers
- **Repo:** https://github.com/kvcache-ai/ktransformers
- **Stars:** 19,538 (2026-09-27)
- **Latest release:** v0.7.1, published 2026-09-15
- **Language:** Python / C++ / CUDA
- **Approach:** CPU/GPU hybrid for sparse MoE models; hot layers (attention, KV cache)
  on GPU, expert matrices in DRAM with Intel AMX kernels; Expert Deferral to maximise
  CPU utilisation. From SOSP 2025 paper (DOI: 10.1145/3731569.3764843).
- **What it does well:** runs DeepSeek-V3/R1 (671B) with ~14 GB VRAM + 128 GB RAM;
  1.25–4.09× decode over llama.cpp; AMX doubles FLOP/cycle vs AVX-512 for the GEMM
  shape that dominates expert load.
- **Gap it leaves:** requires Intel Sapphire Rapids (AMX); requires CUDA/ROCm; hardware
  minimum is 128 GB RAM — does not serve the 16–32 GB class at all. No resource contract
  layer: does not predict peak memory from on-device calibration, does not enforce a
  declared budget, and does not emit a degradation record on mode transitions.
- **What fitsproof does differently:** designed for the hardware class KTransformers
  explicitly excludes; adds the contract layer KTransformers omits.

---

### Category B — Prediction and sizing tools

#### 4. ridgepoint
- **PyPI:** https://pypi.org/project/ridgepoint/ (source: https://github.com/Isk4R1oT/ridgepoint)
- **Stars:** 1 (GitHub, 2026-09-27)
- **Latest release:** v0.1.2, uploaded 2026-09-08
- **Approach:** analytical VRAM sizing with calibrated constants; KV cache formula
  correct for both GQA and MLA attention types to the byte; roofline intervals (not
  point estimates); each output field carries a `calibrated` flag indicating whether the
  value came from a calibrated measurement or a formula. Calibrated to ~1% against real
  vLLM on A100/H100.
- **What it does well:** the most accurate open prediction tool for the A100/H100 class;
  MLA support (DeepSeek-style); per-field provenance (`calibrated` flag is a genuinely
  useful design decision); intervals rather than point predictions.
- **Gap it leaves:** calibration is offline, against A100/H100, not on the user's machine.
  Prediction only — no budget enforcement, no degradation path, no RSS proof harness, no
  stress harness. Does not serve the 4–8 GB VRAM class.
- **What fitsproof does differently:** calibration is on-device (probe → calibrate →
  MAPE on held-out configs from this machine); prediction is followed by enforcement (admit)
  and proof (verify against measured RSS). The pipeline is: predict → enforce → prove.

#### 5. pochenai/llm-inference-calculator
- **Repo:** https://github.com/pochenai/llm-inference-calculator
- **Stars:** 20 (2026-09-27)
- **Last push:** 2026-09-09
- **Approach:** two-phase roofline static performance model: prefill is compute-bound
  (TTFT estimate), decode is bandwidth-bound (TPOT estimate), MoE expert coverage, layout
  solver for multi-GPU configurations, speculative decoding throughput modelling.
- **What it does well:** more rigorous than a simple `bytes / bandwidth` formula; models
  MoE sparsity and speculative decoding's batch-size effect; instantaneous (no runtime
  measurement needed).
- **Gap it leaves:** static model, not calibrated to any real machine. No enforcement.
  No RSS measurement. Targets multi-GPU data-centre configurations, not consumer hardware.
- **What fitsproof does differently:** calibrates constants from measurements on the
  user's actual hardware; enforces the resulting prediction as a contract.

#### 6. Pluenet-Killian/llm-roofline
- **Repo:** https://github.com/Pluenet-Killian/llm-roofline
- **Stars:** 0 (2026-09-27)
- **Last push:** 2026-06-20
- **Approach:** decode throughput floor = bytes_per_token ÷ bandwidth, per GPU; roofline
  chart generation.
- **What it does well:** clean single-formula derivation; produces a readable chart.
- **Gap it leaves:** derives only a lower-bound throughput floor; no peak memory
  prediction, no enforcement, no calibration from measurement, inactive since June 2026.
- **What fitsproof does differently:** adds the memory prediction and enforcement layers;
  calibrates the bandwidth constant from real measurement rather than using a spec-sheet value.

#### 7. JohnScheuer/hardware-aware-llm-runtime
- **Repo:** https://github.com/JohnScheuer/hardware-aware-llm-runtime
- **Stars:** 0 (2026-09-27)
- **Last push:** 2026-06-25
- **Approach:** hardware-calibrated roofline with empirical fitting and analytical optimal
  batch size; predicts batch sweet spot within ~1.
- **What it does well:** fits roofline constants from real measurements (not spec sheets),
  finds compute/bandwidth crossover empirically, accounts for batch-size effects on
  throughput.
- **Gap it leaves:** focused on throughput optimisation (finding the optimal batch), not
  memory budget enforcement. No degradation path. No stress harness. Inactive since June 2026.
- **What fitsproof does differently:** priority is memory safety (enforced budget, measured
  proof), not throughput optimisation. The calibration approach is similar in spirit but
  fitsproof drives it through an enforcement gate.

#### 8. Shun-Calvin/llm-vram-calculator
- **Repo:** https://github.com/Shun-Calvin/llm-vram-calculator
- **Stars:** 1 (2026-09-27)
- **Last push:** 2026-09-26
- **Approach:** VRAM/TTFT/tok/s across 100+ models × 70+ GPUs with a public API.
- **What it does well:** widest model×GPU coverage of any tool in this list; useful for
  rough hardware selection before purchase.
- **Gap it leaves:** formula-based (not calibrated to any machine); GPU-only; no
  enforcement, no degradation, no stress harness.
- **What fitsproof does differently:** calibrates to the user's machine, enforces the
  prediction as a contract, and proves compliance.

---

### Category C — Determinism checking

#### 9. detllm
- **Repo:** https://github.com/tommasocerruti/detllm
- **Stars:** 20 (2026-09-27)
- **Last push:** 2026-08-20
- **Approach:** capability-gated determinism tiers: Tier 0 (artifact reproducibility),
  Tier 1 (run-to-run output repeatability at fixed seed/batch), Tier 2 (Tier 1 + logprob
  equality); repro packs. Always reports the tier actually achieved.
- **What it does well:** clean framing of determinism as a tiered capability rather than
  a binary; the `calibrated` vs `not-calibrated` style of honest reporting.
- **Gap it leaves:** determinism reporting only — does not predict or enforce memory
  budgets.
- **What fitsproof does differently:** adopts the detllm tier model for the verify layer
  (source 14 in the sources section); adds the contract enforcement on top.

---

### Comparison Table

Verified star counts and release dates as of 2026-09-27T13:00 UTC (GitHub REST API).
"Last activity" is the most recent push date from the API.

| Tool | Stars | Last activity | Approach | What it does well | Gap it leaves | What fitsproof does differently |
|---|---|---|---|---|---|---|
| **llama.cpp** (ggml-org/llama.cpp) | 129,661 | 2026-09-27 (v0.5.0) | CPU/GPU inference, GGUF, k-quants | Mature, broad model support, fast CPU kernels, layer offload | Silent OOM; silent CPU fallback; no user budget; no calibrated prediction | Explicit budget declaration; loud degradation record; stress harness proving compliance |
| **vLLM** (vllm-project/vllm) | 92,766 | 2026-09-27 (v0.30.0) | GPU serving, PagedAttention, continuous batching | Highest GPU throughput, production serving, 100+ models | Requires A100/H100 class; does not serve 4–8 GB VRAM; non-deterministic by default | CPU-first; 4–8 GB VRAM class; calibrated per-machine; deterministic by construction |
| **KTransformers** (kvcache-ai/ktransformers) | 19,538 | 2026-09-23 (v0.7.1) | CPU/GPU hybrid MoE, AMX kernels | 671B on 14 GB VRAM; 1.25–4.09× decode over llama.cpp | Requires 128 GB RAM + AMX; no resource contract layer | Targets 16–32 GB RAM class; adds predict→enforce→prove pipeline |
| **ridgepoint** (Isk4R1oT/ridgepoint, PyPI) | 1 | 2026-09-08 (v0.1.2) | Calibrated VRAM + roofline for A100/H100 | ~1% MAPE vs real vLLM; MLA-correct; per-field calibrated flag | Prediction only; offline calibration; no enforcement; no RSS harness | On-device calibration; enforcement gate; measured RSS proof |
| **llm-inference-calculator** (pochenai) | 20 | 2026-09-09 | Two-phase roofline (prefill compute / decode bandwidth); MoE + spec-decoding | Rigorous two-phase model; MoE sparsity; spec-decoding throughput | No calibration; no enforcement; targets data-centre multi-GPU | On-device calibration; single-machine consumer target |
| **llm-roofline** (Pluenet-Killian) | 0 | 2026-06-20 | Decode throughput floor = bytes/bandwidth per GPU | Simple clean derivation | Throughput floor only; no memory prediction; no enforcement; inactive | Memory contract + enforcement + RSS proof |
| **hardware-aware-llm-runtime** (JohnScheuer) | 0 | 2026-06-25 | Hardware-calibrated roofline, empirical optimal batch | Empirical constant fitting; finds compute/bandwidth crossover | Throughput focus; no enforcement; no stress harness; inactive | Memory-safety focus; enforcement gate after calibration |
| **llm-vram-calculator** (Shun-Calvin) | 1 | 2026-09-26 | Formula-based VRAM/tok/s for 100+ models × 70+ GPUs | Widest model×GPU coverage | Formula-based, not calibrated; GPU-only; no enforcement | On-device calibration; enforcement; proof harness |
| **detllm** (tommasocerruti) | 20 | 2026-08-20 | Capability-gated determinism tier reporting | Honest tier framing; repro packs | Determinism only; no memory prediction or enforcement | Adopts detllm tier model; adds contract enforcement on top |

---

### The Gap We Are Claiming

**Stated precisely:** no single tool in this list does all of the following on one machine:

1. Calibrate prediction constants from measurements taken *on the user's actual hardware*
   (not offline against A100/H100).
2. Enforce a user-declared memory budget: admit, degrade, or refuse — loudly, never silently.
3. Prove compliance: a stress harness asserts measured peak RSS never exceeded the declared
   budget, and reports the margin.

Each ingredient exists somewhere. ridgepoint calibrates (against other hardware). llama.cpp
runs inference. detllm checks determinism. No single tool assembles the three-step pipeline:
**probe → calibrate → plan → admit (enforce) → verify (prove)**.

**How a user would notice the gap:** they set `--budget 4G` and run a 7B model. With
llama.cpp, they see either a successful run or a CUDA OOM (no warning, no managed
degradation, no margin report). With fitsproof, they see either `ADMITTED: 3.8 GB predicted
≤ 4.0 GB budget (margin: 200 MB)` followed by a verified measured peak, or `REFUSED:
needs 4.300 GB, budget 4.000 GB; no listed option fits — nearest is "Use int4_sym
quantisation instead of none" at 3.900 GB (0.100 GB above budget)` (wording since
F-2 fix, improve pass c1-p09-improve-2). The contract is explicit
and enforced; the outcome is documented before generation starts.

**The target hardware class is specific:** 4–8 GB VRAM, 16–32 GB RAM. Every mature tool
in the table (llama.cpp, vLLM, KTransformers) was designed for hardware above this range
or does not address it at all. Strata (one-click consumer packaging, not open source)
requires 12 GB+ VRAM. This hardware class is large — it covers most developer-grade
workstations purchased before 2023 and the majority of non-gaming laptops with discrete GPUs.

---

### Falsification for Pass 2

1. **If ridgepoint's A100/H100 calibration transfers to a Quadro M2000 with <5% MAPE:**
   on-device calibration adds no value. This is empirically testable (run both; compare to
   measured RSS). We do not know the answer; this is the honest uncertainty. The stress
   harness in acceptance criterion 7 will surface a budget violation if ridgepoint's
   transfer prediction is wrong and fitsproof's is right.

2. **If llama.cpp ships explicit budget enforcement before this repo is published:** the
   gap closes. The llama.cpp changelog should be checked at publication time. As of
   2026-09-27 (v0.5.0), the `--n-gpu-layers` flag controls offload but there is no
   `--budget` flag or degradation record in the CLI or API.

3. **If the 4–8 GB VRAM / 16–32 GB RAM hardware class is smaller than claimed:** the
   total addressable audience shrinks. The claim rests on market-share data for consumer
   and workstation GPUs sold 2018–2022; this was not independently verified in pass 2.

4. **If the "no single tool does all three" claim is wrong:** there is a tool in this space
   that was not found in the scan. The scan covered the 9 tools named in MARKET-VERDICTS.md
   plus a targeted search for "LLM memory budget enforcement" on GitHub; no additional
   tools appeared. The adversarial reviewer should repeat the search with different query
   terms.

---

## Pass 3 — Real-World Applicability (cycle 1, pass 3)

Deliverables of this pass: **`docs/ADOPTION.md`** (Tuesday adoption against
ollama 0.20.3, executed on this machine) and **`scripts/ollama_gate.py`**
(the gate the recipe chains, committed at `367fa45`). Everything below is
raw output from 2026-09-27.

### Open questions from passes 1–2 — closure table

Status legend: CLOSED (question answered), OBSERVED (the falsifier fired;
recorded as a defect, not hidden), OPEN (cannot be closed in this pass;
closure procedure named).

| # | Question (source) | Status | Evidence |
|---|---|---|---|
| 1 | Does measured bandwidth break the >2× throughput assumption? (P1-F1) | **OBSERVED** | Calibration fits `bandwidth_utilisation=0.0345` (vs the 0.6 default), held-out MAPE 50.3%, tok/s over-predicted ~2× at reference scale (EVIDENCE.md §6). Root cause named there: NumPy per-token dispatch not in the roofline. Falsifier fired; published as required. Feeds the improve pass. |
| 2 | Does KV cache dominate weight streaming at moderate context? (P1-F2) | **CLOSED (not observed)** | First-principles: KV = 2·L·kv_h·head_dim·elem_bytes per token (cost.py `kv_cache_bytes`, same formula as ridgepoint's documented GQA form). Llama-7B fp16 non-GQA: 2·32·32·128·2 = 524,288 B/tok → 4096 ctx = 2.15 GB vs 14 GB weights; crossover ≈ 27k tokens. With GQA kv_h=8: crossover ≈ 107k tokens. Gate default ctx 4096 is an order of magnitude below either. The term is inside `estimate()` (`total_peak = weight + kv + activation`), so an omission fault is covered by tests/contract/test_cost.py. |
| 3 | Does RSS sampling miss the real peak? (P1-F3) | **CLOSED as documented limitation** | Confirmed empirically this campaign: peak_MB is 347.0 flat across all 9 Pareto configs (EVIDENCE.md §10) — the interpreter floor masks the 38 MB fixture. No violation observed (item 7), but the instrument has a known floor; README Limitations carries it. |
| 4 | Speculative equality outside greedy? (P1-F4) | **CLOSED by scope** | v0.1 claims and tests greedy equality only (acceptance criterion 4). Rejection sampling for temp>0 is not implemented and not claimed. No open risk. |
| 5 | Calibration MAPE >20% on held-out configs? (P1-F5) | **OBSERVED** | 50.3% (n_held_out=1, EVIDENCE.md §6). The falsifier fired exactly as written; spec says publish it, so it is published. Root cause + fix path in ADOPTION.md F-1 and the improve pass. |
| 6 | Prediction-interval coverage <80%? (P1-F6) | **OPEN (procedure defined)** | Not measurable at n_held_out=1 (CI degenerates to [50.3%, 50.3%]). Closure: run `calibrate.collect_measurements` until n_held_out ≥ 10, then check coverage. Until then any interval claim must be labelled unvalidated — it is, in this document. |
| 7 | Admitted config exceeding its budget in measurement? (P1-F7) | **CLOSED (not observed)** | 25-config stress harness: zero violations (EVIDENCE.md §8). Real-model cross-check this pass: predicted peak 7.219 GB vs ollama-observed 4.4 GB for gemma3:4b — conservative by +64%, never above. |
| 8 | Tier-1 determinism violated in the NumPy path? (P1-F8) | **CLOSED (not observed)** | Seed-reproducibility asserted in tests/engine/test_sampling.py, test_attention.py, test_speculative.py; full suite 88 passed (re-run this pass, `367fa45` parent). |
| 9 | Does ridgepoint's calibration transfer to a Quadro M2000 (<5% MAPE)? (P1-F9 / P2-F1) | **CLOSED by capability absence** | ridgepoint cannot express this hardware: `ridgepoint fit Qwen/Qwen2.5-7B-Instruct --gpu quadro-m2000` → `ridgepoint: unknown gpu: quadro-m2000`. Run *without* `--gpu` on this CPU box it silently predicts for `1× a100-80gb · Ampere` (raw header captured) — it answers a question about different silicon. A head-to-head MAPE on our class is therefore not merely untested but **unrepresentable**; on-device calibration stands uncontested for 4–8 GB / CPU-first machines. (Also observed: gated HF repos 403 without auth — another adoption friction point for ridgepoint, not ours.) |
| 10 | Has llama.cpp shipped budget enforcement since v0.5.0? (P2-F2) | **CLOSED same-day by pass 2** | Pass 2 verified (2026-09-27T13:00 UTC): v0.5.0 has `--n-gpu-layers` offload control, no `--budget` flag, no degradation record. No newer release exists as of this pass. Re-check at publication time. |
| 11 | Is the 4–8 GB VRAM / 16–32 GB RAM class large enough? (P2-F3) | **CLOSED (claim holds)** | Steam Hardware Survey (Aug 2026): 8 GB VRAM = 25.32–25.74%, 16 GB = 25.90%; Tom's Hardware / TechRadar on the July data: *"just under half (47%) have a graphics card with 8GB or less"*. Plus the CPU-only class this box represents (`VRAM: 0.00 GB` from `fitsproof probe`). The addressable class is ~half of Steam, before counting iGPU/CPU-only hosts. |
| 12 | Is "no single tool does all three" still true? (P2-F4) | **OBSERVED — claim NARROWED** | Re-ran the search with different terms (raw outputs below) and found **Grevix/aura** (4 stars, Rust, MIT/Apache-2.0, pushed 2026-09-03): *"hardware-aware memory-budget enforcement and inference orchestration for local LLMs"*, with pre-execution feasibility modelling, **cgroup v2 / Win32 Job Object enforcement**, a context-ladder degradation (4096→1024), and ollama model discovery. aura covers limb (2) of the claim — enforcement — more aggressively (kernel-level) than fitsproof does today. See "Narrowed claim" below. |

Raw search output for item 12 (GitHub REST, 2026-09-27):

```
$ curl -s "https://api.github.com/search/repositories?q=llm+memory+budget+enforcement&per_page=8"
total 10
Emmimal/context-engine 197  A pure-Python context management layer for LLM systems...
Grevix/aura 4  Hardware-aware memory-budget enforcement & LLM inference orchestration for low-R...
ashcakeancient7671/aura 1  Run low-memory LLMs on consumer hardware with adaptive memory-budget enforcement
...
$ curl -s ".../search/repositories?q=%22gpu+memory%22+guard+llm+inference+oom&per_page=8"
total 0
$ curl -s ".../search/repositories?q=ollama+memory+limit+guard&per_page=8"
total 0
```

### Narrowed claim (replaces pass 2's wording)

Pass 2 claimed: no single tool does (1) on-device calibration,
(2) budget enforcement, (3) proof harness. **aura does (2).** What no
single tool — including aura — documents, as of 2026-09-27:

1. **Calibration fitted on the user's own machine with a held-out split,
   MAPE and bootstrap interval published.** aura's README reports measured
   benchmarks (70-prompt suite, `Provenance: aura_measured`) but documents
   no on-device calibration protocol with train/hold-out evaluation;
   ridgepoint's calibration is offline, for A100/H100, and has no SKU for
   this class (item 9).
2. **A proof harness that asserts measured ≤ declared over ≥20
   configurations with a margin distribution, as a repo test.** aura's own
   README example reports a run that would fail such a harness, verbatim:
   `Memory Budget  : 4.00 GB (Win32 Job Object Enforced)` /
   `Peak RSS       : 4.92 GB`. Whether that is a measurement of a
   different pool or a violation, a fitsproof-style harness is exactly
   what would catch it — that asymmetry *is* the product.
3. **Admission as an embeddable API** (`plan`/`admit` objects a service
   can call before allocating) rather than only a CLI runner.

COMPARISONS.md should gain an aura row (4 stars, 2026-09-03); not edited
here because a parallel lane owns that artifact. aura is 4 stars and
unverified — but if it ships calibration plus a zero-violation stress
harness, this gap closes. Watch it at publication time.

### Findings raised this pass (all in ADOPTION.md with raw output)

| id | severity | finding | status |
|---|---|---|---|
| F-1 | major | Real-model peak over-predicted +64% (7.219 GB vs 4.4 GB observed): head_dim assumption (hidden/heads=320 vs gemma3's actual 256) and fp32 embed+unembed accounting (5.37 of 7.22 GB) | open — fix in cost model; observed 4.4 GB becomes the KAT |
| F-2 | major | Refusal message names a non-fitting config as "nearest fitting" (plan.py DOES_NOT_FIT branch appends `degradations[-1]` unconditionally; at 3 GB it names an offload option that predicts 3.699 GB) | **fixed** — improve pass c1-p09-improve-2: `plan.no_fit_reason()` names the smallest-predicted-peak option and states the gap above budget ("no listed option fits — nearest is ..."); test `test_refusal_never_names_a_non_fitting_config` pins all three faults |
| F-3 | minor | Probe bandwidth drifted 1.8× day-over-day (7.18 → 3.94–4.22 GB/s); run-to-run spread same day ~7% | open — operational rule documented (re-probe on load-profile change) |
| F-4 | minor | CPU-roofline tok/s is meaningless for hybrid CPU/GPU engines (measured 12.38 tok/s vs sub-1 prediction for gemma3:4b at 44/56% CPU/GPU) | documented — never publish tok/s for a GPU-backed engine from the CPU profile |
| F-5 | minor | `probe` reports `VRAM: 0.00 GB` on a box with a Quadro M2000 — GPU memory unmeasured in v0.1 | documented — v0.1 contract is RAM-only |

Design decision recorded: `scripts/ollama_gate.py` treats DEGRADED as
**refuse-by-default** (exit 2) when gating an external engine, because a
fitsproof degradation cannot be applied to ollama and chaining on a
degradation record would be a green light for a possible violation —
the exact silent-mode-change class this repo exists to eliminate.

### Falsification for Pass 3

Real-world applicability is wrong if any of these holds. Status now:

1. **A config the gate ADMITTED exceeds its budget in reality.**
   NOT OBSERVED: predicted 7.219 GB ≥ observed 4.4 GB (gemma3:4b);
   stress harness 0/25 violations at fixture scale. The conservative
   direction is load-bearing — if a future calibration flips the error
   to under-prediction, the core claim dies immediately.
2. **A config the gate REFUSED would have fit.** NOT OBSERVED across
   tested budgets {2, 3, 5, 6, 12} GB: refusals at 2–3 GB are correct
   (actual footprint 4.4 GB), degradations at 5–6 GB are conservative
   over-predictions, not refusals. The dangerous window — actual fit,
   predicted exceed — manifests as DEGRADED, which strict mode refuses
   to chain (ADOPTION §2c).
3. **Prediction error >2× on a real model.** Observed error 1.64×
   (over-predict) — inside 2×, but >20%: falsifier P1-F5 stands
   OBSERVED (item 5). The adoption-blocker analysis in ADOPTION §5
   names this as the single most likely reason a team would not adopt.
4. **Adoption requires reading our source.** The recipe in ADOPTION §2
   is copy-pasteable from a clone; the install is source-only, which is
   the v0.2 MANDATE M1 gap (binary + SHA256 release). OPEN by design,
   not by oversight.
5. **A competitor closes the narrowed claim.** aura (4★) is the one to
   watch; as of 2026-09-27 its README shows no held-out calibration
   protocol and publishes a run whose peak RSS exceeds its stated
   budget. Re-check its repo before publication.

---

## Cycle 2 — Pass 1 — GROUND TRUTH for the v0.2 MANDATE

This pass grounds the v0.2 MANDATE surfaces (specs/fitsproof.md § M1–M4): the
binary, the four plugin surfaces (OpenAI-compatible server, Python client, guard
decorator, MCP server), and M4's requirement that tests cite specific source IDs.
Sources 23–34 are new. Five (23, 25, 27, 28, 29) get the full treatment required
by QUALITY-CONTRACT §3: exact method, equations with notation explained,
assumptions, documented failure modes. All links re-checked with curl on
2026-09-27 (raw output below).

Note on one URL: `spec.modelcontextprotocol.io` fails TLS from this host
(`curl: (35) TLS connect error ... unexpected eof`, two attempts); the canonical
`https://modelcontextprotocol.io/specification/2025-03-26/` resolves (200) and is
what this document cites. The old hostname appears in two docstrings
(`src/fitsproof/mcp.py`, `tests/value/test_incumbent_gap.py`); both were updated
this pass to the resolving URL so the adversarial citation audit cannot flag them.

### 23. MCP Specification (protocol version 2025-03-26) — DEEP [drives M2.4]

**Model Context Protocol, official specification.**
https://modelcontextprotocol.io/specification/2025-03-26/
(Sections read: Base Protocol/Transports, Server Features/Tools.)

**Claim it supports:** the entire `fitsproof mcp` surface
(`src/fitsproof/mcp.py`, tests in `tests/value/test_incumbent_gap.py`): JSON-RPC
messages over the stdio transport, `initialize` / `tools/list` / `tools/call`,
and the refusal protocol (`isError: true` on a REFUSED admit).

**Exact method (transcribed from the spec):**
```
Transport (stdio): client launches the server as a subprocess; server reads
  JSON-RPC from stdin, writes JSON-RPC to stdout.
  "Messages are delimited by newlines, and MUST NOT contain embedded newlines."
  "The server MUST NOT write anything to its stdout that is not a valid MCP
   message." (logging goes to stderr)

Tool call result:
  { "jsonrpc": "2.0", "id": 2,
    "result": { "content": [ { "type": "text", "text": "..." } ],
                "isError": false } }

Error split (spec's own words):
  Protocol errors   -> standard JSON-RPC error objects (unknown tool,
                       invalid arguments, server errors; e.g. -32602)
  Tool execution errors -> reported IN the result with isError: true
                       (API failures, invalid input, business-logic errors)
```

**Assumptions the method requires:**
- Exactly one JSON message per line on stdout; any stray `print()` from library
  code corrupts the stream for the whole session.
- The client (agent) treats `isError` as the only in-band success/failure signal
  for a tool result; a non-protocol error cannot be conveyed any other way.

**Documented failure modes (per the spec itself):**
- stdout pollution: a server that logs to stdout violates the transport rule and
  breaks the client's framing — the reason `mcp.py` writes JSON only.
- Errors on notifications are undetectable: JSON-RPC notifications carry no
  `id`, so the caller "would not be aware of any errors" (JSON-RPC spec §4.1,
  source 24). fitsproof's tools are requests, never notifications, for this reason.
- If `admit` returned `isError: false` on REFUSED, an agent would read a
  refusal as success. This is the fault `test_incumbent_gap.py` pins.

### 24. JSON-RPC 2.0 Specification

**JSON-RPC Working Group.** https://www.jsonrpc.org/specification
(Updated 2013-01-04; the wire format MCP builds on.)

**Claim it supports:** the message envelope and error codes in `mcp.py`
(`"jsonrpc": "2.0"`, request/notification/response distinction, `-32601` /
`-32602` error codes for unknown tool / invalid params).

**Key rules extracted:**
```
Notification := Request object without an "id" member.
  "The Server MUST NOT reply to a Notification, including those that are
   within a batch request."
Error codes: -32601 Method not found, -32602 Invalid params,
             -32603 Internal error.
```

**Failure mode relevant to us:** notifications are unconfirmable by definition —
any fitsproof call that must be able to fail loudly (plan/admit) is issued as a
request with an `id`.

### 25. Server-Sent Events (WHATWG HTML Living Standard) — DEEP [drives M2.1]

**WHATWG HTML Standard, § "Server-sent events".**
https://html.spec.whatwg.org/multipage/server-sent-events.html

**Claim it supports:** `src/fitsproof/engine/server.py` streaming response
format and the fault `test_completion_streaming_returns_chunks` detects
("returning the full response as a single chunk is not real SSE streaming").

**Exact method (transcribed):**
```
MIME type: text/event-stream, always decoded as UTF-8 (no other charset).

ABNF (spec's grammar, notation: [] = optional, * = zero-or-more,
      / = alternation, (... grouping)):
  stream = [ bom ] * event
  event  = *( comment / field ) line-terminator
  field  = event-type / data / id / retry        ; each line "name[: value]"

Dispatch: events are dispatched on a blank line. "If the data buffer is an
  empty string, set the data buffer and the event type buffer to the empty
  string and return." (an empty data buffer discards the event)
EOF: "If the file ends in the middle of an event, before the final empty
  line, the incomplete event is not dispatched."
```

**Assumptions:**
- Line terminator is CR, LF, or CRLF; field names are compared literally (no
  case folding).
- The server flushes each event promptly. The spec documents the risk
  explicitly: "block buffering or line buffering with different expected line
  endings can cause delays in event dispatch" — line buffering with LF is safe.

**Documented failure modes:**
- Buffered-then-flushed-at-end responses satisfy the byte format but are not
  streaming; the spec's own buffering paragraph is why the test asserts
  *multiple* chunks arrive (counted), not just a `data:` prefix.
- An event without a terminating blank line is silently dropped at EOF — the
  server must terminate the final chunk and send the blank line, or the last
  token never reaches the client.

### 26. OpenAI Chat Completions API (official OpenAPI description)

**OpenAI, openai-openapi repository (openapi.yaml).**
https://github.com/openai/openai-openapi

**Claim it supports:** the non-streaming and streaming response shapes served
by `server.py` (`choices[0].message.content`; streaming chunks with
`object: "chat.completion.chunk"`, `choices[0].delta`, `finish_reason:
"stop"`, terminated by `data: [DONE]`). Verified by grep against the canonical
file this pass (raw output below).

**Failure mode:** an SDK client validates `object`/`delta`/`finish_reason`;
omitting the `data: [DONE]` sentinel or emitting `finish_reason` on every chunk
breaks streaming clients that wait for the terminator or for `stop`.

### 27. MAPE — Mean Absolute Percentage Error — DEEP [drives calibrate.py, M4 KAT]

**Hyndman, R. J., Koehler, A. B. (2006).** Another look at measures of
forecast accuracy. *International Journal of Forecasting*, 22(4), 679–688.
DOI: 10.1016/j.ijforecast.2006.03.001
https://doi.org/10.1016/j.ijforecast.2006.03.001
(Abstract verified via RePEc this pass: https://ideas.repec.org/a/eee/intfor/v22y2006i4p679-688.html)

**Claim it supports:** `_mape()` in `src/fitsproof/contract/calibrate.py` —
the headline calibration metric (`mape_held_out`), and falsifiers P1-F5/P1-F6.

**Exact equation (notation explained):**
```
MAPE = (100 / n) * SUM_{i=1..n} ( |y_i - ŷ_i| / |y_i| )

  y_i    = actual value   (measured tok/s on held-out config i)
  ŷ_i    = predicted value (decode_tok_s from the cost model, config i)
  n      = number of held-out configurations
  returns a percentage: 50.3 means 50.3% mean absolute relative error
```
Division is by the ACTUAL, never by the prediction (a swap inverts the error
direction — the fault the KAT in `calibrate` tests detects).

**Assumptions:**
- `y_i != 0` for all i. Our implementation masks zero actuals
  (`nonzero = actual != 0`) and returns `NaN` only if ALL are zero.
- Relative error is meaningful to average across configs of different scale —
  true here because every config is measured in the same unit (tok/s) on one
  machine, and we deliberately do not mix machines in one calibration.

**Documented failure modes (per the literature):**
- Degeneracy at zero: the paper's own finding is that many accuracy measures
  "are degenerate in commonly occurring situations" (abstract, verified) —
  percentage error is undefined/infinite when the actual is zero. Dropping zero
  actuals (our mask) biases the reported MAPE *downward* if zero-measurement
  configs are the hardest ones; this must be re-checked when
  `collect_measurements` grows (closure procedure for P1-F6).
- Asymmetry: over-prediction is penalised without bound (ŷ → ∞ ⇒ error → ∞),
  under-prediction is capped at 100% (ŷ = 0 ⇒ |y−0|/|y| = 1). For a budget gate
  this asymmetry is *desirable* — over-promising throughput is the worse error —
  but it means MAPE is not a symmetric confidence measure.
- Our measured 50.3% MAPE (n_held_out=1, EVIDENCE.md §6) is the published,
  honest number; the falsifier fired and is recorded, not hidden.

### 28. Bootstrap percentile confidence interval — DEEP [drives calibrate.py CI]

**Efron, B. (1979).** Bootstrap Methods: Another Look at the Jackknife.
*The Annals of Statistics*, 7(1), 1–26. DOI: 10.1214/aos/1176344552
https://doi.org/10.1214/aos/1176344552
(Full text resolves: projecteuclid.org, 200.)

**Claim it supports:** `_bootstrap_mape_ci()` in
`src/fitsproof/contract/calibrate.py` — the `ci_lower`/`ci_upper` fields of
`CalibrationResult` and falsifier P1-F6 ("prediction interval coverage below
80%").

**Exact method (notation explained):**
```
F̂_n = empirical distribution of the held-out pairs:
      F̂_n = (1/n) * SUM_{i=1..n} δ_{x_i}
      (δ_x = point mass at x; x_i = (ŷ_i, y_i) pair — pairs are resampled
       JOINTLY, preserving the predicted↔actual pairing)

For b = 1..B  (B = 1000 in _bootstrap_mape_ci, seed fixed at 42):
      draw x*_1, ..., x*_n iid from F̂_n  (with replacement)
      θ*_b = MAPE(ŷ*, y*)                  (the statistic T)

95% percentile interval:
      CI = [ Q_{0.025}(θ*), Q_{0.975}(θ*) ]
      (Q_p = p-quantile of the B bootstrap replicates; alpha = 0.05)
```

**Assumptions:**
- Held-out configs are i.i.d. draws from the config distribution we care
  about. With a fixed 67/33 split of a small config list this is an
  approximation — configs are permutations of one list, not a random sample
  from a population.
- n is large enough that F̂_n approximates F. For n ≥ 2 the code resamples; for
  `n < 2` it returns the degenerate interval `[m, m]` — matching the method's
  own breakdown point, not hiding it.

**Documented failure modes:**
- Small-n percentile intervals undercover: with n = 1 (our current state,
  P1-F6 OPEN) the interval collapses to a point and carries no coverage
  guarantee; the closure procedure (n_held_out ≥ 10, then check coverage) is
  already written in the pass-3 closure table.
- Resampling (ŷ, y) independently instead of in pairs would destroy the
  prediction↔actual relationship and produce a nonsense interval; the
  implementation draws one index vector per replicate — the correct pairing.
- Quantile resolution: with B = 1000 replicates the 2.5% tail is estimated
  from ~25 samples; adequate for a reported interval, not for inference.

### 29. getrusage(2) / ru_maxrss — peak-RSS semantics — DEEP [drives verify.py, M3(d)]

**Linux man-pages project.** getrusage(2).
https://man7.org/linux/man-pages/man2/getrusage.2.html

**Claim it supports:** `_get_rss_bytes()` in
`src/fitsproof/contract/verify.py` and `contract/pareto.py` — the instrument
behind `measured_peak_bytes`, the stress harness (M3(d)), and README
Limitations ("RSS is the high-water mark since process start").

**Exact method (transcribed from the man page):**
```
ru_maxrss (since Linux 2.6.32) = "the maximum resident set size used (in KiB)"
peak_bytes = ru_maxrss * 1024        # Linux   (our code, verify.py:76)
peak_bytes = ru_maxrss               # macOS   (bytes — platform branch)
RUSAGE_SELF = statistics for the calling process,
              "the sum of resources used by all threads in the process"
```

**Assumptions:**
- RSS is a faithful proxy for the budget-relevant footprint. It includes
  shared-library pages mapped into the process and excludes kernel-side
  allocations on our behalf — a budget expressed in RSS is a budget on the
  process image, not on total system memory pressure.
- The harness samples after generation completes, so the reported peak is the
  process-lifetime high-water mark, not a per-call window.

**Documented failure modes:**
- No reset: the man page defines ru_maxrss as a maximum over the process
  lifetime with no reset operation. In a long-running `fitsproof serve`
  process, a peak from request N is permanently attributed to request N+1 —
  measured ≥ budget can be a *stale* peak, never a missed one (error direction
  is conservative: it can only over-report). This is the instrument-floor
  finding P1-F3 (347 MB flat across all stress configs, EVIDENCE.md §10).
- Unit skew: Linux KiB vs macOS bytes — the platform branch is load-bearing;
  dropping the `* 1024` under-reports by 1024× on Linux. Unit KAT lives in
  `tests/contract/test_plan_admit_verify.py`.

### 30. Linux cgroup v2 memory controller [grounds the aura comparison]

**Linux kernel documentation.** Control Group v2 — memory controller.
https://docs.kernel.org/admin-guide/cgroup-v2.html

**Claim it supports:** (a) why fitsproof's in-process RSS gate is not the same
instrument as OS-level enforcement, (b) the aura row in COMPARISONS.md.

**Semantics extracted:**
```
memory.max   = hard limit; "If a cgroup's memory usage reaches this limit and
               can't be reduced, the OOM killer is invoked in the cgroup...
               the usage may go over the limit temporarily."
memory.peak  = max usage since cgroup creation or last reset (resettable by
               writing to the file) — a per-cgroup high-water mark.
Accounted:    userland memory (page cache + anonymous), kernel data structures
              (dentries, inodes), TCP socket buffers.
```

**Failure mode relevant to positioning:** cgroup accounting ≠ process RSS (it
adds page cache and kernel structures), so an aura-style enforcement can show
`peak > budget` measured in one unit while the process's own RSS stayed under
in the other — precisely the ambiguity in aura's published 4.92 GB peak vs
4.00 GB budget example. The units of a budget must be declared with it;
fitsproof's budget is explicitly RSS bytes.

### 31. PyInstaller — one-file mode [drives M1]

**PyInstaller documentation, "What To Generate".**
https://pyinstaller.readthedocs.io/en/stable/usage.html

**Claim it supports:** the M1 delivery surface (single standalone executable,
no Python/venv required) in the release workflow
(`.github/workflows/release.yml`).

**Facts extracted:** `-F, --onefile` builds a one-file bundled executable;
onefile mode runs by unpacking to a temporary directory at startup (the docs
describe splash-screen behaviour as indicating "application activity and
progress during extraction to the temporary directory").

**Documented failure modes:** startup pays an extraction cost; hidden imports
must be declared or the bundle fails at runtime (why the CI clean-job smoke
test — `fitsproof probe` from the downloaded artifact — is the evidence, not
the build succeeding).

### 32. RFC 9110 — HTTP Semantics [drives M2.1]

**IETF RFC 9110.** https://www.rfc-editor.org/rfc/rfc9110.html

**Claim it supports:** status-code usage in `server.py` (200 / 400 invalid JSON
/ 404 unknown path, `application/json` responses). The contract refusal path
maps to 4xx-family semantics rather than a silent 200.

### 33. Nucleus (top-p) Sampling [closes a citation gap in the engine]

**Holtzman, A., Buys, J., Du, L., et al. (2019).** The Curious Case of Neural
Text Degeneration. arXiv:1904.09751. https://arxiv.org/abs/1904.09751

**Claim it supports:** `top_p_sample()` in `src/fitsproof/engine/sampling.py`
(the module docstring already cited this arXiv ID; it was missing from
RESEARCH.md's source list until this pass — the traceability gap M4 exists to
catch).

**Method:** sort token probabilities descending, take the smallest set whose
cumulative probability ≥ p, renormalise over that set, sample. Documented
failure modes: with p → 1 the nucleus degenerates to the full vocabulary (no
truncation); greedy/beam decoding itself produces repetitive text (the paper's
central finding — decoding objective alone changes output quality). Sampling is
stochastic: fitsproof seeds it explicitly (Tier-1 determinism claim, sources
17/19) and uses temperature=0 in all calibration measurements.

### 34. Mutation testing — survey [grounds the ≥70% kill gate]

**Jia, J., Harman, M. (2011).** An Analysis and Survey of the Development of
Mutation Testing. *IEEE Transactions on Software Engineering*, 37(9).
DOI: 10.1109/TSE.2010.62 (resolves: 202 via doi.org, metadata confirmed via
Crossref: title, TSE, 2011-09.)

**Claim it supports:** the QUALITY-CONTRACT §5.3 / ITERATION-PROTOCOL pass-12
gate (≥70% mutants killed) and the interpretation of
`reports/mutation-c1.json`.

**Method:** inject syntactic mutants (operator/boolean/return mutations) into
source; a mutant is *killed* when at least one test fails; score =
killed / total. **Documented failure mode:** equivalent mutants — mutants
behaving identically to the original — cannot be killed by any test and
depress the score; the survey's treatment is why every surviving mutant must be
either given a killing test or argued equivalent. Honest current state:
cycle 1's mutation run recorded `rc: 124` (1800 s timeout during mutant
generation, `killed: null`) — no score exists yet; the gate is not claimed met.

### Link verification — raw output (2026-09-27, curl over every URL in this file)

```
# link check 2026-09-27 (curl -sL -o /dev/null -w '%{http_code}' -A Mozilla/5.0 --max-time 25)
200 https://api.github.com/search/repositories?q=llm+memory+budget+enforcement&per_page=8
200 https://arxiv.org/abs/1706.03762
200 https://arxiv.org/abs/1904.09751
200 https://arxiv.org/abs/1910.07467
200 https://arxiv.org/abs/2001.08361
200 https://arxiv.org/abs/2002.05202
200 https://arxiv.org/abs/2104.09864
200 https://arxiv.org/abs/2210.17323
200 https://arxiv.org/abs/2211.17192
200 https://arxiv.org/abs/2303.06865
200 https://arxiv.org/abs/2305.13245
200 https://arxiv.org/abs/2306.00978
200 https://arxiv.org/abs/2306.15595
200 https://arxiv.org/abs/2309.06180
200 https://arxiv.org/abs/2312.12456
200 https://arxiv.org/abs/2506.09501
200 https://arxiv.org/abs/2601.17768
200 https://arxiv.org/abs/2606.00279
403 https://dl.acm.org/doi/10.1145/1498765.1498785
200 https://docs.kernel.org/admin-guide/cgroup-v2.html
200 https://doi.org/10.1016/j.ijforecast.2006.03.001
200 https://doi.org/10.1214/aos/1176344552
200 https://github.com/ggerganov/llama.cpp/pull/1684
200 https://github.com/ggml-org/llama.cpp
200 https://github.com/Isk4R1oT/ridgepoint
200 https://github.com/JohnScheuer/hardware-aware-llm-runtime
200 https://github.com/kvcache-ai/ktransformers
200 https://github.com/openai/openai-openapi
200 https://github.com/Pluenet-Killian/llm-roofline
200 https://github.com/pochenai/llm-inference-calculator
200 https://github.com/Shun-Calvin/llm-vram-calculator
200 https://github.com/tommasocerruti/detllm
200 https://github.com/vllm-project/vllm
200 https://html.spec.whatwg.org/multipage/server-sent-events.html
200 https://ideas.repec.org/a/eee/intfor/v22y2006i4p679-688.html
200 https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/
200 https://man7.org/linux/man-pages/man2/getrusage.2.html
200 https://modelcontextprotocol.io/specification/2025-03-26/
200 https://pyinstaller.readthedocs.io/en/stable/usage.html
200 https://pypi.org/project/ridgepoint/
200 https://pypi.org/project/ridgepoint/0.1.1/
200 https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
200 https://www.cs.virginia.edu/stream/ref.html
200 https://www.jsonrpc.org/specification
200 https://www.rfc-editor.org/rfc/rfc9110.html
# canonical-vs-old-host check:
200 https://modelcontextprotocol.io/specification/2025-03-26/
000 https://spec.modelcontextprotocol.io/specification/2025-03-26/ (error: )
# Crossref check for the 403 (dl.acm.org is bot-blocked to curl):
crossref-200 10.1145/1498765.1498785 | Roofline | Communications of the ACM | [2009, 4]
```

### Star-count refresh (raw GitHub REST output, 2026-09-27 ~18:45 UTC)

Cycle-1 pass 2 recorded counts at 13:00 UTC; re-fetched this pass for the
named tools (raw):

```
$ curl -s https://api.github.com/repos/<owner>/<repo>   # one call per row
ggml-org/llama.cpp |stars 129690 |push 2026-09-27T18:36:21Z |lic MIT
vllm-project/vllm |stars 92788 |push 2026-09-27T17:48:55Z |lic Apache-2.0
kvcache-ai/ktransformers |stars 19540 |push 2026-09-23T05:07:33Z |lic Apache-2.0
Isk4R1oT/ridgepoint |stars 1 |push 2026-09-08T18:40:09Z |lic MIT
pochenai/llm-inference-calculator |stars 20 |push 2026-09-09T15:58:07Z |lic None
Pluenet-Killian/llm-roofline |stars 0 |push 2026-06-20T19:26:33Z |lic MIT
JohnScheuer/hardware-aware-llm-runtime |stars 0 |push 2026-06-25T09:50:23Z |lic MIT
Shun-Calvin/llm-vram-calculator |stars 1 |push 2026-09-26T06:38:02Z |lic MIT
tommasocerruti/detllm |stars 20 |push 2026-08-20T21:07:45Z |lic Apache-2.0
Grevix/aura |stars 4 |push 2026-09-03T17:50:25Z |lic Apache-2.0
modelcontextprotocol/modelcontextprotocol |stars 9320 |push 2026-09-24T20:40:42Z
```
Deltas vs the 13:00 UTC table: llama.cpp +29, vLLM +22, KTransformers +2;
ranking and conclusions unchanged. Comparison-table deepening beyond this
refresh is pass 2's job per ITERATION-PROTOCOL.

### OpenAI spec shape check — raw output

```
$ curl -sL https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml | grep -n "chat.completion.chunk\|\[DONE\]"
2834: {"id":"chatcmpl-123","object":"chat.completion.chunk",... "choices":[{"index":0,"delta":{"role":"assistant","content":""},"logprobs":null,"finish_reason":null}]}
2840: {"id":"chatcmpl-123","object":"chat.completion.chunk",... "choices":[{"index":0,"delta":{},"logprobs":null,"finish_reason":"stop"}]}
23140: data: [DONE]
```

### Alternatives considered — additions for this pass

| Approach | Why rejected |
|---|---|
| Enforcing budgets via cgroup v2 from inside fitsproof | requires privileges the target class (user laptops) does not grant; cgroup accounting ≠ RSS (source 30), so the declared unit would differ from the enforced unit. In-process RSS gate + declared-degradation records keep one consistent unit. |
| Reporting a bootstrap CI from training configurations | the interval would describe fit residual, not generalisation; source 28's method is applied to held-out pairs only — fitting on held-out is the exact fault the calibrate tests name. |
| Streaming responses as HTTP chunked JSON lines instead of SSE | the OpenAI SDK (source 26) and EventSource consumers expect `data:` framing; a non-SSE format breaks the "swap `base_url` and it works unchanged" M2.1 promise. |
| Sampled RSS (poll in a thread) instead of ru_maxrss | polling can miss peaks between samples (unsound for a proof); ru_maxrss is kernel-maintained and can only over-report (source 29). Trade: stale peaks in long-lived processes, accepted and documented (P1-F3). |

### Falsification for Cycle 2 — Pass 1

What observation would prove this pass's ground truth (and with it the v0.2
MANDATE design) wrong:

1. **An MCP client fails to complete a `tools/call` round-trip against
   `fitsproof mcp` while following the 2025-03-26 spec** (framing error, missing
   `initialize`, malformed result envelope). Not observed: the in-repo
   round-trip test passes; but the in-repo client is *ours* — an independent
   client (e.g. an official SDK) run in pass 3 or the adversarial pass is the
   real test. If it fails, M2.4's "an agent can consult the contract" is false
   as specified and the server must be fixed, not the spec reinterpreted.
2. **A streaming client (openai SDK or raw EventSource) sees the whole response
   in one chunk, or never sees the final event.** Not observed: the multi-chunk
   test counts chunks; if a client under buffering sees one chunk, the spec's
   own buffering warning (source 25) was right about *our* server and M2.1's
   "real chunked SSE" claim is false.
3. **Held-out measurements containing zero tok/s become common enough that the
   MAPE zero-mask changes the reported number materially** (>5 points vs the
   unmasked definition). Not observed (no zero held-out actuals yet). Would
   mean the headline calibration metric is biased by its own mask → report both
   masked and unmasked, per source 27's degeneracy finding.
4. **Bootstrap interval coverage measured over ≥10 held-out configs falls
   below 80%.** Not yet observable (n_held_out = 1, P1-F6 OPEN). This falsifies
   the interval fields of every Plan — the contract would be publishing
   uncertainty numbers that do not cover.
5. **A config admitted by `verify` shows ru_maxrss attributable to an earlier
   request in the same process, and the *current* request actually exceeded its
   budget below the stale peak.** Structurally possible (source 29: no reset)
   and *undetectable by the current instrument* — if observed, the proof harness
   must move to per-call measurement (fresh process or cgroup memory.peak) or
   the M3(d) claim must be narrowed to "process-lifetime peak". This is the
   strongest open threat to the "proves it" limb, and it is named here rather
   than left for the reviewer to find.
6. **PyInstaller onefile artifact fails `fitsproof probe` in the clean CI job**
   on a machine without Python (hidden import or extraction failure). Would
   falsify M1 ("a wheel-only story does not satisfy this") — the evidence bar
   is the clean-job run, not a green build.

None of 1–6 has been observed as a pass yet; 4 and 5 are structurally OPEN by
design (need data / a better instrument), and both closure procedures are
stated above. The conflict with the product spec's "three things no single
existing tool does together" (aura covers the enforcement limb) is recorded in
`docs/EVIDENCE.md` per the positioning rule; MARKET-VERDICTS wording wins.

---

## Cycle 2 — Pass 2 — ECOSYSTEM AND COMPETITION DEEPENING

*Pass 2 of cycle 2. Star counts and release data retrieved via GitHub REST API at
2026-09-27T20:42Z (raw output below). This pass: (1) refreshes the comparison table
with new star counts for all named tools, (2) documents two new tools surfaced by
re-running the c2-p1 search queries, (3) deepens the aura entry with evidence from
aura's own BENCHMARK.md, and (4) states which items in the pass-3 narrowed-claim
table are now confirmed, still open, or newly resolved.*

### Raw star-count refresh (2026-09-27T20:42:43Z)

Retrieved in a single batch via `curl -s https://api.github.com/repos/<owner>/<repo>`:

```
ggml-org/llama.cpp          | stars=129701 | push=2026-09-27T20:07:41Z | license=MIT
vllm-project/vllm           | stars=92792  | push=2026-09-27T19:47:37Z | license=Apache-2.0
kvcache-ai/ktransformers    | stars=19541  | push=2026-09-23T05:07:33Z | license=Apache-2.0
Isk4R1oT/ridgepoint         | stars=1      | push=2026-09-08T18:40:09Z | license=MIT
pochenai/llm-inference-calculator | stars=20 | push=2026-09-09T15:58:07Z | license=None
Pluenet-Killian/llm-roofline      | stars=0  | push=2026-06-20T19:26:33Z | license=MIT
JohnScheuer/hardware-aware-llm-runtime | stars=0 | push=2026-06-25T09:50:23Z | license=MIT
Shun-Calvin/llm-vram-calculator   | stars=1  | push=2026-09-26T06:38:02Z | license=MIT
tommasocerruti/detllm       | stars=20     | push=2026-08-20T21:07:45Z | license=Apache-2.0
Grevix/aura                 | stars=4      | push=2026-09-03T17:50:25Z | license=Apache-2.0
```

Latest releases (confirmed via `/releases/latest`):

```
ggml-org/llama.cpp       | tag=v0.5.0  | published=2026-09-23T20:50:06Z
vllm-project/vllm        | tag=v0.30.0 | published=2026-09-22T05:20:54Z
kvcache-ai/ktransformers | tag=v0.7.1  | published=2026-09-15T10:17:55Z
```

Deltas vs the c2-p1 18:45 UTC snapshot: llama.cpp +11, vLLM +4, KTransformers +1;
ridgepoint/detllm/aura/inactive repos unchanged. Rankings and gap conclusions
unchanged. PyPI ridgepoint still at v0.1.2 (last upload 2026-09-08).

### New tools surfaced by repeat search (same queries as c2-p1)

Two new entries appeared in the `llm+memory+budget+enforcement` search results that
were not present at the c2-p1 snapshot. Both were evaluated and neither closes the
fitsproof gap.

#### ashcakeancient7671/aura (1 star, pushed 2026-09-27T19:50:02Z)

- **Not a fork** (GitHub API `fork: false`). An independent rewrite of the aura
  concept, Windows-focused, targeting a different audience (consumer laptops with 4 GB
  RAM). The README describes itself as "a smart helper program" and directs users to
  download a ZIP binary. No CI, no MAPE, no calibration protocol, no proof harness.
- **Conclusion:** does not enter the comparison table. Simpler tool, single platform,
  no research contribution.

#### acasavaraju/AIOS (1 star, pushed 2026-03-29T22:54:39Z, Apache-2.0)

- **Description:** CPU-native LLM inference architecture with a "Model Contract spec",
  weight aliasing, sparsity maps, KV cache tiering, activation chunking.
- **Critical self-disclosure (README):** *"AIOS is a published framework and open
  specification. It is not yet a working implementation."* The runtime C ABI
  (`aios.h`) is "specified — not implemented". Profiler is "stubbed — not
  implemented". All performance numbers are analytical projections `[A]` or from
  prior work `[P]`. *"No inference has run under AIOS."*
- **Also surfaced by:** `llm+inference+contract+enforcement` search.
- **Conclusion:** not a competitor today (no working runtime). Does not enter the
  comparison table. Provides no evidence against the gap claim.

### Deepened aura entry — BENCHMARK.md evidence

c2-p1 noted that aura's README shows `Peak RSS: 4.92 GB` against a `4.00 GB
(Win32 Job Object Enforced)` budget, and flagged this as ambiguous (different
accounting units). This pass retrieved aura's `BENCHMARK.md` directly:

```
aura BENCHMARK.md (fetched 2026-09-27T20:42Z):

| Model            | Runtime                | Hardware         | Budget      | Pass Rate | Peak Working Set | Provenance      |
|qwen3:8b          | AURA llama-server      | Win32 Job Object | 4.00 GB     | 100% (1/1)| 4.92 GB          | aura_measured   |
|nous-hermes2:11B  | AURA llama-server      | Win32 Job Object | 8.00 GB     | 100% (1/1)| 7.41 GB          | aura_measured   |
```

The `4.92 GB` peak is in the same column as the 4.00 GB budget and labelled
`aura_measured` with `Simulated: false`. Win32 Job Object limits Working Set, not
page file usage; the Working Set is the resident physical memory of the process group.
The observed Working Set (4.92 GB) is 23% above the declared budget (4.00 GB).

aura's README does not contain the words "calibration", "held-out", "MAPE",
"prediction interval", "stress harness", or "violation". The 70-prompt benchmark
suite is an inference quality check (pass/fail per prompt), not a budget compliance
assertion. The audit tool (`aura audit`) checks 10-tier quality gates; based on the
README there is no gate that asserts `measured_peak ≤ declared_budget`.

**What this means for the gap claim:** aura's own published data, measured under its
own enforcement, shows peak Working Set exceeding declared budget by 0.92 GB. The
aura README does not report this as a violation and provides no mechanism to detect
or fail-fast on it. A fitsproof-style stress harness (which asserts
`measured_peak <= declared_budget` and exits non-zero on any violation) would flag
this run. That asymmetry — detect-and-fail vs run-and-report-post-hoc — is the
operationally relevant difference for a CI gate.

### Updated comparison table — all tools, counts as of 2026-09-27T20:42Z

| Tool | Stars | Latest release / Last push | Approach | What it does well | Gap it leaves | What fitsproof does differently |
|---|---|---|---|---|---|---|
| **llama.cpp** (ggml-org/llama.cpp) | 129,701 | v0.5.0 (2026-09-23) | CPU/GPU inference, GGUF, k-quants | Mature (3+ yr), broad model + quant support, fast CPU kernels, layer offload | Silent OOM; silent CPU fallback; no user-declared budget; no calibrated prediction; no proof harness | Explicit budget; loud degradation record; zero-violation stress harness |
| **vLLM** (vllm-project/vllm) | 92,792 | v0.30.0 (2026-09-22) | GPU serving, PagedAttention, continuous batching | Highest GPU throughput, 100+ models, production serving | Targets A100/H100; does not serve 4–8 GB VRAM class; non-deterministic by default | CPU-first; 4–8 GB class; per-machine calibration; deterministic by construction |
| **KTransformers** (kvcache-ai/ktransformers) | 19,541 | v0.7.1 (2026-09-15) | CPU/GPU hybrid MoE, Intel AMX kernels | Runs 671B on ~14 GB VRAM; 1.25–4.09× over llama.cpp; SOSP 2025 | Requires 128 GB RAM + AMX + CUDA/ROCm; no resource contract layer; no proof harness | Targets 16–32 GB RAM class; adds predict→enforce→prove pipeline |
| **ridgepoint** (Isk4R1oT/ridgepoint, PyPI v0.1.2) | 1 | 2026-09-08 | Calibrated VRAM + roofline for A100/H100 | ~1% MAPE vs real vLLM; MLA-correct; per-field `calibrated` flag; roofline intervals | Prediction only; offline calibration for A100/H100 only; no enforcement; no RSS proof harness | On-device calibration with held-out MAPE; enforcement gate; measured RSS proof |
| **llm-inference-calculator** (pochenai) | 20 | 2026-09-09 | Two-phase roofline; MoE sparsity; spec-decoding modelling | Rigorous two-phase model; data-centre multi-GPU scope | No calibration; no enforcement; no consumer-hardware focus | On-device calibration; single-machine consumer target; enforcement |
| **llm-roofline** (Pluenet-Killian) | 0 | 2026-06-20 (inactive) | Decode throughput floor = bytes/bandwidth per GPU | Simple clean derivation and chart | Throughput floor only; no memory prediction; no enforcement; inactive | Memory contract + enforcement + RSS proof |
| **hardware-aware-llm-runtime** (JohnScheuer) | 0 | 2026-06-25 (inactive) | Hardware-calibrated roofline; empirical optimal batch | Empirical constant fitting; finds compute/bandwidth crossover | Throughput focus; no enforcement; no stress harness; inactive | Memory-safety focus; enforcement gate after calibration |
| **llm-vram-calculator** (Shun-Calvin) | 1 | 2026-09-26 | Formula-based VRAM/tok/s for 100+ models × 70+ GPUs | Widest model×GPU coverage; public API | Formula-based, not calibrated; GPU-only; no enforcement | On-device calibration; enforcement; proof harness |
| **aura** (Grevix/aura, Rust, MIT/Apache-2.0) | 4 | 2026-09-03 | Kernel-level enforcement (cgroup v2 / Win32 Job Object); pre-execution feasibility model; context-ladder degradation; NVMe/GPU/SIMD diagnostics | More aggressive enforcement (OS-level) than fitsproof; four-tier memory hierarchy; `MetricProvenance` tagging | **BENCHMARK.md shows 4.92 GB peak against 4.00 GB budget** — no mechanism asserts `peak ≤ budget`; no held-out calibration protocol; no MAPE; no embeddable `plan`/`admit` API; no OpenAI/MCP plugin surfaces | Held-out MAPE (published even when bad); zero-violation stress harness as repo test; embeddable API surfaces |
| **detllm** (tommasocerruti) | 20 | 2026-08-20 | Capability-gated determinism tier reporting; repro packs | Honest tier framing (reports the tier actually achieved, never claims higher) | Determinism only; no memory prediction or enforcement | Adopts detllm tier model for verify layer (source 14); adds contract enforcement |

### Confirmed gap claim — state as of 2026-09-27T20:42Z

The narrowed claim from pass 3 survives this deepening pass:

**No single tool in the table above does all three of the following:**

1. Calibrate prediction constants from measurements on the user's own hardware with a
   held-out train/test split and a published MAPE. (ridgepoint calibrates, but against
   A100/H100 offline; aura probes hardware but documents no calibration protocol with
   held-out evaluation.)
2. Enforce a declared budget with a gate that emits a structured degradation record
   naming exactly what changed. (aura enforces at the OS level — more aggressive than
   fitsproof's in-process gate — but its BENCHMARK.md shows a run exceeding the
   declared budget without flagging it as a violation.)
3. Prove compliance: a test-suite-wired stress harness that asserts
   `measured_peak ≤ declared_budget` across ≥20 configurations and exits non-zero on
   any violation.

**How a user would notice:**

With aura, they set `--memory 4G`, run `qwen3:8b`, and get `Peak Working Set: 4.92 GB`
in the telemetry — 23% over budget, reported as a pass. With fitsproof, the stress
harness fails the build if any of ≥20 measured peaks exceeds the declared budget
(acceptance criterion 7). The contract either holds or the run fails loudly; there
is no middle ground.

**Watch items:**

1. **aura at publication time.** aura is active (4 stars, Rust, active contributor
   call). If it ships (a) a held-out calibration protocol with published MAPE and (b) a
   CI-integrated stress harness asserting zero violations, the gap closes. Check its
   repo before the fitsproof release commit.
2. **llama.cpp budget enforcement.** Confirmed absent in v0.5.0 (c2-p1 item 10).
   Re-check at publication time.

### Falsification for Cycle 2 — Pass 2

Observations that would prove this pass's findings wrong:

1. **aura's BENCHMARK.md figure is a different memory unit than the Job Object budget.**
   If Win32 Job Object limits Working Set in one unit and aura reports Peak Working Set
   in a different one, the 4.92 vs 4.00 comparison is not apples-to-apples.
   Observable: run the same config and compare `task manager → working set` vs
   `aura telemetry → Peak Working Set` on the same Windows machine. We cannot run
   this (Linux-only dev box); this remains an open measurement question. The conservative
   reading is that the figure is measured in the same unit as the budget (aura labels it
   identically, `Provenance: aura_measured`, `Simulated: false`).

2. **ashcakeancient7671/aura is a fork with different features from Grevix/aura.**
   GitHub API reported `fork: false` and the README confirms independent authorship.
   If further inspection reveals a shared codebase with features not in Grevix/aura,
   this entry should be updated.

3. **acasavaraju/AIOS ships a working runtime after this pass.**
   Last push is 2026-03-29 (6 months ago at time of writing). If it ships a working
   inference runtime with a calibrated contract, it enters the comparison table as a
   competitor. Re-check at publication time.

4. **A search with different query terms finds a tool not found here.**
   The adversarial reviewer should repeat with "LLM resource budget enforcement CLI"
   and "inference memory proof harness" to independently validate the table is complete.

---

## Cycle 2 — Pass 2 (this session) — STAR COUNT REFRESH + NEW TOOLS

*Dispatched 2026-09-27T22:07Z. Star counts retrieved via GitHub REST API at
2026-09-27T22:10Z (raw batch output below). This pass: (1) refreshes all named
tools to the session timestamp, (2) evaluates two new repos that appeared in a
repeat search run, (3) records the narrow delta vs the prior c2-p2 session.*

### Raw star-count refresh (2026-09-27T22:10Z)

All calls made in parallel with `curl -s https://api.github.com/repos/<owner>/<repo>`:

```
ggml-org/llama.cpp               | stars=129700 | push=2026-09-27T20:07:41Z | license=MIT
vllm-project/vllm                | stars=92792  | push=2026-09-27T20:43:56Z | license=Apache-2.0
kvcache-ai/ktransformers         | stars=19541  | push=2026-09-23T05:07:33Z | license=Apache-2.0
Isk4R1oT/ridgepoint              | stars=1      | push=2026-09-08T18:40:09Z | license=MIT
pochenai/llm-inference-calculator| stars=20     | push=2026-09-09T15:58:07Z | license=None
Pluenet-Killian/llm-roofline     | stars=0      | push=2026-06-20T19:26:33Z | license=MIT
JohnScheuer/hardware-aware-llm-runtime | stars=0 | push=2026-06-25T09:50:23Z | license=MIT
Shun-Calvin/llm-vram-calculator  | stars=1      | push=2026-09-26T06:38:02Z | license=MIT
tommasocerruti/detllm            | stars=20     | push=2026-08-20T21:07:45Z | license=Apache-2.0
Grevix/aura                      | stars=4      | push=2026-09-03T17:50:25Z | license=Apache-2.0
```

Deltas vs the prior c2-p2 session (20:42Z): llama.cpp −1 (rounding/race in API cache),
vLLM +0, KTransformers +0; all others unchanged. Rankings, conclusions, and gap claim
are stable across this refresh.

### New tools surfaced by repeat search (2026-09-27T22:10Z)

Re-ran `llm+memory+budget+enforcement&sort=updated&per_page=10`. Two entries not
evaluated in either prior c2-p2 session appeared:

#### mrshelll/baton (0 stars, Python, pushed 2026-09-25)

- **Description:** "Context handoff between Claude Code sessions with a document that
  doesn't grow: rewritten whole every time, with a hard budget enforced by code and
  derived from the harness's real 8000-character ceiling."
- **Conclusion:** enforces a *character/token context size* budget for agent session
  handoff, not a VRAM/RAM budget for model loading. Not a competitor. Does not enter
  the comparison table.

#### edouard-claude/longe (3 stars, Rust, pushed 2026-09-12, MIT)

- **Description:** "A self-improving harness for any LLM, in one Rust binary. Persistent
  Lua REPL as the single tool, three-level memory, persistent sub-agents, enforced
  budgets, external verification, per-process sandbox, and post-run reflection."
- **From README (fetched 2026-09-27T22:10Z):** the "enforced budgets" are *token and
  turn budgets* for the agent loop (context budget, sub-agent call limits), not memory
  budgets for model loading. There is no probe, calibrate, predict-peak, or RSS
  assertion in the codebase. The word "budget" refers to the harness's compute
  allowance, not the inference engine's memory footprint.
- **Conclusion:** different problem domain (agent harness budget vs inference memory
  contract). Not a competitor. Does not enter the comparison table.

  However, longe's existence is notable evidence for the adjacent claim: "budget
  enforcement" as a concept is now active in at least three distinct layers of the LLM
  stack — agent turn budgets (longe, baton), OS-level memory budgets (aura), and
  in-process inference memory budgets (fitsproof). The three layers are not
  interchangeable; fitsproof's budget is specifically `peak_RSS ≤ declared_bytes`
  during model loading and generation, not a context or turn limit.

### Gap claim — stable after this refresh

The narrowed claim from pass 3 / prior c2-p2 session stands:

**No single tool in the comparison table does all three:**

1. Calibrate prediction constants from measurements on the user's own hardware with a
   train/hold-out split and a published held-out MAPE.
2. Enforce a declared budget with a structured degradation record naming what changed.
3. Prove compliance: a repo test (stress harness) asserts `measured_peak ≤ budget`
   across ≥20 configurations and exits non-zero on any violation.

This session's search found no new tool that closes the gap. Watch items from prior
c2-p2 (aura adding calibration + zero-violation harness; llama.cpp shipping `--budget`)
are unchanged.

### Falsification — additions for this session

5. **`edouard-claude/longe` or `mrshelll/baton` adds VRAM/RAM memory-budget
   enforcement as a feature.** Both are active repos (pushed 2026-09-12 and 2026-09-25).
   If either adds peak-memory assertion (not context-size assertion), the enforcement
   half of the claim narrows further. Not observed as of this session.

6. **The "budget enforcement" term has drifted toward agent-harness budgets** (longe,
   baton) in the GitHub search results, which means future adversarial reviewers searching
   for competitors may find these tools rather than memory-contract tools. The adversarial
   review should explicitly specify "peak RAM budget enforcement for LLM loading" in its
   search to distinguish the two layers.
