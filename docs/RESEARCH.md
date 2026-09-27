# RESEARCH — fitsproof v0.1

All links verified to resolve at the time of writing (2026-09-27).
Pass 1 (2026-09-26): sources 1–15, ground truth. Updated with 7 new sources and
extended falsification in pass 1 (2026-09-27).
Each citation is attached to the specific claim it supports.

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
