# RESEARCH — fitsproof v0.1

All links verified to resolve at the time of writing (2026-09-26).
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
