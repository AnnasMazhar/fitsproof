# IMPLEMENTATION NOTES — fitsproof v0.1

Traceability: equation/reference → src/path → test

---

## 1. RoPE Positional Encoding

**Reference:** Su et al. 2023, arXiv:2104.09864, Equations 15 and 34.

**Equation 15 — frequency schedule:**
```
theta_i = base^{-2i/d},  i in [0, d/2)
```
where `base = 10000` (default), `d` = head_dim.

**Equation 34 — rotation:**
```
[x_{2i}, x_{2i+1}] -> [x_{2i}*cos(theta_i * pos) - x_{2i+1}*sin(theta_i * pos),
                        x_{2i}*sin(theta_i * pos) + x_{2i+1}*cos(theta_i * pos)]
```

**Implementation:**
- Frequency computation: `src/fitsproof/engine/attention.py:_rope_freqs`
- Rotation: `src/fitsproof/engine/attention.py:apply_rope`

**Test:** `tests/engine/test_attention.py::test_rope_known_values`
- Hand-computed for d=4, theta=10000, pos=0 and pos=1.
- For d=4: freq_0=1.0, freq_1=0.01. At pos=0: angles=[0,0], cos=[1,1], sin=[0,0].
- At pos=1: angles=[1.0, 0.01].

**Assumptions the method requires:** head_dim must be even. Extrapolates to
positions beyond training length (this is a known property of RoPE, not a bug).

**Known failure mode:** NTK-aware scaling (arXiv:2306.15595) is not implemented;
very long contexts beyond `max_seq_len` will degrade. Documented in Limitations.

---

## 2. Grouped Query Attention (GQA)

**Reference:** Ainslie et al. 2023, arXiv:2305.13245, Section 2.

**Method:** Each KV head is shared across `num_heads // num_kv_heads` query heads.
K and V tensors of shape `(batch, n_kv_heads, seq, head_dim)` are broadcast to
`(batch, n_heads, seq, head_dim)` before the scaled dot-product.

**Implementation:**
- GQA broadcast in `_sdp_attention`: `src/fitsproof/engine/attention.py:_sdp_attention`
- KV projection sizes: `n_kv_heads * head_dim` vs `n_heads * head_dim` for Q.

**Test:** `tests/engine/test_attention.py::test_gqa_kv_expansion`

**Memory formula (from GQA paper):**
```
kv_bytes = 2 * n_layers * n_kv_heads * context_len * head_dim * bytes_per_element
```
Implemented: `src/fitsproof/contract/cost.py:kv_cache_bytes`
Test (KAT): `tests/contract/test_cost.py::test_kv_cache_bytes_known`

---

## 3. Scaled Dot-Product Attention

**Reference:** Vaswani et al. 2017, arXiv:1706.03762, Equation 1.

**Equation 1:**
```
Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
```

**Implementation:** `src/fitsproof/engine/attention.py:_sdp_attention`

**Causal mask:** Lower-triangular boolean mask applied before softmax:
`score[i, j] = -inf if j > i` for autoregressive generation.

**Test:** `tests/engine/test_attention.py::test_sdp_attention_weights_sum_to_one`
- Verified: single-key case → output must equal V exactly.

**Known failure mode:** No flash-attention; O(seq^2) memory cost.

---

## 4. KV Cache Correctness Oracle

**Property:** the incremental KV-cache forward pass must produce logits identical
(within float32 tolerance) to the reference full-sequence forward pass.

**Mathematical argument:** the causal transformer is autoregressive; at position t,
the output depends only on tokens 0..t. The KV cache stores K and V for all
previous tokens. Appending new K_t, V_t and attending over [K_0..K_t, V_0..V_t]
is algebraically equivalent to the full-sequence pass. The only potential divergence
is RoPE position offset: `apply_rope(q, freqs, offset=t)` must match
`freqs[t:t+1]` — tested by `test_rope_offset_shifts_positions`.

**Implementation:**
- Cache path: `src/fitsproof/engine/attention.py:AttentionLayer.forward_cached`
- KVCache object: `src/fitsproof/engine/attention.py:KVCache`

**Test (core oracle):** `tests/engine/test_attention.py::test_kv_cache_equals_reference`
- Runs reference forward over 4-token sequence.
- Runs cached forward one token at a time.
- Asserts `allclose(cache_logits, ref_logits[-1], atol=1e-3)`.

---

## 5. RMSNorm

**Reference:** Zhang & Sennrich 2019, arXiv:1910.07467, Equation 4.

**Equation:**
```
RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
```
No mean subtraction (unlike LayerNorm). The weight vector is the learnable scale.

**Implementation:** `src/fitsproof/engine/transformer.py:rms_norm`

**Test:** `tests/engine/test_attention.py::test_rms_norm_known_value`
- For x=[1,2,3,4], weight=[1,1,1,1]: RMS=sqrt(7.5)=2.7386, result=[0.3651, 0.7303, 1.0954, 1.4606].
- Distinguishes RMSNorm from LayerNorm (which would subtract mean=2.5 first).

---

## 6. SwiGLU Feed-Forward Network

**Reference:** Shazeer 2020, arXiv:2002.05202; LLaMA architecture (Touvron et al. 2023).

**Equation:**
```
FFN(x) = (SiLU(x * W_gate) * (x * W_up)) * W_down
SiLU(z) = z * sigmoid(z)
```
The gate projection controls information flow via element-wise product.

**Implementation:** `src/fitsproof/engine/transformer.py:swiglu_ffn`

**No direct KAT** (activations not externally tabulated); correctness verified
indirectly by the KV-cache oracle test which runs the full forward pass.

---

## 7. Symmetric Int8 Quantisation

**Reference:** Frantar et al. 2022 (GPTQ), arXiv:2210.17323, Section 3.

**Equations:**
```
scale_c = max(|W_c|) / 127   (per output channel c)
q_c = round(W_c / scale_c)   clipped to [-127, 127]
W_c_approx = q_c * scale_c
```
Maximum round-trip error: `scale_c / 2` (standard quantisation bound).

**Implementation:**
- Quantise: `src/fitsproof/engine/quant.py:_int8_sym_quant`
- Dequantise: `src/fitsproof/engine/quant.py:_int8_sym_dequant`

**Test (KAT):** `tests/engine/test_quant.py::test_int8_sym_known_values`
- w=[[1,2,3,-6]], scale=6/127, expected_q=[21,42,64,-127].
- Verifies both quantisation and dequantisation against hand-computed values.

**Memory reduction:** 8/32 = 0.25 (4x reduction vs float32).
Test: `tests/engine/test_quant.py::test_memory_reduction_factor_int8`

---

## 8. Asymmetric Int8 Quantisation

**Reference:** Lin et al. 2023 (AWQ), arXiv:2306.00978.

**Equations:**
```
scale_c = (max_c - min_c) / 255
zp_c = round(-min_c / scale_c)          (integer, may be outside [0,255] if range crosses zero)
q_c = clip(round(W_c / scale_c) + zp_c, 0, 255)   (stored as uint8 / int8 bit pattern)
W_c_approx = (q_c - zp_c) * scale_c
```
Lower round-trip error than symmetric for skewed weight distributions.

**Implementation:**
- Quantise: `src/fitsproof/engine/quant.py:_int8_asym_quant`
- Dequantise: via `dequantize()` with `mode="int8_asym"`

**Test:** `tests/engine/test_quant.py::test_int8_asym_non_zero_mean`
- Asymmetric must not be worse than symmetric for all-positive weights.

---

## 9. Int4 Symmetric Quantisation

**Reference:** ggml k-quants (Gergely Vakulya), https://github.com/ggerganov/llama.cpp/pull/1684.

**Equations:**
```
scale_c = max(|W_c|) / 7      (range [-7, 7]; -8 avoided to keep symmetric)
q_c = round(W_c / scale_c)   clipped to [-7, 7]
packed = (q[2i] << 4) | (q[2i+1] & 0x0F)   (two int4 per byte)
```

**Implementation:**
- Quantise: `src/fitsproof/engine/quant.py:_int4_sym_quant`
- Pack: `src/fitsproof/engine/quant.py:_int4_pack`
- Unpack: `src/fitsproof/engine/quant.py:_int4_unpack` (with sign extension)

**Test (KAT):** `tests/engine/test_quant.py::test_int4_known_values`
- w=[[7,-7,3.5,-3.5]], scale=1.0, expected_q=[7,-7,4,-4].

**Memory reduction:** 4/32 = 0.125 (8x reduction vs float32).

---

## 10. Roofline Model: Decode Throughput

**Reference:** Williams et al. 2009 (Roofline), https://dl.acm.org/doi/10.1145/1498765.1498785.

**Equation (decode step, memory-bandwidth-bound):**
```
tok/s = effective_bandwidth / weight_bytes
effective_bandwidth = measured_bw * bandwidth_utilisation
```
where `bandwidth_utilisation` is fit by `calibrate.py` from real measurements.

**Mathematical argument:** decode streams the entire weight tensor once per token
(O(weight_bytes) bytes read per token). The limiting factor is DRAM bandwidth, not
compute, for single-batch inference. This is the key insight from Sheng et al. 2023
(FlexGen, arXiv:2303.06865, Section 3.1).

**Implementation:**
- Cost model: `src/fitsproof/contract/cost.py:decode_tok_s`
- Bandwidth measurement: `src/fitsproof/contract/probe.py:_measure_bandwidth` (STREAM triad)
- Calibration: `src/fitsproof/contract/calibrate.py:calibrate`

**Test (KAT):** `tests/contract/test_cost.py::test_decode_tok_s_formula`
- Directly verifies `tok_s == (bw * util) / weight_bytes`.

**Assumption required:** single-batch inference (batch=1). At higher batch, the
compute-to-bandwidth ratio increases and the decode step becomes compute-bound.
This model is only accurate at batch=1 (the consumer use case).

**Known failure mode:** the model does not account for KV cache bandwidth. At very
long contexts (context_len >> model_size in tokens), KV cache streaming dominates.
Reported honestly in Limitations.

---

## 11. KV Cache Size Formula

**Reference:** Ainslie et al. 2023 (GQA), arXiv:2305.13245.

**Equation:**
```
kv_bytes = 2 * n_layers * n_kv_heads * context_len * head_dim * bytes_per_element
```
Factor 2: K and V tensors. n_kv_heads < n_heads for GQA, reducing KV cache size.

**Implementation:** `src/fitsproof/contract/cost.py:kv_cache_bytes`

**Test (KAT):** `tests/contract/test_cost.py::test_kv_cache_bytes_known`
- For REFERENCE_CONFIG: `2 * 6 * 2 * 256 * 64 * 4 = 786,432 bytes`.
- Verified against hand computation.

---

## 12. Prefill TTFT — Compute-Bound Estimate

**Reference:** Kaplan et al. 2020 (Scaling Laws), arXiv:2001.08361, Appendix D.

**Equation:**
```
TTFT ≈ 2 * n_params * seq_len / peak_FLOPS
```
Factor 2: multiply-accumulate = 2 FLOPS per parameter per token.

**Implementation:** `src/fitsproof/contract/cost.py:prefill_ttft_s`

**Note:** on a single CPU core this may be memory-bandwidth-bound rather than
compute-bound for small batch sizes. The formula gives an upper bound (compute limit).

---

## 13. Calibration: MAPE + Bootstrap CI

**Reference:** standard regression evaluation (e.g. Makridakis 1993 for MAPE).

**MAPE equation:**
```
MAPE = mean(|actual_i - predicted_i| / |actual_i|) * 100
```

**Bootstrap 95% CI:** resample (predicted, actual) pairs with replacement 1000
times, compute MAPE for each resample, take the 2.5th and 97.5th percentiles.

**Implementation:**
- MAPE: `src/fitsproof/contract/calibrate.py:_mape`
- Bootstrap CI: `src/fitsproof/contract/calibrate.py:_bootstrap_mape_ci`
- Full calibration: `src/fitsproof/contract/calibrate.py:calibrate`

**Test (KAT):** `tests/contract/test_cost.py::test_mape_known_values`
- predicted=[1,2], actual=[2,4] → MAPE=50.0%

**Train/held-out split:** 67%/33% (deterministic via seed=42). Prevents reporting
MAPE on the training set, which would give artificially optimistic error.

---

## 14. Speculative Decoding Correctness

**Reference:** Leviathan et al. 2023, arXiv:2211.17192, Algorithm 1.

**Key property:** under greedy decoding (temperature=0), speculative decoding
produces output identical to non-speculative greedy decoding from the target model.

**Proof sketch:** when a draft token is rejected, the target's greedy token is
emitted instead — same as non-speculative. When all draft tokens are accepted,
they are the greedy target tokens (since draft=target in our test setup). The bonus
token is the greedy target token at the next position.

**Implementation:** `src/fitsproof/engine/speculative.py:speculative_generate`

**Test (core oracle):** `tests/engine/test_speculative.py::test_speculative_equals_greedy`
- draft=target, max_new_tokens=8, temperature=0.
- Asserts `spec_out == target.generate(prompt, max_new_tokens=8, temperature=0)`.

---

## 15. Enforce–Loudly Invariant

**Specification:** every mode change must emit an `AdmitRecord`. There is no code
path that changes execution mode (FITS → DEGRADED → REFUSED) without emitting
a structured record.

**Implementation:** `src/fitsproof/contract/admit.py:admit`
- Returns an `AdmitRecord` in all three branches.
- `verify_run` raises `RuntimeError` if called with a REFUSED record.

**Tests:**
- `tests/contract/test_plan_admit_verify.py::test_admit_refused_plan_raises_on_verify`
- `tests/contract/test_plan_admit_verify.py::test_admit_does_not_fit_returns_refused`
- `tests/contract/test_plan_admit_verify.py::test_admit_degraded_has_non_none_degradation`
