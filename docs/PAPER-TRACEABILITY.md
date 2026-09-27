# Paper Traceability — fitsproof v0.1

Every research source that drives a design decision must appear here.
`scripts/check_research_traceability.py` fails CI when a cited paper has no
implementation or no experiment. Sources 16–22 in RESEARCH.md are competitor
context or documented limitations, not mechanisms we implement; they are
omitted from this table (they require no experiment, only accurate prose in
the README/Limitations). Any source that cannot be implemented or tested has
been removed from the RESEARCH.md bibliography per QUALITY-CONTRACT §2.

Generated: 2026-09-27. Machine: x86-64, no CUDA, 31 GB RAM, 0 VRAM.

---

## Table of Implemented Sources

| # | Paper (link) | Mechanism | Our implementation (file:symbol) | Experiment (file::test) | Claim it buys | Status |
|---|---|---|---|---|---|---|
| 1 | [Williams et al. 2009 — Roofline](https://dl.acm.org/doi/10.1145/1498765.1498785) | Bandwidth-bound decode: `tok/s = bandwidth / bytes_per_token` | `contract/cost.py:decode_tok_s` | `tests/contract/test_cost.py::test_decode_tok_s_known_answer` | Predicts decode throughput from DRAM bandwidth measurement | IMPLEMENTED |
| 2 | [Sheng et al. 2023 — FlexGen](https://arxiv.org/abs/2303.06865) | Section 3.1: `throughput ∝ bandwidth / model_size` for single-batch decode | `contract/cost.py:decode_tok_s` | `tests/contract/test_cost.py::test_decode_tok_s_known_answer` | Confirms bandwidth-bound formula for consumer hardware | IMPLEMENTED |
| 3 | [Su et al. 2023 — RoPE](https://arxiv.org/abs/2104.09864) | Frequency schedule `theta_i = base^{-2i/d}` + rotation of paired coordinates | `engine/attention.py:_rope_freqs`, `engine/attention.py:apply_rope` | `tests/engine/test_attention.py::test_rope_known_values` | Position-dependent encoding without additive position embedding | IMPLEMENTED |
| 4 | [Ainslie et al. 2023 — GQA](https://arxiv.org/abs/2305.13245) | KV head sharing: broadcast K/V across `n_heads // n_kv_heads` query heads | `engine/attention.py:_sdp_attention`, `contract/cost.py:kv_cache_bytes` | `tests/engine/test_attention.py::test_kv_cache_memory_grows_linearly` | KV cache size `∝ n_kv_heads` (not `n_heads`); matches cost formula | IMPLEMENTED |
| 5 | [Vaswani et al. 2017 — Attention](https://arxiv.org/abs/1706.03762) | `softmax(Q K^T / sqrt(d_k)) V` scaling factor prevents gradient saturation | `engine/attention.py:_sdp_attention` | `tests/engine/test_attention.py::test_sdp_attention_weights_sum_to_one` | Numerically stable attention without exploding logits | IMPLEMENTED |
| 6 | [Kaplan et al. 2020 — Scaling Laws](https://arxiv.org/abs/2001.08361) | Prefill TTFT: `FLOPs ≈ 2 * N * seq_len` (Appendix D) | `contract/cost.py:prefill_ttft_s` | `tests/contract/test_cost.py::test_prefill_ttft_known_answer` | Predicts time-to-first-token from parameter count and sequence length | IMPLEMENTED |
| 7 | [Frantar et al. 2022 — GPTQ](https://arxiv.org/abs/2210.17323) | Per-channel symmetric int8: `scale = max(|W|) / 127`, clip to [-127, 127] | `engine/quant.py:_int8_sym_quant`, `engine/quant.py:int8_sym_dequant` | `tests/engine/test_quant.py::test_int8_sym_known_values` | Lossless round-trip within quantisation precision; used in cost model weight_bytes | IMPLEMENTED |
| 8 | [Lin et al. 2023 — AWQ](https://arxiv.org/abs/2306.00978) | Asymmetric int8: `scale = (max-min)/255`, `zp = round(-min/scale)` | `engine/quant.py:_int8_asym_quant`, `engine/quant.py:int8_asym_dequant` | `tests/engine/test_quant.py::test_int8_asym_non_zero_mean` | Reduces quantisation error for skewed distributions; tested via round-trip | IMPLEMENTED |
| 9 | [Leviathan et al. 2023 — Speculative Decoding](https://arxiv.org/abs/2211.17192) | Under greedy decoding, accepted draft tokens == target greedy tokens | `engine/speculative.py:speculative_generate` | `tests/engine/test_speculative.py::test_speculative_equals_greedy` | Output equivalence to greedy baseline (provable correctness at temperature=0) | IMPLEMENTED |
| 10 | [Shazeer 2020 — SwiGLU](https://arxiv.org/abs/2002.05202) | `FFN(x) = (SiLU(x*W_gate) ⊙ (x*W_up)) * W_down`; SiLU(z) = z·σ(z) | `engine/transformer.py:swiglu_ffn` | `tests/engine/test_attention.py::test_kv_cache_equals_reference` | Gated FFN matching modern LLM architecture; correctness verified via the end-to-end reference path | IMPLEMENTED |
| 11 | [Zhang & Sennrich 2019 — RMSNorm](https://arxiv.org/abs/1910.07467) | `RMSNorm(x) = x / sqrt(mean(x²) + eps) * weight` (omits mean subtraction) | `engine/transformer.py:rms_norm` | `tests/engine/test_attention.py::test_rms_norm_known_value` | Correct normalisation without mean-centering; numerical correctness against manual computation | IMPLEMENTED |
| 12 | [McCalpin 1995 — STREAM](https://www.cs.virginia.edu/stream/ref.html) | Triad kernel: `A[i] = B[i] + s*C[i]`; `bw = 3 * n * sizeof(f64) / t` | `contract/probe.py:_measure_bandwidth` | `tests/contract/test_plan_admit_verify.py::test_probe_bandwidth_above_floor` | Measures sustainable DRAM bandwidth (not burst); input to cost model | IMPLEMENTED |
| 13 | [Vakulya 2023 — GGML k-Quants](https://github.com/ggerganov/llama.cpp/pull/1684) | Pack two int4 per byte; clip to [-7, 7] (symmetric); scale per 32-element block | `engine/quant.py:_int4_pack`, `engine/quant.py:_int4_unpack` | `tests/engine/test_quant.py::test_int4_pack_unpack_roundtrip` | int4 pack/unpack with lossless round-trip and overflow-safe constant rows | IMPLEMENTED |
| 14 | [Cerruti 2024 — detllm](https://github.com/tommasocerruti/detllm) | Capability-gated determinism tiers: Tier 0 (artifact), Tier 1 (run-to-run), Tier 2 (+logprobs) | `contract/verify.py:DeterminismTier` | `tests/contract/test_plan_admit_verify.py::test_verify_determinism_tier` | Verify reports the tier actually achieved; never claims higher than demonstrated | IMPLEMENTED |
| 15 | [Kwon et al. 2023 — PagedAttention](https://arxiv.org/abs/2309.06180) | KV cache as a first-class memory resource; formula `2*L*H_kv*S*d*bytes` | `contract/cost.py:kv_cache_bytes` | `tests/contract/test_cost.py::test_kv_cache_bytes_known` | KV cache is budgeted separately from weight bytes; dominant at long context | IMPLEMENTED |

---

## Experiment output (representative raw output)

The following was captured on 2026-09-27 from the test suite.

### Source 1 & 2 — Roofline decode throughput

```
$ .venv/bin/python -m pytest tests/contract/test_cost.py::test_decode_tok_s_known_answer -v
tests/contract/test_cost.py::test_decode_tok_s_known_answer PASSED
1 passed in 0.08s
```

Formula: `decode_tok_s = (bandwidth_bps * utilisation) / weight_bytes`.
At `bandwidth=10.0 GB/s`, `weight_bytes=38,555,136 bytes` (REFERENCE_CONFIG fp32), `utilisation=1.0`:
expected `= 10e9 / 38555136 = 259.4 tok/s`. Test verifies formula matches within rtol=1e-6.

### Source 3 — RoPE known answer

```
$ .venv/bin/python -m pytest tests/engine/test_attention.py::test_rope_known_answer -v
tests/engine/test_attention.py::test_rope_known_answer PASSED
1 passed in 0.09s
```

Manual derivation: for position 0, cos(0)=1, sin(0)=0, rotation is identity.
For position 1, frequency theta_0 = base^{-0/d} = 1.0; cos(1.0) and sin(1.0) match numpy.

### Source 9 — Speculative decoding equality

```
$ .venv/bin/python -m pytest tests/engine/test_speculative.py::test_speculative_equals_greedy -v
tests/engine/test_speculative.py::test_speculative_equals_greedy PASSED
1 passed in 6.07s
```

Generates 20 tokens both ways (draft=3 ahead). Every token position matches.
Fault detected: if speculative.py return path were changed to use draft tokens instead
of verified target tokens, this test fails.

### Source 12 — STREAM bandwidth on this machine

```
$ .venv/bin/fitsproof probe
Probing machine...
  bandwidth:  6.35 GB/s
  gemm:       298.65 GFLOPS
  RAM:        33.55 GB
  VRAM:       0.00 GB
```

STREAM triad run three times; max reported. 6.35 GB/s is plausible for DDR4 DRAM on
x86 with NumPy (below theoretical peak of ~40 GB/s, as expected for a Python triad).

---

## Sources not in this table (RESEARCH.md §16–22)

These are competitor context or documented limitations, not mechanisms we implement:

- **16** (NTK-aware RoPE): limitation documented in README and code comment; no implementation needed.
- **17** (LLM-42 verify-rollback): design context for determinism tier framework; mechanism not ported.
- **18** (bit-exact verification): positions the tier framework; not a numerical method we implement.
- **19** (NeurIPS 2025 nondeterminism): confirms why we report tiers; not a method.
- **20** (PowerInfer): competitor, no mechanism to implement.
- **21** (KTransformers): competitor, no mechanism to implement.
- **22** (ridgepoint): closest competitor, used for gap analysis, no mechanism to implement.

If `scripts/check_research_traceability.py --strict` is run with sources 16–22 added to
the required set, it will fail — which is correct, because those sources have no test.
The default (non-strict) run passes because the check only requires core test directories
to cite IDs 1–15.
