# ADVERSARIAL REVIEW — fitsproof

Independent review, pass `c1-p10-adversarial-1` (cycle 1, adversarial pass 1 of 3).
Reviewer lane: opencode (mimo-v2.6-flash-free). The reviewer does not fix code — it
reports; the builder fixes; the reviewer re-verifies in pass 11.

Baseline before attack (repo state `ef82b10`, branch `feat/v0.1`):

```
$ .venv/bin/pytest -q
======================= 155 passed in 166.77s (0:02:46) ========================
$ .venv/bin/ruff check . && .venv/bin/ruff format --check .
All checks passed!
$ .venv/bin/python scripts/check_research_traceability.py
TRACEABILITY OK (core only): all core test files cite valid research sources. Checked 22 source IDs from RESEARCH.md. PAPER-TRACEABILITY.md table validated (15 IMPLEMENTED rows).
```

All fault injections below were reverted with `git checkout -- <file>` immediately
after each run; `git status --short` was empty after the review (repo green).

---

## 1. Claims audit — the 3 most load-bearing README claims, attacked

### Claim C1 (HEADLINE): "fitsproof stress runs 25 configurations against a declared budget and fails the build on any violation or undocumented mode change."

Attack 1a — reproduce the published output:

```
$ .venv/bin/fitsproof stress
ADMITTED: 0.039 GB predicted peak <= 4.000 GB budget (margin: 3961.0 MB)
Stress harness: 25 configs, 0 violations, 0 silent mode changes. Margin: min=3697.9 MB, median=3697.9 MB, max=3697.9 MB.
rc=0
```

Verdict: the default-budget run reproduces (README prints 3698.3 MB, this run
3697.9 MB — RSS jitter only).

Attack 1b — can the harness ever FAIL? (a "0 violations" that cannot be non-zero is
not evidence):

```
$ .venv/bin/fitsproof stress --budget-gb 0.01
DEGRADED: base config needs 0.039 GB > budget 0.010 GB. Applying: Use int4_sym quantisation instead of none. New predicted peak: 0.006 GB.
Stress harness: 25 configs, 25 violations, 0 silent mode changes. Margin: min=-292.3 MB, median=-292.3 MB, max=-292.3 MB.
rc=1
```

Verdict: violation detection has teeth (rc=1, build fails). **But two attacks on
this claim succeeded:**

Attack 1c — the "undocumented mode change" half of the claim:

```
src/fitsproof/contract/verify.py (verify_run):
    # Check for silent mode changes: if admit_record is DEGRADED, the
    # applied degradation must be described; we can only check
    # the record exists (runtime mode enforcement is in admit.py).
    mode_changed_silently = False  # no silent changes if admit() was called

$ .venv/bin/python - <<'PY'
# feed verify_run a DEGRADED record (a mode change did happen)
... verify_run(fn, 10**9, AdmitRecord(status=AdmitStatus.DEGRADED, ...), "demo") ...
PY
mode_changed_silently = False (hardcoded False in verify.py)
```

`mode_changed_silently` is a hardcoded constant. The `silent_mode_changes`
counter in `StressResult` sums a value that is always False, so "0 silent mode
changes" is unfalsifiable: **no fault can make this metric non-zero.** See ADV-03.

Attack 1d — the margin distribution:

```
Margin: min=3697.9 MB, median=3697.9 MB, max=3697.9 MB   (identical, both runs)
```

min = median = max is not a distribution. `_get_rss_bytes()` returns
`resource.getrusage(...).ru_maxrss` — the **process-lifetime high-water mark** —
so all 25 configs measured in one process necessarily report the same number once
the first config sets it. The 25 margins are one measurement repeated 25 times;
the harness cannot attribute any peak to any configuration. See ADV-04.

### Claim C2 (QUICKSTART): "`fitsproof admit --budget-gb 0.001   # REFUSED — names the binding constraint, exit code 2`"

```
$ .venv/bin/fitsproof admit --budget-gb 4 >/dev/null 2>&1; echo "budget4 rc=$?"
budget4 rc=0
$ .venv/bin/fitsproof admit --budget-gb 0.001
REFUSED: needs 0.042 GB, budget 0.001 GB; no listed option fits — nearest is "Use int4_sym quantisation instead of none" at 0.006 GB (0.005 GB above budget)
Degradation options:
  [does not fit] Use int8_sym quantisation instead of none -> 0.011 GB
  [does not fit] Use int4_sym quantisation instead of none -> 0.006 GB
  [does not fit] Reduce context to 256 tokens (1/2 of 512) -> 0.040 GB
  [does not fit] Reduce context to 128 tokens (1/4 of 512) -> 0.039 GB
  [does not fit] Reduce context to 64 tokens (1/8 of 512) -> 0.039 GB
  [does not fit] Offload ~50% of layers to system RAM (CPU fallback for those layers) -> 0.022 GB
rc=2
```

Verdict: **claim holds.** Exit 0 on admit, exit 2 on refusal, binding constraint
named, every listed option honestly tagged `[does not fit]`, nearest option's gap
stated. The F-2 fix from the improve pass is real.

### Claim C3 (SERVER): "A request whose predicted peak exceeds the server's budget is refused with HTTP 503 and the binding constraint named — never a silent OOM."

Live attack — start the server with `budget_bytes=1024` and POST a completion:

```
$ .venv/bin/python - <<'PY'   # start_server(..., budget_bytes=1024); POST /v1/chat/completions
HTTP 503
{"error": {"message": "REFUSED: needs 0.039 GB, budget 0.000 GB; no listed option fits \u2014 nearest is \"Use int4_sym quantisation instead of none\" at 0.006 GB (0.006 GB above budget)", "type": "fitsproof_refused"}, "fitsproof": {"admission": "refused", "verdict": "does_not_fit", "message": "REFUSED: needs 0.039 GB, budget 0.000 GB; no listed option fits \u2014 nearest is \"Use int4_sym quantisation instead of none\" at 0.006 GB (0.006 GB above budget)", "predicted_peak_bytes": 38605312, "budget_bytes": 1024}}
```

Verdict: **claim holds behaviourally**, and the suite defends it — deleting the
refusal path makes the suite fail (fault injection F-I6 below).

Supporting checks:

- Quickstart install URL resolves: `curl -o /dev/null -w '%{http_code}' https://github.com/AnnasMazhar/fitsproof` → `200` (`.git` → `301`).
- "every snippet below is executed by the test suite": verified structurally —
  `tests/value/test_readme_snippets.py` regex-extracts the ```python blocks *from
  README.md itself* at test time (not a copied fixture), so a stale README snippet
  fails the suite. Claim holds.
- "Prediction accuracy … reproduce: `python scripts/calibration_demo.py`" — **does
  not reproduce the printed numbers** (see ADV-06):

```
$ .venv/bin/python scripts/calibration_demo.py
=== Calibration demo ===
bandwidth: 1.79 GB/s
gemm:      13.78 GFLOPS
RAM:       33.5 GB
bandwidth_utilisation: 0.0195
MAPE (held-out):       64.3%
CI (95%):              [64.3%, 64.3%]
n_train=2, n_held_out=1
rc=0
```

README publishes 46.1% / gemm 37.19 GFLOPS / CI [46.1%, 46.1%]; this run gives
64.3% / 13.78 GFLOPS / CI [64.3%, 64.3%] on a loaded box.

---

## 2. Citation audit — every link in docs/RESEARCH.md

### 2a. Do they resolve? (all 32 extracted URLs, curl status)

```
200  https://api.github.com/search/repositories?q=llm+memory+budget+enforcement&per_page=8"
200  https://arxiv.org/abs/1706.03762
200  https://arxiv.org/abs/1910.07467
200  https://arxiv.org/abs/2001.08361
200  https://arxiv.org/abs/2002.05202
200  https://arxiv.org/abs/2104.09864
200  https://arxiv.org/abs/2210.17323
200  https://arxiv.org/abs/2211.17192
200  https://arxiv.org/abs/2303.06865
200  https://arxiv.org/abs/2305.13245
200  https://arxiv.org/abs/2306.00978
200  https://arxiv.org/abs/2306.15595
200  https://arxiv.org/abs/2309.06180
200  https://arxiv.org/abs/2312.12456
200  https://arxiv.org/abs/2506.09501
200  https://arxiv.org/abs/2601.17768
200  https://arxiv.org/abs/2606.00279
403  https://dl.acm.org/doi/10.1145/1498765.1498785
200  https://github.com/ggerganov/llama.cpp/pull/1684
200  https://github.com/ggml-org/llama.cpp
200  https://github.com/Isk4R1oT/ridgepoint
200  https://github.com/JohnScheuer/hardware-aware-llm-runtime
200  https://github.com/kvcache-ai/ktransformers
200  https://github.com/Pluenet-Killian/llm-roofline
200  https://github.com/pochenai/llm-inference-calculator
200  https://github.com/Shun-Calvin/llm-vram-calculator
200  https://github.com/tommasocerruti/detllm
200  https://github.com/vllm-project/vllm
200  https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/
200  https://pypi.org/project/ridgepoint/
200  https://pypi.org/project/ridgepoint/0.1.1/
200  https://www.cs.virginia.edu/stream/ref.html
```

Notes:
- The first line is a shell transcript inside RESEARCH.md (line 899), not a
  citation; the trailing `"` is part of the pasted curl command.
- The ACM DOI 403s automation. Resolved independently via Crossref:

```
CROSSREF: Roofline | Communications of the ACM | [[2009, 4]]
```

  Title, venue and date match the citation. Link resolves for humans; automation
  is bot-blocked (ADV-07, limitation).

### 2b. Do they support the claims attached? (full-text checks on the design drivers)

Method: RESEARCH.md attaches a `**Claim it supports:**` block to each source; I
fetched the ar5iv full text of the design-driving papers and grepped the exact
claim terms.

**RoFormer (arXiv 2104.09864)** — claim: frequency schedule + rotation formula in
`_rope_freqs`/`apply_rope`. Abstract confirms RoPE's positional rotation encoding.
**Supported.**

**GQA (arXiv 2305.13245)** — claim: KV-head sharing + KV-cache size formula
`2 * n_layers * n_kv_heads * seq * head_dim * bytes`. Abstract confirms GQA
(generalisation of MQA, KV heads shared across query groups). The byte formula is
a first-principles derivation documented in IMPLEMENTATION-NOTES; the paper
supplies the head-sharing structure it is derived from. **Supported** (formula is
derived, not quoted — correctly presented as such).

**Leviathan et al. (arXiv 2211.17192)** — claim: greedy speculative output is
identical to non-speculative greedy. Abstract: speculative decoding runs "without
any changes to the outputs". **Supported.**

**GPTQ (arXiv 2210.17323)** — claim: "GPTQ uses per-channel scaling (each output
channel has its own scale), which we adopt." Full-text grep result:

```
GPTQ: 'scale'/'scaling' hits: 17   (all examined)
  ... Table 7 shows results on WikiText2 when quantizing the biggest models to
  2-bit with varying group-sizes. At ≈2.2 bit (group-size 128; using FP16 scale
  and 2-bit zero point per group) ...
  [no occurrence of "per-channel"; no per-output-channel scale statement anywhere]
```

The paper describes **per-group** FP16 scales, not per-output-channel scales.
The claim as attached is not in the cited paper. **NOT SUPPORTED — ADV-02
(blocker).**

**FlexGen (arXiv 2303.06865)** — claim: "For single-batch LLM decode, memory
bandwidth is the bottleneck. FlexGen Section 3.1 explicitly derives
`throughput ∝ bandwidth / model_size`." Full-text grep result:

```
FLEXGEN: 'bandwidth' hits = 15 (all examined)
  [disk_to_cpu_bandwidth / ctog_bdw / gtoc_bdw transfer-cost terms in the
   offloading cost model; SSD and network bandwidth in the evaluation; no
   decode-throughput-vs-bandwidth derivation]
FLEXGEN sections: 1 Introduction / 2 Related Work / 3 Background /
  4 Offloading Strategy (4.1 Problem Formulation, 4.3 Cost Model and Policy Search) ...
```

There is no §3.1 deriving `throughput ∝ bandwidth / model_size`; §3 is
"Background", and the paper's cost model is an *offloading transfer* model. The
same mis-attribution appears in code: `src/fitsproof/contract/cost.py:decode_tok_s`
docstring says "Source: Sheng et al. 2023 (FlexGen), Sec 3.1".
**NOT SUPPORTED — ADV-01 (blocker).** (The underlying roofline fact is real and
is supported by source 1, Williams et al. 2009 — this is an attribution error,
not a false engineering claim.)

**Roofline (doi 10.1145/1498765.1498785)** — resolves via Crossref (above);
claim (memory-bandwidth-bound kernels) is the paper's central result.
**Supported.**

Remaining links: resolution checked for all (2a); content spot-checked for the
six design drivers above. The remaining arXiv/ GitHub links attach
ecosystem/context claims (verified-resolving only) — flagged honestly as
not content-audited in this pass.

---

## 3. Fault injection — 6 tests sampled, fault each claims to detect

| # | Test (its named fault) | Injected fault | Suite result |
|---|---|---|---|
| F-I1 | `test_refusal_never_names_a_non_fitting_config` (F-2: refusal naming a non-fitting "nearest fitting config") | plan.py: message reverted to `nearest fitting config is "..."` | **FAILED (killed)** |
| F-I2 | `test_int8_sym_known_values` ("using 128 instead of 127 clips the range asymmetrically") | quant.py: `scales = max_abs / 128.0` | **FAILED (killed)** |
| F-I3 | `test_decode_tok_s_known_answer` (wrong roofline tok/s formula) | cost.py: `return 2.0 * effective_bw / w_bytes` | **FAILED (killed)** |
| F-I4 | `test_kv_cache_equals_reference` ("if offset is wrong, RoPE positions are off") | attention.py: `apply_rope(q, ..., offset=0)` in `forward_cached` | **FAILED (killed)** |
| F-I5 | `test_speculative_equals_greedy` ("verification step accepts a wrong draft token") | speculative.py: `if greedy_target == draft_tok:` → `if True:` (accept-all) | **PASSED (SURVIVED)** |
| F-I6 | server 503 refusal path (suite-level) | server.py: `if record.status == AdmitStatus.REFUSED:` → `if False and ...` | **FAILED (killed)** — full suite |

Raw output:

```
### FAULT: F-I1 F-2 regression: refusal says 'nearest fitting config' again
FAILED tests/contract/test_plan_admit_verify.py::test_refusal_never_names_a_non_fitting_config
============================== 1 failed in 0.52s ===============================

### FAULT: F-I2 int8_sym scale uses 128 instead of 127 (GPTQ symmetric range)
FAILED tests/engine/test_quant.py::test_int8_sym_known_values - AssertionError:
============================== 1 failed in 0.49s ===============================

### FAULT: F-I3 decode roofline doubles bandwidth (wrong tok/s formula)
FAILED tests/contract/test_cost.py::test_decode_tok_s_known_answer - Assertio...
============================== 1 failed in 0.43s ===============================

### FAULT: F-I4 KV-cache path applies RoPE at offset 0 (positions wrong)
FAILED tests/engine/test_attention.py::test_kv_cache_equals_reference - Asser...
============================== 1 failed in 6.68s ===============================

### FAULT: F-I5 speculative verify accepts ALL draft tokens without checking target greedy
tests/engine/test_speculative.py .                                       [100%]
============================== 1 passed in 21.91s ==============================

### Do ANY tests reference the HTTP 503 refusal path?
tests/adversarial/test_byzantine_inputs.py:308:        assert status == 503, f"expected 503 refusal, got {status}: {body}"
tests/adversarial/test_byzantine_inputs.py:309:        assert body["error"]["type"] == "fitsproof_refused"

### FAULT: F-I6 server refusal path removed (never returns 503) — full suite:
FAILED tests/adversarial/test_byzantine_inputs.py::test_server_refuses_request_over_budget
================== 1 failed, 154 passed in 210.69s (0:03:30) ===================
```

**5 of 6 killed. F-I5 survived** — cause identified: the `draft` fixture in
`tests/engine/test_speculative.py` is *the same model as `target`* (its own
docstring: "Use the same model as draft (trivial case, but guarantees
correctness)"), so a "wrong draft token" cannot exist under the fixture and
accept-all verification is behaviourally equivalent there. The test cannot detect
the fault its docstring claims it detects. See ADV-05.

---

## 4. Findings

`id | severity | finding | evidence | status`

| id | severity | finding | evidence | status |
|---|---|---|---|---|
| ADV-01 | blocker | RESEARCH.md attaches "FlexGen Section 3.1 explicitly derives `throughput ∝ bandwidth / model_size`" to arXiv 2303.06865; full text contains no such derivation (15/15 "bandwidth" hits are transfer/network costs; §3 is Background, cost model is §4.3 offloading). Same mis-attribution in `cost.py:decode_tok_s` docstring ("Sheng et al. 2023 (FlexGen), Sec 3.1"). QUALITY-CONTRACT §3: a citation that does not support its claim is blocking. | §2b ar5iv grep output in this file | open — builder re-attributes to Williams et al. 2009 (source 1) or another source that actually derives the decode floor, in RESEARCH.md + cost.py; reviewer re-verifies |
| ADV-02 | blocker | RESEARCH.md claims "GPTQ uses per-channel scaling (each output channel has its own scale)" (arXiv 2210.17323); full text has 0 "per-channel" hits and describes FP16 scale **per group** (group-size 128/32). | §2b ar5iv grep output | open — builder corrects the claim to match the paper (per-group scale) or re-attributes the per-channel design; reviewer re-verifies |
| ADV-03 | blocker | Headline claim "fails the build on any … undocumented mode change" is vacuous: `verify_run` hardcodes `mode_changed_silently = False`, so `silent_mode_changes` in `StressResult` is always 0 and cannot fail. Demonstrated with a DEGRADED record → still False. The metric in `fitsproof stress` output is unfalsifiable. | §1 Attack 1c (code + REPL output) | open — builder implements detection (e.g. compare admitted plan config against the config actually executed) or removes the claim; a test must make the counter non-zero |
| ADV-04 | major | Stress-harness "measured peak" is the process-lifetime `ru_maxrss`, so all 25 configs report one identical number (min=median=max in README's own output); the per-configuration margin distribution is degenerate and cannot attribute memory to a config. At `--budget-gb 0.01` all 25 violations are startup RSS, not config-specific. | §1 Attacks 1a/1b/1d | open — builder measures per-config in a subprocess or scopes the high-water mark per config; or the README must state the number is a single process peak |
| ADV-05 | major | `test_speculative_equals_greedy` cannot detect its named fault ("verification accepts a wrong draft token"): draft fixture == target model, so accept-all passes. Fault injection F-I5 survived (1 passed). | §3 F-I5 | open — builder makes draft differ from target (e.g. reduced/different-seed draft) so the equality property is actually at risk, then re-runs F-I5 |
| ADV-06 | minor | README's "reproduce: `python scripts/calibration_demo.py`" does not reproduce the published numbers: 46.1% → 64.3% MAPE, gemm 37.19 → 13.78 GFLOPS (load-dependent probe). Also `CI (95%): [x, x]` with n_held_out=1 is not a confidence interval. | §1 Claim C3 output | open — mark the numbers as load-dependent with the load context, or publish a scripted load-independent check; drop/qualify the degenerate CI |
| ADV-07 | minor | Roofline ACM link (dl.acm.org) returns 403 to curl/automation; resolves via Crossref with matching title/venue/date. | §2a + CROSSREF output | limitation — publisher bot-protection; content verified via Crossref |
| ADV-08 | minor | README Limitations RSS bullet self-contradicts: it states peak RSS is "the high-water mark since process start" and, in the next sentence, that "allocations freed before the post-call sample may not be captured" — `ru_maxrss` does capture freed peaks; the sampling-based limitation applies to the two-sample wrapper, not to `ru_maxrss`. | `src/fitsproof/contract/verify.py:66-93`; README Limitations | open — builder rewords |

### Failed attacks (evidence for the defence)

- Refusal gate C2: could not falsify — exit codes, message and option tags all
  correct (§1 C2).
- Harness-has-teeth C1b: could not falsify — stress fails (rc=1) at a violating
  budget.
- Server 503 C3: could not falsify behaviourally, and the suite catches its
  removal (F-I6: 1 failed, 154 passed).
- README snippet freshness: could not falsify — snippets are extracted from
  README.md at test time, so drift fails the suite.
- Fault injections F-I1..F-I4: all killed by the exact test that names the fault.

### Gate status for this pass

- Artifact: `docs/ADVERSARIAL_REVIEW.md` (this file), written this pass.
- Open blockers at end of pass 1: ADV-01, ADV-02, ADV-03 (disposition: builder
  pass; reviewer re-verifies in `c1-p11-adversarial-2`).
- Repo left green: 155 passed, ruff clean, traceability OK (baseline rerun after
  all injections reverted).

PASS_c1-p10-adversarial-1 COMPLETE
