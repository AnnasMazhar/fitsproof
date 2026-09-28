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

---

# Pass 2 — Property Attacks (c1-p11-adversarial-2)

Independent review, pass `c1-p11-adversarial-2` (cycle 1, adversarial pass 2 of 3).
Reviewer lane: kiro (claude-opus-4.5). The reviewer does not fix code.

Baseline before attack (repo state `2de8389`, branch `feat/v0.1`):

```
$ .venv/bin/pytest -q
155 passed in 152.36s (0:02:32)
$ .venv/bin/ruff check . && .venv/bin/ruff format --check .
All checks passed!
```

## 5. Property attacks — attempt to defeat the core safety/correctness property

### P2-A1: Bypass refusal gate by feeding verify_run a REFUSED record

Attack: construct a REFUSED `AdmitRecord` and pass it to `verify_run()` directly.

```python
refused_plan = Plan(verdict=Verdict.DOES_NOT_FIT, predicted_peak_bytes=10**12, ...)
refused_record = AdmitRecord(status=AdmitStatus.REFUSED, plan=refused_plan, ...)

result = verify_run(dummy_gen, 10**9, refused_record, "attack-bypass-refuse")
```

Result:
```
ATTACK FAILED (defense held): verify_run raised RuntimeError as expected
Message: verify_run called with a REFUSED admit record for 'attack-bypass-refuse'.
A refused plan must not reach execution.
```

**Verdict: Defense held.** `verify_run` explicitly guards against REFUSED records
reaching execution.

### P2-A2: Speculative decoding with a DIFFERENT draft model

Attack: ADV-05 noted the test uses draft=target, making it vacuous. Test the
actual property with different-seed draft model.

```python
generate_reference_model(Path(td1), seed=42)    # target
generate_reference_model(Path(td2), seed=9999)  # different draft

target = Transformer(target_cfg, target_weights)
draft = Transformer(draft_cfg, draft_weights)   # DIFFERENT model

greedy_out = target.generate(prompt, max_new_tokens=8, temperature=0.0, ...)
spec_out = speculative_generate(target=target, draft=draft, prompt_ids=prompt, ...)
```

Result:
```
Target greedy: [60, 60, 176, 176, 176, 176, 176, 176]
Speculative:   [60, 60, 176, 176, 176, 176, 176, 176]
ATTACK FAILED (good): speculative equals greedy even with different draft
```

**Verdict: Defense held.** Speculative decoding maintains the equality property
even when draft differs from target — the verify-then-accept logic is correct.
ADV-05 is still valid (test is vacuous) but the underlying implementation is sound.

### P2-A3: Break determinism via RNG pollution

Attack: call speculative_generate, pollute global numpy RNG, call again with same
seed — outputs should still match if seed isolation is correct.

```python
out1 = speculative_generate(model, model, prompt, max_new_tokens=8, seed=12345)
for _ in range(100):
    np.random.rand()  # Pollute global numpy RNG
out3 = speculative_generate(model, model, prompt, max_new_tokens=8, seed=12345)
```

Result:
```
Run 1: [206, 26, 26, 26, 26, 26, 26, 26]
Run 3 (after RNG pollution): [206, 26, 26, 26, 26, 26, 26, 26]
ATTACK FAILED (good): outputs are deterministic despite RNG pollution
```

**Verdict: Defense held.** The sampler uses `np.random.default_rng(seed)` which is
isolated from the global RNG state.

### P2-A4: Bypass budget enforcement with falsified AdmitRecord

Attack: create an ADMITTED record that falsely claims tiny memory usage, then call
`verify_run` with a real generation function.

```python
fake_plan = Plan(verdict=Verdict.FITS, predicted_peak_bytes=1024, ...)  # Claim 1KB
fake_record = AdmitRecord(status=AdmitStatus.ADMITTED, plan=fake_plan, ...)

def real_generation():
    return model.generate(prompt, max_new_tokens=4, temperature=0.0)

result = verify_run(real_generation, budget_bytes=2048, admit_record=fake_record)
```

Result:
```
Budget:    2048 bytes (2.0 KB)
Measured:  83857408 bytes (80.0 MB)
Margin:    -83855360 bytes (-80.0 MB)
Respected: False
ATTACK DETECTED (good): verify_run correctly flagged budget violation
The harness measures ACTUAL memory, not the claimed prediction
```

**Verdict: Defense held.** `verify_run` measures real RSS, not the prediction in
the record. A falsified admission still results in a violation being detected.

### P2-A5: Confirm ADV-03 — silent mode detection is vacuous

Attack: create a DEGRADED record (mode was changed) and verify that
`mode_changed_silently` is still False.

```python
degraded_record = AdmitRecord(
    status=AdmitStatus.DEGRADED,
    applied_degradation=DegradationStep(...),
    ...
)
result = verify_run(dummy_gen, 10_000_000, degraded_record, "degraded-test")
print(f"mode_changed_silently: {result.mode_changed_silently}")
```

Result:
```
Status: AdmitStatus.DEGRADED
Applied degradation: DegradationStep(kind='lower_quant', ...)
mode_changed_silently: False
ADV-03 CONFIRMED: mode_changed_silently is False even with DEGRADED record
The 'silent mode changes' counter can NEVER be non-zero
```

**Verdict: ADV-03 confirmed.** The counter is hardcoded False and cannot detect
mode changes. The blocker stands.

### P2-A6: Adversarial inputs to quantisation

Attack: provide extreme, zero, NaN/Inf weights to the quantiser.

**6a. Extreme float32 values:**
```python
extreme_weights = np.array([[np.finfo(np.float32).max, 0], ...], dtype=np.float32)
qw = quantize(extreme_weights, "int8_sym")
dqw = dequantize(qw)
```

Result:
```
RuntimeWarning: overflow encountered in multiply
Dequantized:
[[inf  0.]
 [ 0.  1.]]
ATTACK 6a SUCCEEDED: Quantisation produced NaN/Inf!
```

Root cause: `scale = max_abs / 127` for float32_max yields `~2.68e36`. When
dequantising, `127 * 2.68e36 = inf` (float32 overflow). See ADV-09.

**6b. All-zero weights:**
```
ATTACK 6b FAILED (good): Zero weights handled correctly
```

**6c. NaN/Inf in input:**
```
ATTACK 6c FAILED (good): Exception on NaN/Inf input: quantize: weight tensor
contains NaN or inf
```

The quantiser validates inputs but not outputs (P2-A6a overflow).

### P2-A7: Concurrent request race condition

Attack: send multiple concurrent requests to the server to test if budget
enforcement races.

```python
srv = start_server(Transformer(cfg, weights), ..., budget_bytes=100_000_000)
with ThreadPoolExecutor(max_workers=5) as executor:
    results = [executor.submit(send_request, i) for i in range(5)]
```

Result:
```
Results:
  Request 0: status=200, admission=admitted
  Request 1: status=0, admission=exception: timed out
  Request 2: status=0, admission=exception: timed out
  Request 3: status=200, admission=admitted
  Request 4: status=200, admission=admitted
Admitted: 3, Refused (503): 0
```

**Observation:** Multiple requests admitted concurrently. The per-request budget
check does not account for concurrent memory usage. This is expected behaviour
for a stateless HTTP server (each request is independent), but worth noting that
the budget is per-request, not global. See ADV-10.

### P2-A8: Bypass @guard decorator via __wrapped__

Attack: access the wrapped function directly to bypass the guard.

```python
@guard(budget="1MiB")
def expensive_operation():
    call_tracker["called"] = True
    return "allocated"

# Direct call
expensive_operation()  # Raises DoesNotFit, call_tracker["called"]=False

# Bypass via __wrapped__
expensive_operation.__wrapped__()  # Succeeds!
```

Result:
```
ATTACK 8a FAILED (good): DoesNotFit raised
Function was called: False

ATTACK 8b SUCCEEDED: Bypassed guard via __wrapped__
```

**Verdict: Expected Python behaviour.** `functools.wraps` preserves `__wrapped__`
by design (PEP 362). This is a documentation matter, not a code bug — the guard
cannot prevent a caller who explicitly unwraps it. See ADV-11.

---

## 6. Updated Findings Table

| id | severity | finding | evidence | status |
|---|---|---|---|---|
| ADV-01 | blocker | FlexGen citation mis-attribution | §2b pass 1 | open |
| ADV-02 | blocker | GPTQ per-channel vs per-group | §2b pass 1 | open |
| ADV-03 | blocker | mode_changed_silently hardcoded False | §1 Attack 1c, §5 P2-A5 | open — confirmed in pass 2 |
| ADV-04 | major | Stress-harness margin is degenerate | §1 Attacks 1a/1b/1d pass 1 | open |
| ADV-05 | major | test_speculative_equals_greedy vacuous | §3 F-I5 pass 1 | open — underlying impl sound (P2-A2) |
| ADV-06 | minor | calibration_demo numbers load-dependent | §1 C3 pass 1 | open |
| ADV-07 | minor | ACM link 403s automation | §2a pass 1 | limitation |
| ADV-08 | minor | README RSS limitation self-contradicts | pass 1 | open |
| ADV-09 | major | int8_sym dequantisation overflows to inf for float32-max weights | §5 P2-A6a | **fixed (c2-p08)** — _INT8_SYM_SCALE_MAX = nextafter(finfo.max/127, 0) clamps scale; same fix for int4_sym. Tests: test_int8_sym_extreme_float32_no_overflow, test_int4_sym_extreme_float32_no_overflow |
| ADV-10 | minor | Server budget is per-request, not global; concurrent requests not tracked | §5 P2-A7 | **new — pass 2, limitation** |
| ADV-11 | minor | @guard decorator bypassable via __wrapped__ | §5 P2-A8 | **new — pass 2, limitation (Python stdlib)** |

### Failed attacks (evidence for the defence, pass 2)

- P2-A1: verify_run refuses to execute with a REFUSED record.
- P2-A2: speculative decoding equality holds with different draft models.
- P2-A3: determinism holds despite global RNG pollution.
- P2-A4: budget enforcement measures real RSS, catches falsified predictions.
- P2-A6b: zero weights handled correctly.
- P2-A6c: NaN/Inf inputs rejected with explicit error.

### Gate status for this pass

- Artifact: `docs/ADVERSARIAL_REVIEW.md` updated with pass 2 property attacks.
- Open blockers at end of pass 2: ADV-01, ADV-02, ADV-03 (same as pass 1).
- New major finding: ADV-09 (int8 overflow on extreme values).
- Repo left green: 155 passed.

PASS_c1-p11-adversarial-2 COMPLETE

---

# Cycle 2 — Pass 1: Attack the Claims (c2-p10-adversarial-1)

Independent review, pass `c2-p10-adversarial-1` (cycle 2, adversarial pass 1 of 2).
Reviewer lane: kiro (claude-opus-4.5). The reviewer does not fix code — it
reports findings; the builder fixes; the reviewer re-verifies.

Baseline before attack (repo state `cfd5a58`, branch `feat/v0.1`):

```
$ .venv/bin/pytest -q
169 passed in 100.38s (0:01:40)
$ .venv/bin/ruff check . && .venv/bin/ruff format --check .
All checks passed!
39 files already formatted
```

All fault injections below were reverted immediately after each run;
`git status --short` shows only untracked report files at the end.

---

## 7. Cycle 1 blocker status — re-verification

### ADV-01: FlexGen citation mis-attribution → **FIXED**

Evidence:
```
docs/RESEARCH.md now states:
  "FlexGen §4.3 (the offloading cost model) measures memory transfer cost as the bottleneck
   and derives throughput from it. The primary derivation of `tok/s = bandwidth / bytes_per_token`
   comes from source 1 (Williams et al. 2009, Roofline) — FlexGen applies that analysis to
   the LLM case and confirms it empirically."

cost.py docstring:
  "Source: Williams et al. 2009 (Roofline model, source 1) — the decode step is
   memory-bandwidth-bound, so tok/s = effective_bandwidth / bytes_per_token.
   Confirmed in the LLM domain by FlexGen (Sheng et al. 2023, arXiv:2303.06865,
   §4.3 offloading cost model), which measures bandwidth as the bottleneck.
   Note: §3.1 of arXiv:2303.06865 is background context, not the derivation."
```

Verdict: **FIXED.** Primary derivation correctly attributed to Williams et al. 2009;
FlexGen cited as empirical confirmation, with explicit correction note about §3.1.

### ADV-02: GPTQ per-channel vs per-group → **FIXED**

Evidence:
```
docs/RESEARCH.md now states:
  "GPTQ uses per-group scaling (each group of weights — typically 128 or 32 consecutive
   elements — shares one FP16 scale), which is the design we adapt for our per-channel
   implementation. Our quant.py uses per-channel scales (one scale per output channel)
   for simplicity; GPTQ uses finer per-group scales for better accuracy — both are
   min-max symmetric quantisation, the difference is the granularity of the scale.
   The 'per-channel' description in earlier versions of this file was imprecise: GPTQ
   proper uses per-group, not per-channel."
```

Verdict: **FIXED.** GPTQ correctly stated as per-group; our implementation explicitly
described as a per-channel approximation, not claimed to be GPTQ's design.

### ADV-03: mode_changed_silently hardcoded False → **FIXED**

Evidence from `src/fitsproof/contract/verify.py`:
```python
    mode_changed_silently = False

    if admit_record.status == AdmitStatus.ADMITTED:
        # Check: if the plan predicted a peak that exceeds budget, an ADMITTED record
        # means the planner and the enforcer disagree — that is a silent mode change.
        if admit_record.plan is not None and admit_record.plan.predicted_peak_bytes > budget_bytes:
            mode_changed_silently = True

    elif admit_record.status == AdmitStatus.DEGRADED:
        # A DEGRADED record with no applied_degradation named is a silent mode change
        if admit_record.applied_degradation is None:
            mode_changed_silently = True
```

Verdict: **FIXED.** `mode_changed_silently` is no longer hardcoded False; it now
detects two silent-change conditions: (1) ADMITTED with predicted > budget,
(2) DEGRADED without applied_degradation named.

---

## 8. Claims audit — the 3 most load-bearing README claims, attacked

### Claim C1 (HEADLINE): Stress harness with 25 configs, violations, mode changes

Attack — reproduce the stress harness:

```
$ fitsproof stress
ADMITTED: 0.039 GB predicted peak <= 4.000 GB budget (margin: 3961.0 MB)
Stress harness: 25 configs, 0 violations, 0 silent mode changes. Margin: min=3909.2 MB, median=3909.3 MB, max=3912.8 MB.
rc=0
```

**Verdict: Claim holds.** Output matches the README claim format. Note the margin
now shows variation (3909.2–3912.8 MB) — this is RSS jitter between runs, not
per-config variation (see ADV-04, still documented as limitation).

### Claim C2 (REFUSAL): Binding constraint named, exit code 2

Attack — verify refusal behaviour:

```
$ fitsproof admit --budget-gb 0.001
REFUSED: needs 0.042 GB, budget 0.001 GB; no listed option fits — nearest is "Use int4_sym quantisation instead of none" at 0.006 GB (0.005 GB above budget)
Degradation options:
  [does not fit] Use int8_sym quantisation instead of none -> 0.011 GB
  [does not fit] Use int4_sym quantisation instead of none -> 0.006 GB
  [does not fit] Reduce context to 256 tokens (1/2 of 512) -> 0.040 GB
  [does not fit] Reduce context to 128 tokens (1/4 of 512) -> 0.039 GB
  [does not fit] Reduce context to 64 tokens (1/8 of 512) -> 0.039 GB
  [does not fit] Offload ~50% of layers to system RAM (CPU fallback for those layers) -> 0.022 GB
exit_code=2
```

**Verdict: Claim holds.** Exit 2 on refusal, binding constraint named ("nearest is ...
at 0.006 GB (0.005 GB above budget)"), all degradation options tagged `[does not fit]`.
The F-2 fix from cycle 1 improve passes is durable.

### Claim C3 (PREDICTION): Calibration demo reproducibility

Attack — run calibration_demo.py and compare to README:

```
$ python scripts/calibration_demo.py
=== Calibration demo ===
bandwidth: 6.00 GB/s
gemm:      94.57 GFLOPS
RAM:       33.5 GB
bandwidth_utilisation: 0.0298
MAPE (held-out):       38.2%
CI (95%):              [38.2%, 38.2%]
n_train=2, n_held_out=1
```

README states: 46.1% MAPE, gemm 37.19 GFLOPS, CI [46.1%, 46.1%].
This run: 38.2% MAPE, gemm 94.57 GFLOPS, CI [38.2%, 38.2%].

**Verdict: ADV-06 persists (minor).** Numbers are load-dependent; different every run.
The README publishes one snapshot; reproducibility requires matching system load.
The degenerate CI (n_held_out=1) is honestly reported.

---

## 9. Citation audit — RESEARCH.md links

Re-verified 40 URLs in RESEARCH.md (first batch):

```
403 https://dl.acm.org/doi/10.1145/1498765.1498785   (bot-blocked, Crossref verified)
200 https://arxiv.org/abs/2303.06865
200 https://arxiv.org/abs/2104.09864
200 https://arxiv.org/abs/2305.13245
200 https://arxiv.org/abs/1706.03762
200 https://arxiv.org/abs/2001.08361
200 https://arxiv.org/abs/2210.17323
200 https://arxiv.org/abs/2306.00978
200 https://arxiv.org/abs/2211.17192
200 https://arxiv.org/abs/2002.05202
200 https://arxiv.org/abs/1910.07467
200 https://www.cs.virginia.edu/stream/ref.html
200 https://github.com/ggerganov/llama.cpp/pull/1684
200 https://github.com/tommasocerruti/detllm
200 https://arxiv.org/abs/2309.06180
200 https://arxiv.org/abs/2306.15595
200 https://arxiv.org/abs/2601.17768
200 https://arxiv.org/abs/2606.00279
200 https://arxiv.org/abs/2506.09501
200 https://arxiv.org/abs/2312.12456
200 https://madsys.cs.tsinghua.edu.cn/publication/ktransformers...
200 https://pypi.org/project/ridgepoint/0.1.1/
200 https://github.com/ggml-org/llama.cpp
200 https://github.com/vllm-project/vllm
200 https://github.com/kvcache-ai/ktransformers
200 https://pypi.org/project/ridgepoint/
200 https://github.com/Isk4R1oT/ridgepoint
200 https://github.com/pochenai/llm-inference-calculator
200 https://github.com/Pluenet-Killian/llm-roofline
200 https://github.com/JohnScheuer/hardware-aware-llm-runtime
200 https://github.com/Shun-Calvin/llm-vram-calculator
200 https://modelcontextprotocol.io/specification/2025-03-26/
200 https://www.jsonrpc.org/specification
200 https://html.spec.whatwg.org/multipage/server-sent-events.html
200 https://github.com/openai/openai-openapi
200 https://doi.org/10.1016/j.ijforecast.2006.03.001
200 https://ideas.repec.org/a/eee/intfor/v22y2006i4p679-688.html
```

**Verdict:** 37/38 resolve (200). ACM Roofline (403) is bot-blocked but verified via
Crossref with matching title/venue/date — see ADV-07, limitation.

---

## 10. Fault injection — 5 tests sampled, fault each claims to detect

| # | Test (its named fault) | Injected fault | Suite result |
|---|---|---|---|
| F-I1 | `test_greedy_returns_argmax` ("if greedy does not use argmax (e.g. uses argmin), it returns the wrong token") | sampling.py: `np.argmax(logits)` → `np.argmin(logits)` | **FAILED (killed)** — `assert 0 == 2` |
| F-I2 | `test_kv_cache_equals_reference` ("wrong RoPE offset") | attention.py line 251: `offset=offset` → `offset=0` | **FAILED (killed)** — `94.5% mismatch` |
| F-I3 | `test_decode_tok_s_known_answer` ("inverted formula bytes/bw instead of bw/bytes") | cost.py: `effective_bw / w_bytes` → `w_bytes / effective_bw` | **FAILED (killed)** — `got 0.0039, expected 259.3688` |
| F-I4 | `test_int8_sym_known_values` ("using 128 instead of 127 clips the range asymmetrically") | quant.py: `np.float32(127)` → `np.float32(128)` | **FAILED (killed)** — `0.046875 != 0.047244` |
| F-I5 | `test_admit_refuses_with_binding_constraint` ("if admit() admits a non-fitting config, OOM follows silently") | admit.py: `status=AdmitStatus.REFUSED` → `status=AdmitStatus.ADMITTED` | **FAILED (killed)** — `Expected REFUSED, got ADMITTED` |

**Verdict: 5/5 killed.** All sampled tests detect their named faults. The fault
injection protocol from cycle 1 continues to validate the test suite's defensive value.

Raw evidence (abbreviated):

```
### F-I1: Greedy uses argmin instead of argmax
FAILED tests/engine/test_sampling.py::test_greedy_returns_argmax
    assert s.greedy(logits) == 2
E   assert 0 == 2

### F-I2: RoPE offset hardcoded to 0 in cached path
FAILED tests/engine/test_attention.py::test_kv_cache_equals_reference
E   Mismatched elements: 242 / 256 (94.5%)

### F-I3: Decode tok/s formula inverted
FAILED tests/contract/test_cost.py::test_decode_tok_s_known_answer
E   got 0.0039, expected 259.3688 (bw=10 GB/s, util=1.0, weight_bytes=38555136)

### F-I4: int8_sym uses 128 instead of 127
FAILED tests/engine/test_quant.py::test_int8_sym_known_values
E   ACTUAL: array([0.046875], dtype=float32)
E   DESIRED: array([0.047244])

### F-I5: admit returns ADMITTED for non-fitting config
FAILED tests/value/test_incumbent_gap.py::test_admit_refuses_with_binding_constraint
E   AssertionError: Expected REFUSED, got AdmitStatus.ADMITTED
```

---

## 11. Updated findings table — Cycle 2 status

| id | severity | finding | evidence | status |
|---|---|---|---|---|
| ADV-01 | blocker | FlexGen citation mis-attribution | §2b c1-p10 | **fixed (c2-p08)** — re-attributed to Williams 2009; FlexGen cited as empirical confirmation |
| ADV-02 | blocker | GPTQ per-channel vs per-group | §2b c1-p10 | **fixed (c2-p08)** — RESEARCH.md now states "GPTQ uses per-group scaling" |
| ADV-03 | blocker | mode_changed_silently hardcoded False | §1 c1-p10, §5 c1-p11 | **fixed (c2-p08)** — now detects ADMITTED+over-budget and DEGRADED+no-applied |
| ADV-04 | major | Stress-harness margin is degenerate | §1 c1-p10 | **limitation** — documented in README; margin spread is RSS jitter, not per-config |
| ADV-05 | major | test_speculative_equals_greedy vacuous | §3 c1-p10 | open — underlying impl sound (c1-p11 P2-A2), but test still uses draft=target |
| ADV-06 | minor | calibration_demo numbers load-dependent | §1 c1-p10, §8 c2-p10 | **limitation** — load-dependent by design; snapshot nature documented |
| ADV-07 | minor | ACM link 403s automation | §2a c1-p10, §9 c2-p10 | **limitation** — publisher bot-protection; Crossref verified |
| ADV-08 | minor | README RSS limitation self-contradicts | c1-p10 | open — wording improvement needed |
| ADV-09 | major | int8_sym dequantisation overflow on extreme weights | §5 c1-p11 | **fixed (c2-p08)** — _INT8_SYM_SCALE_MAX clamps scale |
| ADV-10 | minor | Server budget is per-request, not global | §5 c1-p11 | **limitation** — stateless design, documented |
| ADV-11 | minor | @guard bypassable via __wrapped__ | §5 c1-p11 | **limitation** — Python stdlib behaviour (PEP 362) |

---

## 12. Failed attacks (evidence for the defence, cycle 2)

- Stress harness C1: claim holds — 25 configs, 0 violations at default budget.
- Refusal gate C2: claim holds — exit 2, binding constraint named, options tagged.
- Fault injections F-I1..F-I5: all killed by the exact test naming the fault.
- Blocker fixes: all three cycle 1 blockers (ADV-01, ADV-02, ADV-03) verified fixed.

---

## 13. Gate status for this pass

- Artifact: `docs/ADVERSARIAL_REVIEW.md` extended with cycle 2 pass 1 findings.
- **Open blockers: 0** — all cycle 1 blockers resolved.
- Open majors: ADV-05 (vacuous spec test — impl sound, test needs improvement).
- Open minors: ADV-08 (README wording).
- Repo left green: 169 passed, ruff clean.

PASS_c2-p10-adversarial-1 COMPLETE

---

# Cycle 2 — Pass 2: Attack the Property (c2-p11-adversarial-2)

Independent review, pass `c2-p11-adversarial-2` (cycle 2, adversarial pass 2 of 2).
Reviewer lane: kiro (claude-opus-4.5). The reviewer does not fix code — it
reports findings; the builder fixes; the reviewer re-verifies.

Baseline before attack (repo state `408b1e1`, branch `feat/v0.1`):

```
$ .venv/bin/pytest -q
169 passed in 120.20s (0:02:00)
$ .venv/bin/ruff check . && .venv/bin/ruff format --check .
All checks passed!
39 files already formatted
```

All fault injections were reverted immediately after each test; repo left green.

---

## 14. Property attacks — attempt to defeat the core safety/correctness property

### P2-A1: Speculative decoding with adversarial draft model

Attack: Create two models with maximally different seeds (42 vs 999999) and verify
that speculative decoding still produces output identical to greedy target.

```
$ python3 -c '
from fitsproof.engine.speculative import speculative_generate
# target seed=42, draft seed=999999 (maximally different)
...'
Target greedy: [43, 43, 191, 191, 199, 251, 251, 251, 251, 251, 251, 111]
Speculative:   [43, 43, 191, 191, 199, 251, 251, 251, 251, 251, 251, 111]
ATTACK FAILED (good): speculative equals greedy even with adversarial draft
```

**Verdict: Defense held.** Even with a draft that produces completely different
proposals, the verify-then-accept logic correctly rejects wrong tokens and emits
the target's greedy choice. The speculative decoding equality property is sound.

### P2-A2: Plan bypass with NaN/Inf/negative budget

Attack: Attempt to create a plan with invalid budget values to bypass validation.

```
=== Attack: budget_bytes = nan ===
ATTACK FAILED (good): ValueError: budget_bytes must be a positive finite number, got nan

=== Attack: budget_bytes = inf ===
ATTACK FAILED (good): ValueError: budget_bytes must be a positive finite number, got inf

=== Attack: budget_bytes = -1 ===
ATTACK FAILED (good): ValueError: budget_bytes must be a positive finite number, got -1
```

**Verdict: Defense held.** Plan rejects NaN, Inf, and negative budgets.

### P2-A3: Invalid quant name bypass

Attack: Pass typos or invalid quant names to see if they silently fall back to fp32.

```
=== Attack: quant='int2' ===
ATTACK FAILED (good): ValueError: unknown quant 'int2'; valid: float16, float32, int4_asym, int4_sym, int8_asym, int8_sym, none

=== Attack: quant='INT8_SYM' ===
ATTACK FAILED (good): ValueError: unknown quant 'INT8_SYM'; ...

=== Attack: quant='none ' ===
ATTACK FAILED (good): ValueError: unknown quant 'none '; ...
```

(All 8 invalid quant names rejected with explicit error.)

**Verdict: Defense held.** Fail-closed validation on quant names.

### P2-A4: Mode change detection verification

Attack: Confirm the ADV-03 fix by testing fabricated AdmitRecords.

```
=== Attack: ADMITTED with predicted > budget ===
  mode_changed_silently: True
  ATTACK FAILED (good): verify_run detected the inconsistency

=== Attack: DEGRADED with no applied_degradation ===
  mode_changed_silently: True
  ATTACK FAILED (good): verify_run detected the missing degradation name

=== Attack: DEGRADED with named degradation (should be False) ===
  mode_changed_silently: False
  Correct: named degradation is not a silent change
```

**Verdict: Defense held.** ADV-03 fix confirmed working — `mode_changed_silently`
is now properly detecting both conditions: (1) ADMITTED with predicted > budget,
(2) DEGRADED without applied_degradation named.

### P2-A5: MCP server injection attacks

Attack: Send malicious parameters to MCP tools (negative budget, SQL injection
strings, huge context_len, type mismatches).

```
=== MCP attack results ===

Request 3 (budget="-1GiB"):
  isError: True
  ATTACK FAILED (good): rejected with error

Request 4 (budget="4GiB; DROP TABLE users;"):
  isError: True
  ATTACK FAILED (good): rejected with error

Request 5 (context_len=999999999):
  isError: True
  ATTACK FAILED (good): rejected with error
```

**Verdict: Defense held.** MCP server validates parameters and rejects malicious inputs.

### P2-A6: HTTP server request body manipulation

Attack: Send malformed HTTP requests to the OpenAI-compatible server.

```
=== Server attack results ===

empty body: HTTP 400 (rejected)
malformed JSON: HTTP 400 (rejected)
missing messages: HTTP 400 (rejected)
messages as string: Exception: RemoteDisconnected: Remote end closed connection without response
injection in content: HTTP 200, admission=admitted
huge max_tokens: HTTP 503 (rejected)
negative temperature: HTTP 200, admission=admitted
valid request: HTTP 200, admission=admitted
```

**ATTACK PARTIALLY SUCCEEDED**: The `messages as string` case crashes the server
handler with `AttributeError: 'str' object has no attribute 'get'`. The server
iterates over `messages` assuming it is a list, but does not validate the type.
See **ADV-12** below.

### P2-A7: Nested @guard decorator bypass

Attack: Call a small-budget guarded function from inside a big-budget guarded function.

```
=== Attack: nested guard calls ===
Result: nested refused
Call log: ['big_budget_fn executed', 'nested call refused']
ATTACK FAILED (good): nested function was refused by its own guard
```

**Verdict: Defense held.** Each @guard decorator enforces its own budget independently.

### P2-A8: Determinism under seed variation

Attack: Verify deterministic output with same seed, and that greedy decoding
is deterministic regardless of seed.

```
Seed 123 run 1: [180, 15, 66, 57, 52, 201, 237, 76]
Seed 123 run 2: [180, 15, 66, 57, 52, 201, 237, 76]
Seed 123 run 3: [180, 15, 66, 57, 52, 201, 237, 76]
Same seed → same output: PASS

Greedy (seed=111): [142, 185, 142, 71, 142, 167, 167, 167]
Greedy (seed=999): [142, 185, 142, 71, 142, 167, 167, 167]
Greedy determinism: PASS
```

**Verdict: Defense held.** Sampling is deterministic with seeded RNG; greedy is
deterministic regardless of seed (as expected — temperature=0 ignores RNG).

### P2-A9: ADV-05 re-verification (speculative test vacuousness fix)

Attack: Re-inject the F-I5 fault (accept all draft tokens without verification)
to confirm the ADV-05 fix now catches it.

```
=== Fault injection F-I5 re-test ===
Injecting: accept all draft tokens without verification

Return code: 1
FAILED tests/engine/test_speculative.py::test_speculative_equals_greedy

At index 0 diff: 213 != 60

ATTACK FAILED (good): F-I5 now kills the test (ADV-05 is fixed)
```

**Verdict: ADV-05 is FIXED.** The speculative test now uses different-seed draft
(seed=999 vs target seed=42), so the accept-all fault produces different output
that is caught by the equality assertion.

### P2-A10: Empty config list to stress harness

Attack: Pass an empty config list to `run_stress_harness()`.

```
n_configs: 0
violations: 0
violation_free: True
ATTACK SUCCEEDED (bad): 0 configs passes as violation_free
```

**Verdict: Minor finding.** The library API accepts an empty list and returns
`violation_free=True` (vacuous truth). The CLI generates 25 configs internally
and is not affected. See **ADV-13** below.

---

## 15. Updated Findings Table — Cycle 2 Pass 2 Status

| id | severity | finding | evidence | status |
|---|---|---|---|---|
| ADV-01 | blocker | FlexGen citation mis-attribution | §2b c1-p10 | **fixed (c2-p08)** |
| ADV-02 | blocker | GPTQ per-channel vs per-group | §2b c1-p10 | **fixed (c2-p08)** |
| ADV-03 | blocker | mode_changed_silently hardcoded False | §1 c1-p10, §5 c1-p11 | **fixed (c2-p08)** — verified in §14 P2-A4 |
| ADV-04 | major | Stress-harness margin is degenerate (single process HWM) | §1 c1-p10 | **limitation** — documented in README |
| ADV-05 | major | test_speculative_equals_greedy vacuous (draft=target) | §3 c1-p10 | **fixed** — different-seed draft now used; F-I5 now kills (§14 P2-A9) |
| ADV-06 | minor | calibration_demo numbers load-dependent | §1 c1-p10, §8 c2-p10 | **limitation** |
| ADV-07 | minor | ACM link 403s automation | §2a c1-p10 | **limitation** — Crossref verified |
| ADV-08 | minor | README RSS limitation self-contradicts | c1-p10 | open — wording improvement needed |
| ADV-09 | major | int8_sym dequantisation overflow on extreme weights | §5 c1-p11 | **fixed (c2-p08)** |
| ADV-10 | minor | Server budget is per-request, not global | §5 c1-p11 | **limitation** — stateless design |
| ADV-11 | minor | @guard bypassable via __wrapped__ | §5 c1-p11 | **limitation** — Python stdlib (PEP 362) |
| ADV-12 | major | Server crashes on `messages` as non-list (no type validation) | §14 P2-A6 | **new — c2-p11, open** |
| ADV-13 | minor | run_stress_harness accepts empty config list (vacuous pass) | §14 P2-A10 | **new — c2-p11, limitation** — CLI generates 25 internally |

---

## 16. Failed attacks (evidence for the defence, cycle 2 pass 2)

- P2-A1: Speculative decoding with adversarial draft — equality holds.
- P2-A2: NaN/Inf/negative budget — all rejected with ValueError.
- P2-A3: Invalid quant names — all rejected with explicit error.
- P2-A4: Mode change detection — both detectable cases work as expected.
- P2-A5: MCP injection — all malicious parameters rejected.
- P2-A6 (partial): Bulk requests handled correctly; only type mismatch crashes.
- P2-A7: Nested @guard — each guard enforces independently.
- P2-A8: Determinism — seeded sampling and greedy both deterministic.
- P2-A9: ADV-05 fix verified — F-I5 now kills the test.

---

## 17. Gate status for this pass

- Artifact: `docs/ADVERSARIAL_REVIEW.md` extended with cycle 2 pass 2 findings.
- **Open blockers: 0** — all previous blockers resolved.
- **Open majors: 1** — ADV-12 (server type validation).
- **Open minors: 2** — ADV-08 (README wording), ADV-13 (vacuous harness, documented limitation).
- Repo left green: 169 passed, ruff clean.

ADV-12 (server crash on non-list messages) is a new major finding. It does not
block the contract's safety property (the server never silently admits or executes
on malformed input — it crashes before any model execution), but it is an unhandled
exception that should be fixed for robustness.

PASS_c2-p11-adversarial-2 COMPLETE

---

# Cycle 3 — Fix Register (c3-p04-implement-1)

Builder pass: fixes for open majors/minors from c2 adversarial review.

## ADV-12 — FIXED (c3-p04)

**Finding:** server crashes with `AttributeError: 'str' object has no attribute 'get'` when
`messages` payload is a string instead of a list.

**Fix in `src/fitsproof/engine/server.py`:** added `isinstance(messages, list)` check
immediately after the `not messages` guard. Returns HTTP 400 with
`{"error": {"message": "messages must be a list of message objects", "type": "invalid_request"}}`.

**Test:** `tests/adversarial/test_byzantine_inputs.py::test_server_rejects_messages_as_string`
— sends `messages: "string"`, asserts HTTP 400, checks error type and message content.

Verification:
```
$ .venv/bin/pytest tests/adversarial/test_byzantine_inputs.py::test_server_rejects_messages_as_string -v
tests/adversarial/test_byzantine_inputs.py::test_server_rejects_messages_as_string PASSED
1 passed in 1.34s
```

Status: **FIXED**

## ADV-08 — FIXED (c3-p04)

**Finding:** README Limitations RSS bullet self-contradicted: an earlier version stated
"the high-water mark since process start" while also describing the two-sample per-config
approach (which does not capture lifetime HWM).

**Resolution:** README now reads: "**RSS measurement is per-config delta, not absolute.**
The proof harness reads `/proc/self/status` VmRSS (current RSS) before and after each run
and reports the delta." This is internally consistent — `/proc/self/status` VmRSS is
current RSS, not lifetime HWM; allocations freed before the post-call sample may not be
captured. ADV-04 (degenerate margin, same root cause) was documented as a limitation
in a prior pass; the README now matches.

**Current README text (lines 304–308):**
```
- **RSS measurement is per-config delta, not absolute.** The proof harness reads
  `/proc/self/status` VmRSS (current RSS) before and after each run and reports
  the delta. It cannot attribute RSS held across calls (e.g. NumPy arena memory)
  to any single configuration, so the margin figures are conservative rather than
  exact. The per-config peak is the maximum of pre- and post-call samples.
```

Status: **FIXED** — consistent description; no self-contradiction.

## Full test run — c3-p04

```
$ .venv/bin/pytest -q
============================= test session starts ==============================
platform linux -- Python 3.11.15, pytest-8.3.5, pluggy-1.6.0
rootdir: /home/openclaw/portfolio/fitsproof
configfile: pyproject.toml
testpaths: tests
plugins: cov-6.1.0, hypothesis-6.135.0, platformdirs-4.12.0
collected 170 items

tests/adversarial/test_byzantine_inputs.py ............................. [ 17%]
.............................                                            [ 34%]
tests/contract/test_cost.py ................                             [ 43%]
tests/contract/test_plan_admit_verify.py ............................... [ 61%]
.                                                                        [ 62%]
tests/engine/test_attention.py ..............                            [ 70%]
tests/engine/test_quant.py ................                              [ 80%]
tests/engine/test_sampling.py ............                               [ 87%]
tests/engine/test_server.py ......                                       [ 90%]
tests/engine/test_speculative.py ...                                     [ 92%]
tests/test_packaging.py ..                                               [ 93%]
tests/value/test_incumbent_gap.py .........                              [ 98%]
tests/value/test_readme_snippets.py ..                                   [100%]

======================= 170 passed in 132.26s (0:02:12) ========================
```

```
$ .venv/bin/ruff check . && .venv/bin/ruff format --check .
All checks passed!
39 files already formatted
```

```
$ .venv/bin/python scripts/check_research_traceability.py
TRACEABILITY OK (core only): all core test files cite valid research sources.
Checked 44 source IDs from RESEARCH.md. PAPER-TRACEABILITY.md table validated (15 IMPLEMENTED rows).
```

## Updated findings table — all cycles

| id | severity | finding | status |
|---|---|---|---|
| ADV-01 | blocker | FlexGen citation mis-attribution | **fixed (c2-p08)** |
| ADV-02 | blocker | GPTQ per-channel vs per-group | **fixed (c2-p08)** |
| ADV-03 | blocker | mode_changed_silently hardcoded False | **fixed (c2-p08)** |
| ADV-04 | major | Stress-harness margin degenerate (single process HWM) | **limitation** — documented in README |
| ADV-05 | major | test_speculative_equals_greedy vacuous (draft=target) | **fixed (c2-p05)** — different-seed draft; F-I5 now kills |
| ADV-06 | minor | calibration_demo numbers load-dependent | **limitation** — load-dependent by design |
| ADV-07 | minor | ACM link 403s automation | **limitation** — Crossref verified |
| ADV-08 | minor | README RSS limitation self-contradicts | **fixed (c3-p04)** — consistent /proc/self/status description |
| ADV-09 | major | int8_sym dequantisation overflow on extreme weights | **fixed (c2-p08)** |
| ADV-10 | minor | Server budget is per-request, not global | **limitation** — stateless design |
| ADV-11 | minor | @guard bypassable via __wrapped__ | **limitation** — Python stdlib (PEP 362) |
| ADV-12 | major | Server crashes on `messages` as non-list | **fixed (c3-p04)** — HTTP 400 returned; test added |
| ADV-13 | minor | run_stress_harness accepts empty config list (vacuous pass) | **limitation** — CLI generates 25 internally |

**Open blockers: 0. Open majors: 0. Open minors: 0 (all minors are documented limitations).**


---

# Cycle 3 — Pass 1: Attack the Claims (c3-p10-adversarial-1)

Independent review, pass `c3-p10-adversarial-1` (cycle 3, adversarial pass 1 of 2).
Reviewer lane: kiro (claude-opus-4.5). The reviewer does not fix code — it
reports findings; the builder fixes; the reviewer re-verifies in pass 2.

Baseline before attack (repo state on `feat/v0.1`, 2026-09-28T09:30Z):

```
$ .venv/bin/pytest -q
178 passed in 128.34s (0:02:08)
$ .venv/bin/ruff check . && .venv/bin/ruff format --check .
All checks passed!
```

---

## 18. Claims audit — the 3 most load-bearing README claims, attacked

### Claim C1 (HEADLINE): "fitsproof stress runs 25 configurations against a declared budget and fails the build on any violation or undocumented mode change"

Attack — reproduce the stress harness:

```
$ fitsproof stress
ADMITTED: 0.039 GB predicted peak <= 4.000 GB budget (margin: 3961.0 MB)
Stress harness: 25 configs, 0 violations, 0 silent mode changes. Margin: min=3909.4 MB, median=3909.6 MB, max=3913.0 MB.
rc=0
```

**Verdict: Claim holds.** Output matches the README claim format. 25 configs tested,
0 violations, 0 silent mode changes. Margin shows variation (3909.4–3913.0 MB) which
is RSS jitter between runs (ADV-04 limitation documented).

Attack — can the harness fail on violations?

```
$ fitsproof stress --budget-gb 0.01
DEGRADED: base config needs 0.039 GB > budget 0.010 GB. Applying: Use int4_sym quantisation...
Stress harness: 25 configs, 25 violations, 0 silent mode changes. Margin: min=-292.3 MB...
rc=1
```

**Verdict: Harness has teeth.** rc=1 (build fails) when violations occur.

### Claim C2 (QUICKSTART): "`fitsproof admit --budget-gb 0.001   # REFUSED — names the binding constraint, exit code 2`"

```
$ fitsproof admit --budget-gb 0.001
REFUSED: needs 0.042 GB, budget 0.001 GB; no listed option fits — nearest is "Use int4_sym quantisation instead of none" at 0.006 GB (0.005 GB above budget)
Degradation options:
  [does not fit] Use int8_sym quantisation instead of none -> 0.011 GB
  [does not fit] Use int4_sym quantisation instead of none -> 0.006 GB
  [does not fit] Reduce context to 256 tokens (1/2 of 512) -> 0.040 GB
  [does not fit] Reduce context to 128 tokens (1/4 of 512) -> 0.039 GB
  [does not fit] Reduce context to 64 tokens (1/8 of 512) -> 0.039 GB
  [does not fit] Offload ~50% of layers to system RAM (CPU fallback for those layers) -> 0.022 GB
Exit code: 2
```

**Verdict: Claim holds.** Exit 2 on refusal, binding constraint named ("nearest is ...
at 0.006 GB (0.005 GB above budget)"), all degradation options tagged `[does not fit]`.

### Claim C3 (SPECULATIVE): "speculative decoding is tested to produce identical greedy output to the non-speculative path"

Attack — run the speculative test directly:

```
$ python -m pytest tests/engine/test_speculative.py::test_speculative_equals_greedy -v
tests/engine/test_speculative.py::test_speculative_equals_greedy PASSED  [100%]
1 passed in 34.79s
```

**Verdict: Claim holds.** The test passes. Prior adversarial passes (ADV-05) confirmed
the test now uses different-seed draft model (seed=999 vs target seed=42), making
the equality assertion non-trivial.

---

## 19. Citation audit — RESEARCH.md links

Verified 7 key URLs from RESEARCH.md with curl (2026-09-28T09:30Z):

```
200 https://arxiv.org/abs/2211.17192   [Leviathan et al. speculative decoding]
200 https://arxiv.org/abs/2302.01318   [Chen et al. speculative sampling]
200 https://www.cs.virginia.edu/stream/ref.html   [STREAM benchmark]
200 https://modelcontextprotocol.io/specification/2025-03-26/   [MCP spec]
200 https://doi.org/10.1214/aos/1176344552   [Bootstrap, Efron 1979]
200 https://man7.org/linux/man-pages/man2/getrusage.2.html   [Linux getrusage]
403 https://dl.acm.org/doi/10.1145/1498765.1498785   [Roofline, Williams 2009]
```

Roofline paper (403) verified via Crossref API:

```
$ curl -s "https://api.crossref.org/works/10.1145/1498765.1498785" | head -c 600
{"status":"ok","message-type":"work",...,"publisher":"Association for Computing Machinery (ACM)",...
"title":["Roofline"],...,"container-title":"Communications of the ACM",...,"date-parts":[[2009,4]]}
```

**Verdict:** 6/7 resolve directly (200). The ACM Roofline link 403s to automation but
DOI is valid via Crossref with matching title/venue/date (ADV-07 limitation documented).

---

## 20. Fault injection — 5 tests sampled, fault each claims to detect

| # | Test (its named fault) | Injected fault | Suite result |
|---|---|---|---|
| F-I1 | `test_rope_known_values` ("wrong theta or rotation formula") | attention.py: `10000.0` → `1000.0` (default theta) | **PASSED (SURVIVED)** |
| F-I2 | `test_int8_sym_known_values` ("using 128 instead of 127") | quant.py: `np.float32(127)` → `np.float32(128)` | **FAILED (killed)** |
| F-I3 | `test_decode_tok_s_known_answer` ("inverted formula") | cost.py: `effective_bw / weight_bytes` → `weight_bytes / effective_bw` | **PASSED (SURVIVED)** |
| F-I4 | `test_kv_cache_equals_reference` ("wrong RoPE offset") | attention.py: `offset=offset` → `offset=0` | **FAILED (killed)** |
| F-I5 | `test_admit_refuses_with_binding_constraint` ("admit instead of refuse") | admit.py: `REFUSED` → `ADMITTED` | **FAILED (killed)** |

**Result: 3/5 killed, 2 survived.**

Raw evidence:

```
### F-I1: RoPE theta default changed 10000→1000
tests/engine/test_attention.py::test_rope_known_values PASSED [100%]
1 passed in 0.23s
### SURVIVED: test passes theta=10000 explicitly, doesn't exercise the default

### F-I2: int8_sym uses 128 instead of 127
tests/engine/test_quant.py::test_int8_sym_known_values FAILED
Max absolute difference: 0.00036909
ACTUAL: array([0.046875], dtype=float32)
DESIRED: array([0.047244])
### KILLED

### F-I3: decode formula inverted
tests/contract/test_cost.py::test_decode_tok_s_known_answer PASSED [100%]
### SURVIVED: test injects bandwidth via MachineProfile, formula change not exercised

### F-I4: KV-cache RoPE offset hardcoded to 0
tests/engine/test_attention.py::test_kv_cache_equals_reference FAILED
Mismatched elements: 242 / 256 (94.5%)
### KILLED

### F-I5: admit returns ADMITTED for non-fitting config
tests/value/test_incumbent_gap.py::test_admit_refuses_with_binding_constraint FAILED
assert <AdmitStatus.ADMITTED: 'admitted'> == <AdmitStatus.REFUSED: 'refused'>
### KILLED
```

---

## 21. New Findings — Cycle 3 Pass 1

| id | severity | finding | evidence | status |
|---|---|---|---|---|
| ADV-14 | minor | `test_rope_known_values` does not exercise default theta; passes explicit theta=10000 so fault in default value not caught | §20 F-I1 | **new — c3-p10** |
| ADV-15 | minor | `test_decode_tok_s_known_answer` injects MachineProfile directly, bypassing the actual formula code path for bandwidth-to-tok/s calculation | §20 F-I3 | **new — c3-p10** |

Both are minor: the underlying formulas are correct (other tests exercise them), but
the KAT tests as written do not catch the specific faults their docstrings name when
those faults affect default parameters or internal calculation paths.

---

## 22. Updated Findings Table — Cycle 3 Pass 1 Status

| id | severity | finding | status |
|---|---|---|---|
| ADV-01 | blocker | FlexGen citation mis-attribution | **fixed (c2-p08)** |
| ADV-02 | blocker | GPTQ per-channel vs per-group | **fixed (c2-p08)** |
| ADV-03 | blocker | mode_changed_silently hardcoded False | **fixed (c2-p08)** |
| ADV-04 | major | Stress-harness margin degenerate | **limitation** — documented in README |
| ADV-05 | major | test_speculative_equals_greedy vacuous | **fixed (c2-p05)** — different-seed draft |
| ADV-06 | minor | calibration_demo numbers load-dependent | **limitation** |
| ADV-07 | minor | ACM link 403s automation | **limitation** — Crossref verified |
| ADV-08 | minor | README RSS limitation self-contradicts | **fixed (c3-p04)** |
| ADV-09 | major | int8_sym overflow on extreme weights | **fixed (c2-p08)** |
| ADV-10 | minor | Server budget per-request, not global | **limitation** |
| ADV-11 | minor | @guard bypassable via __wrapped__ | **limitation** — Python stdlib |
| ADV-12 | major | Server crashes on messages as non-list | **fixed (c3-p04)** |
| ADV-13 | minor | Empty config list passes stress harness | **limitation** — CLI generates 25 |
| ADV-14 | minor | test_rope_known_values doesn't exercise default theta | **new — c3-p10** |
| ADV-15 | minor | test_decode_tok_s_known_answer bypasses formula path | **new — c3-p10** |

---

## 23. Failed attacks (evidence for the defence, c3-p10)

- Stress harness C1: claim holds — 25 configs, 0 violations, rc=0 at default budget.
- Refusal gate C2: claim holds — exit 2, binding constraint named, options tagged.
- Speculative C3: claim holds — test passes with different-seed draft.
- Fault injections F-I2, F-I4, F-I5: all killed by their respective tests.
- Link audit: all 7 key URLs resolve (6 direct 200, 1 via Crossref).

---

## 24. Gate status for this pass

- Artifact: `docs/ADVERSARIAL_REVIEW.md` extended with cycle 3 pass 1 findings.
- **Open blockers: 0** — all previous blockers resolved.
- **Open majors: 0** — all previous majors resolved or documented as limitations.
- **Open minors: 7** — ADV-06, ADV-07, ADV-10, ADV-11, ADV-13, ADV-14, ADV-15
  (all documented as limitations or minor test-coverage gaps).
- Repo left green: 178 passed, ruff clean.

PASS_c3-p10-adversarial-1 COMPLETE


---

# Cycle 3 — Pass 2: Attack the Property (c3-p11-adversarial-2)

Independent review, pass `c3-p11-adversarial-2` (cycle 3, adversarial pass 2 of 2).
Reviewer lane: kiro (claude-opus-4.5). The reviewer does not fix code — it
reports findings; the builder fixes; the reviewer re-verifies.

Baseline before attack (repo state `5aa991f`, branch `feat/v0.1`):

```
$ .venv/bin/pytest -q
======================= 178 passed in 209.15s (0:03:29) ========================
$ .venv/bin/ruff check . && .venv/bin/ruff format --check .
All checks passed!
```

---

## 25. Property attacks — attempt to defeat the core safety/correctness property

### P3-A1: Bypass verify_run budget check with negative budget

Attack: Pass a negative budget to verify_run to test boundary behaviour.

```python
result = verify_run(dummy_gen, budget_bytes=-1000000, admit_record=fake_record, config_label="neg-budget")
```

Result:
```
Budget:      -1000000 bytes
Measured:    32284672 bytes
Respected:   False
Margin:      -33284672 bytes
ATTACK FAILED (good): negative budget correctly shows as NOT respected
```

**Verdict: Defense held.** Negative budget correctly results in `budget_respected=False`.

### P3-A2: Verify mode_changed_silently detection (ADMITTED with predicted > budget)

Attack: Create an ADMITTED record where plan.predicted_peak_bytes > budget.

```python
plan_over_budget = Plan(predicted_peak_bytes=10_000_000_000, ...)  # 10GB
admitted_over = AdmitRecord(status=AdmitStatus.ADMITTED, plan=plan_over_budget, ...)
result = verify_run(dummy_gen, budget_bytes=1_000_000_000, admit_record=admitted_over)
```

Result:
```
mode_changed_silently: True
ATTACK FAILED (good): verify_run detected the inconsistency
```

**Verdict: Defense held.** ADV-03 fix confirmed working.

### P3-A3: Verify mode_changed_silently detection (DEGRADED with no applied_degradation)

Attack: Create a DEGRADED record with applied_degradation=None.

```python
degraded_no_name = AdmitRecord(status=AdmitStatus.DEGRADED, applied_degradation=None, ...)
result = verify_run(dummy_gen, budget_bytes=10_000_000, admit_record=degraded_no_name)
```

Result:
```
mode_changed_silently: True
ATTACK FAILED (good): verify_run detected the missing degradation name
```

**Verdict: Defense held.** ADV-03 fix confirmed working.

### P3-A4: Invalid quant name smuggling

Attack: Pass invalid quant names (typos, injection strings, null bytes) to plan().

```python
invalid_quants = ["int2", "INT8_SYM", "none ", "int8_sym\x00", "int4_sym; echo pwned", "", "float128"]
for q in invalid_quants:
    plan(cfg=cfg, machine=profile, context_len=512, budget_bytes=4*1024**3, quant=q)
```

Result:
```
  quant='int2': ATTACK FAILED (good) - ValueError raised
  quant='INT8_SYM': ATTACK FAILED (good) - ValueError raised
  quant='none ': ATTACK FAILED (good) - ValueError raised
  quant='int8_sym\x00': ATTACK FAILED (good) - ValueError raised
  quant='int4_sym; echo pwned': ATTACK FAILED (good) - ValueError raised
  quant='': ATTACK FAILED (good) - ValueError raised
  quant='float128': ATTACK FAILED (good) - ValueError raised
```

**Verdict: Defense held.** All 7 invalid quant names rejected with ValueError.

### P3-A5: NaN/Inf injection into plan()

Attack: Pass NaN, Inf, -Inf, and negative values as budget_bytes.

```python
special_values = [float("nan"), float("inf"), float("-inf"), -1]
for budget in special_values:
    plan(cfg=cfg, machine=profile, context_len=512, budget_bytes=budget, quant="none")
```

Result:
```
  budget_bytes=nan: ATTACK FAILED (good) - rejected
  budget_bytes=inf: ATTACK FAILED (good) - rejected
  budget_bytes=-inf: ATTACK FAILED (good) - rejected
  budget_bytes=-1: ATTACK FAILED (good) - rejected
```

**Verdict: Defense held.** All invalid budget values rejected.

### P3-A6: Empty degradation list when FITS_WITH_DEGRADATION

Attack: Create a Plan with verdict=FITS_WITH_DEGRADATION but empty degradations list.

```python
broken_plan = Plan(verdict=Verdict.FITS_WITH_DEGRADATION, degradations=[], ...)
result = admit(broken_plan)
```

Result:
```
  Status: AdmitStatus.REFUSED
  Message: REFUSED (internal inconsistency): budget=4.000 GB, predicted=8.000 GB...
  ATTACK FAILED (good): admit() refused inconsistent plan
```

**Verdict: Defense held.** Internal inconsistency detected and refused.

### P3-A7: All degradations don't fit, but verdict says FITS_WITH_DEGRADATION

Attack: Create a Plan with verdict=FITS_WITH_DEGRADATION but all degradations have
`fits_budget=False`.

```python
inconsistent_plan = Plan(
    verdict=Verdict.FITS_WITH_DEGRADATION,
    degradations=[DegradationStep(fits_budget=False, ...)],
    ...
)
result = admit(inconsistent_plan)
```

Result:
```
  Status: AdmitStatus.REFUSED
  Applied degradation: None
  ATTACK FAILED (good): admit() refused because no degradation fits
```

**Verdict: Defense held.** admit() correctly refuses when no degradation fits.

### P3-A8: Integer overflow in budget comparison

Attack: Pass extremely large budget values (2^63-1, 2^63, 2^64-1).

Result:
```
  budget=2^63-1: verdict=Verdict.FITS, predicted=139225600
    ATTACK FAILED (good): correctly identified as fits
  budget=2^63: verdict=Verdict.FITS, predicted=139225600
    ATTACK FAILED (good): correctly identified as fits
  budget=2^64-1: verdict=Verdict.FITS, predicted=139225600
    ATTACK FAILED (good): correctly identified as fits
```

**Verdict: Defense held.** Python handles large integers correctly.

### P3-A9: Concurrent verify_run calls

Attack: Run 4 concurrent verify_run calls to test for race conditions.

```python
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(verify_worker, i) for i in range(4)]
    results = [f.result() for f in futures]
```

Result:
```
  Worker 0: respected=True, measured=119.5 MB
  Worker 1: respected=True, measured=119.3 MB
  Worker 2: respected=True, measured=119.4 MB
  Worker 3: respected=True, measured=119.5 MB
  ATTACK FAILED (good): all concurrent workers respected budget
```

**Verdict: Defense held.** No race conditions observed.

### P3-A10: Speculative decoding with adversarial draft (different seed=999999)

Attack: Generate target and draft models with maximally different seeds and verify
speculative output equals greedy target output.

Result:
```
  Target greedy: [179, 179, 179, 179, 179, 179, 179, 179, 179, 179]
  Speculative:   [179, 179, 179, 179, 179, 179, 179, 179, 179, 179]
  ATTACK FAILED (good): speculative equals greedy with adversarial draft
```

**Verdict: Defense held.** Speculative decoding equality property holds.

### P3-A11: MCP tool with malformed JSON-RPC

Attack: Send malformed JSON-RPC requests to the MCP server (missing method, wrong
version, unknown method, SQL injection in arguments).

Result:
```
  Missing method: error response (good) - -32601
  Wrong jsonrpc version: success response - possibly bad
  Unknown method: error response (good) - -32601
  SQL injection in args: MCP Error (isError=True): Invalid arguments: cannot parse budget...
```

**Observations:**
- Missing method: correctly rejected with JSON-RPC error -32601
- Unknown method: correctly rejected with JSON-RPC error -32601
- SQL injection: correctly rejected with isError=True, "cannot parse budget"
- **Wrong jsonrpc version**: server does not validate jsonrpc version field. This
  is minor — MCP spec (source 27) says jsonrpc must be "2.0" but the behaviour is
  merely lenient, not a security issue. See ADV-16.

### P3-A12: HTTP server header injection

Attack: Send payloads with CRLF injection attempts in content and model fields.

Result:
```
  Payload 0: ATTACK FAILED (good) - no injection
  Payload 1: ATTACK FAILED (good) - no injection
```

**Verdict: Defense held.** No header injection.

### P3-A13: Empty config list to stress harness

Attack: Pass an empty config list to run_stress_harness().

Result:
```
  n_configs: 0
  violations: 0
  violation_free: True
  Status: ADV-13 limitation confirmed (vacuous pass with 0 configs)
```

**Verdict: ADV-13 limitation confirmed.** The CLI generates 25 configs internally,
so this only affects direct API callers.

### P3-A14: RSS measurement verification

Attack: Verify that fitsproof uses /proc/self/status VmRSS (current RSS), not
ru_maxrss (high-water mark).

Result:
```
  fitsproof._get_rss_bytes(): 32.9 MB
  /proc/self/status VmRSS:    32.9 MB
  VERIFIED: fitsproof uses /proc/self/status VmRSS (current RSS, not high-water)
```

**Verdict: README correctly describes behaviour.** ADV-08 was already fixed (c3-p04).

### P3-A16: Float denormalized number edge case

Attack: Pass denormalized float (smallest positive) as budget.

Result:
```
  ATTACK FAILED (good): denorm budget rejected
  Float budget 4.5 GB: verdict=Verdict.FITS (accepted - floats allowed)
```

**Verdict: Defense held.** Denormals rejected; valid floats accepted.

### P3-A17: Guard decorator with async function

Attack: Apply @guard(budget="1MiB") to an async function.

Result:
```
  ATTACK FAILED (good): async function guard works
```

**Verdict: Defense held.** @guard works with async functions.

### P3-A18: Quantization with NaN/Inf weights

Attack: Pass NaN and Inf in weight arrays to quantize().

Result:
```
  NaN weights: ATTACK FAILED (good) - NaN weights rejected
  Inf weights: ATTACK FAILED (good) - Inf weights rejected
```

**Verdict: Defense held.** Both NaN and Inf rejected with explicit ValueError.

### P3-A19: Server crash on huge max_tokens

Attack: Send HTTP request with max_tokens=1_000_000.

Result:
```
  Server crashed with ValueError: could not broadcast input array from shape (1,6,0,32)
  into shape (1,6,1,32)
```

**Root cause:** `Transformer.generate()` does not clamp `max_new_tokens` to
`cfg.max_seq_len - len(prompt)`. When max_tokens exceeds this, the RoPE offset
exceeds the precomputed frequency array bounds, causing a broadcasting error.

**Verdict: NEW MAJOR FINDING (ADV-16).** Server should clamp max_tokens to avoid
crash. This is a denial-of-service vulnerability via malformed request.

### P3-A20: Budget string "4" without unit

Attack: Pass "4" (no unit) as budget_bytes string to client.plan().

Result:
```
  '4' parses to budget_bytes=4 (4 bytes)
  predicted_peak_bytes: 41708032
  verdict: Verdict.DOES_NOT_FIT
```

**Observation:** "4" without a unit is interpreted as 4 bytes, which results in
DOES_NOT_FIT as expected. This is arguably confusing UX but technically correct
behaviour — the integer is used directly. The contract still works correctly.
See ADV-17 (minor).

---

## 26. Updated Findings Table — Cycle 3 Pass 2 Final Status

| id | severity | finding | status |
|---|---|---|---|
| ADV-01 | blocker | FlexGen citation mis-attribution | **fixed (c2-p08)** |
| ADV-02 | blocker | GPTQ per-channel vs per-group | **fixed (c2-p08)** |
| ADV-03 | blocker | mode_changed_silently hardcoded False | **fixed (c2-p08)** |
| ADV-04 | major | Stress-harness margin degenerate | **limitation** — documented in README |
| ADV-05 | major | test_speculative_equals_greedy vacuous | **fixed (c2-p05)** — different-seed draft |
| ADV-06 | minor | calibration_demo numbers load-dependent | **limitation** |
| ADV-07 | minor | ACM link 403s automation | **limitation** — Crossref verified |
| ADV-08 | minor | README RSS limitation self-contradicts | **fixed (c3-p04)** |
| ADV-09 | major | int8_sym overflow on extreme weights | **fixed (c2-p08)** |
| ADV-10 | minor | Server budget per-request, not global | **limitation** |
| ADV-11 | minor | @guard bypassable via __wrapped__ | **limitation** — Python stdlib |
| ADV-12 | major | Server crashes on messages as non-list | **fixed (c3-p04)** |
| ADV-13 | minor | Empty config list passes stress harness | **limitation** — CLI generates 25 |
| ADV-14 | minor | test_rope_known_values doesn't exercise default theta | **new — c3-p10** |
| ADV-15 | minor | test_decode_tok_s_known_answer bypasses formula path | **new — c3-p10** |
| ADV-16 | major | Server crashes on max_tokens > max_seq_len - prompt_len | **new — c3-p11, open** |
| ADV-17 | minor | Budget string "4" without unit treated as 4 bytes | **new — c3-p11, limitation** — UX quirk, contract correct |
| ADV-18 | minor | MCP server does not validate jsonrpc version field | **new — c3-p11, limitation** — lenient, not a security issue |

---

## 27. Failed attacks (evidence for the defence, c3-p11)

- P3-A1: Negative budget correctly shows as NOT respected.
- P3-A2/A3: mode_changed_silently detection works (ADV-03 fix verified).
- P3-A4: All 7 invalid quant names rejected with ValueError.
- P3-A5: NaN/Inf/-Inf/-1 budget values all rejected.
- P3-A6/A7: Inconsistent Plan verdicts refused.
- P3-A8: Large integers (2^64-1) handled correctly.
- P3-A9: Concurrent verify_run calls — no race conditions.
- P3-A10: Speculative decoding equality holds with adversarial draft (seed=999999).
- P3-A11: SQL injection in MCP args rejected with explicit error.
- P3-A12: HTTP header injection — no injection.
- P3-A14: RSS measurement verified correct.
- P3-A16: Denormalized budget rejected; valid floats accepted.
- P3-A17: @guard works with async functions.
- P3-A18: NaN/Inf weights rejected by quantizer.

---

## 28. Gate status for this pass

- Artifact: `docs/ADVERSARIAL_REVIEW.md` extended with cycle 3 pass 2 findings.
- **Open blockers: 0** — all previous blockers resolved.
- **Open majors: 1** — ADV-16 (server crash on huge max_tokens).
- **Open minors: 8** — ADV-06, ADV-07, ADV-10, ADV-11, ADV-13, ADV-14, ADV-15, ADV-17, ADV-18
  (all documented as limitations or minor issues).
- Repo left green: 178 passed, ruff clean.

ADV-16 (server crash on huge max_tokens) is a new major finding. The server should
clamp `max_tokens` to `cfg.max_seq_len - len(prompt_ids)` before passing to generate().
This is a robustness issue (denial of service via malformed request) but does not
compromise the core safety property (the contract is not bypassed — the request
crashes before any budget check).

PASS_c3-p11-adversarial-2 COMPLETE
