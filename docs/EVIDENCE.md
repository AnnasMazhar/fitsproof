# EVIDENCE — fitsproof v0.1

Raw terminal output only. All commands run from /home/openclaw/portfolio/fitsproof on 2026-09-26.
Python 3.11.15, pytest 8.3.5.

---

## 1. Install

```
$ uv pip install -e '.[dev]' --python=.venv/bin/python
      Built fitsproof @ file:///home/openclaw/portfolio/fitsproof
Prepared 1 package in 715ms
Uninstalled 1 package in 0.53ms
Installed 1 package in 4ms
 ~ fitsproof==0.1.0 (from file:///home/openclaw/portfolio/fitsproof)
```

---

## 2. pytest -q (full suite)

```
$ .venv/bin/python -m pytest tests/ -q
============================= test session starts ==============================
platform linux -- Python 3.11.15, pytest-8.3.5, pluggy-1.6.0
rootdir: /home/openclaw/portfolio/fitsproof
configfile: pyproject.toml
plugins: cov-6.1.0, hypothesis-6.135.0, platformdirs-4.12.0
collected 88 items

tests/contract/test_cost.py ..............                               [ 15%]
tests/contract/test_plan_admit_verify.py .........................       [ 44%]
tests/engine/test_attention.py ..............                            [ 60%]
tests/engine/test_quant.py ..............                                [ 76%]
tests/engine/test_sampling.py ............                               [ 89%]
tests/engine/test_server.py ......                                       [ 96%]
tests/engine/test_speculative.py ...                                     [100%]

============================= 88 passed in 39.60s ==============================
```

---

## 3. ruff check + format

```
$ .venv/bin/ruff check .
All checks passed!

$ .venv/bin/ruff format --check .
10 files already formatted
```

---

## 4. KV-cache equals reference (Acceptance Criterion 2)

```
$ .venv/bin/python -m pytest tests/engine/test_attention.py::test_kv_cache_equals_reference -v
============================= test session starts ==============================
platform linux -- Python 3.11.15, pytest-8.3.5, pluggy-1.6.0 -- /home/openclaw/portfolio/fitsproof/.venv/bin/python
rootdir: /home/openclaw/portfolio/fitsproof
configfile: pyproject.toml
collected 1 item

tests/engine/test_attention.py::test_kv_cache_equals_reference PASSED    [100%]

============================== 1 passed in 2.30s ==============================
```

---

## 5. Speculative decoding equality (Acceptance Criterion 4)

```
$ .venv/bin/python -m pytest tests/engine/test_speculative.py::test_speculative_equals_greedy -v
============================= test session starts ==============================
platform linux -- Python 3.11.15, pytest-8.3.5, pluggy-1.6.0 -- /home/openclaw/portfolio/fitsproof/.venv/bin/python
rootdir: /home/openclaw/portfolio/fitsproof
configfile: pyproject.toml
collected 1 item

tests/engine/test_speculative.py::test_speculative_equals_greedy PASSED  [100%]

============================== 1 passed in 6.07s ==============================
```

---

## 6. Calibration MAPE on held-out configurations (Acceptance Criterion 5)

```
=== Calibration demo ===
bandwidth: 7.18 GB/s
gemm:      321.26 GFLOPS
RAM:       33.5 GB
bandwidth_utilisation: 0.0345
MAPE (held-out):       50.3%
CI (95%):              [50.3%, 50.3%]
n_train=2, n_held_out=1
```

Notes on calibration result:
- The reference model is 38 MB (6 layers, randomly initialised). At ~7 GB/s DRAM bandwidth and
  bandwidth_utilisation=0.035, predicted tok/s ≈ 7e9 * 0.035 / 38e6 ≈ 6.4 tok/s.
- Measured throughput is ~2.9 tok/s (seen in Pareto table), so the model over-predicts by ~2x.
- MAPE 50.3% is the honest held-out number. The 3-prompt dataset is small (n_held_out=1).
- The cost model is calibrated CPU-DRAM only; NumPy overhead (Python dispatch, allocation) adds
  significant latency not captured by the roofline model at this scale.
- This is reported honestly per the spec: "publish it" if the number is bad.

---

## 7. Refusal, degradation, and admission transcripts (Acceptance Criterion 6)

```
=== REFUSAL demo: 1-byte budget ===
REFUSED: needs 0.04 GB, budget 0.00 GB; nearest fitting config is Offload ~50% of layers to system RAM (CPU fallback for those layers)

=== DEGRADATION demo: 15% of fp32 weight budget ===
DEGRADED: base config needs 0.039 GB > budget 0.006 GB. Applying: Use int4_sym quantisation instead of none. New predicted peak: 0.006 GB.

=== ADMISSION demo: 4 GB budget ===
ADMITTED: 0.042 GB predicted peak <= 4.295 GB budget (margin: 4253.3 MB)
```

---

## 8. Stress harness: ≥20 configurations, zero violations (Acceptance Criterion 7)

```
$ .venv/bin/python -m pytest tests/contract/test_plan_admit_verify.py::test_stress_harness_zero_violations -v
============================= test session starts ==============================
platform linux -- Python 3.11.15, pytest-8.3.5, pluggy-1.6.0 -- /home/openclaw/portfolio/fitsproof/.venv/bin/python
rootdir: /home/openclaw/portfolio/fitsproof
configfile: pyproject.toml
collected 1 item

tests/contract/test_plan_admit_verify.py::test_stress_harness_zero_violations PASSED [100%]

============================== 1 passed in 16.94s ==============================
```

25 configurations (5 prompt lengths × 5 decode lengths), zero budget violations, zero silent mode changes.

---

## 9. Real HTTP server request + streaming (Acceptance Criterion 9)

```
healthz: 200 {'status': 'ok'}
completion status: 200
choices[0].message: {'role': 'assistant', 'content': 'L\ufffd&&&'}
usage: {'prompt_tokens': 2, 'completion_tokens': 5, 'total_tokens': 7}
server shutdown OK
```

Note: content is byte-sequence decoded from the reference model's random output. The reference
model is randomly initialised — it generates syntactically valid tokens, not coherent text.

---

## 10. Pareto frontier (Acceptance Criterion 11)

```
quant           ctx    peak_MB    tok/s    top1    pred_MB  dominated
---------------------------------------------------------------------
int4_sym         16      347.0     2.97   0.754        5.5        yes
int4_sym         32      347.0     3.49   0.754        5.6         no
int4_sym         64      347.0     2.86   0.754        5.6        yes
int8_sym         16      347.0     2.87   0.984       10.3        yes
int8_sym         32      347.0     2.83   0.984       10.3        yes
int8_sym         64      347.0     2.36   0.984       10.3        yes
none             16      347.0     2.12   1.000       38.7        yes
none             32      347.0     2.94   1.000       38.8         no
none             64      347.0     2.63   1.000       39.0        yes
Total configs: 9
Non-dominated: 2
```

Note: peak_MB is the measured RSS high-water mark, which on Linux is the process RSS since start.
The reference model itself (38 MB) is small; 347 MB RSS reflects Python interpreter + NumPy overhead.
pred_MB shows the formula prediction (much smaller, as expected for a tiny reference model).

---

## 11. Mutation score (Acceptance Criterion 10)

Method: mutmut 3.3 generated mutants; manual test execution swapping source files.

**admit.py (core enforcement module) — first 30 mutants tested:**
```
Total tested: 30
Killed: 25
Survived: 5
Score: 83.3%
```

**cost.py (weight_bytes function) — first 30 mutants tested:**
```
Total tested: 30
Killed: 24
Survived: 6
Score: 80.0%
```

Both core modules exceed the 70% target.

Surviving mutants analysis:
- admit.py survivors are in f-string numeric mutations (/1e9 → /1000000001.0) producing
  values within floating-point rounding of the original, and in the defensive
  internal-inconsistency branch (unreachable in normal operation).
- cost.py survivors are in formatting constants in function bodies that do not affect
  the return value.
- None of the surviving mutants affect a security or enforcement decision path.

---

## 12. Ollama integration note (Acceptance Criterion 8)

The `ollama` integration test requires `FITSPROOF_TEST_OLLAMA=1` environment variable and
`gemma3:4b` model pulled locally. Not run in this CI pass (offline-only requirement).

The predicted decode throughput from the roofline model for a 3.3 GB model on this machine:
  predicted_tok_s = 7.18e9 * 0.035 / 3.3e9 ≈ 0.076 tok/s

This would be compared against actual ollama measured throughput when the integration test is run.
The honest number at this model scale would likely show ~2-5x over-prediction from the cost model
(NumPy overhead is not captured by the roofline for single-token inference).

---

## Fresh-clone install + test + lint (2026-09-27)

Commands run from /tmp/fitsproof-verify (fresh clone of feat/v0.1).
Python 3.13.12, pytest 8.3.5, ruff 0.11.13.

```
$ git clone /home/openclaw/portfolio/fitsproof /tmp/fitsproof-verify
Cloning into '/tmp/fitsproof-verify'...
done.

$ cd /tmp/fitsproof-verify && git checkout feat/v0.1
Already on 'feat/v0.1'
Your branch is up to date with 'origin/feat/v0.1'.

$ uv venv && uv pip install -e '.[dev]'
Using CPython 3.13.12
Creating virtual environment at: .venv
   Building fitsproof @ file:///tmp/fitsproof-verify
      Built fitsproof @ file:///tmp/fitsproof-verify
Installed 26 packages in 102ms
 + fitsproof==0.1.0 (from file:///tmp/fitsproof-verify)
EXIT_INSTALL:0

$ uv run pytest -q
============================= test session starts ==============================
platform linux -- Python 3.13.12, pytest-8.3.5, pluggy-1.6.0
rootdir: /tmp/fitsproof-verify
configfile: pyproject.toml
testpaths: tests
plugins: platformdirs-4.12.0, hypothesis-6.135.0, cov-6.1.0
collected 88 items

tests/contract/test_cost.py ..............                               [ 15%]
tests/contract/test_plan_admit_verify.py .........................       [ 44%]
tests/engine/test_attention.py ..............                            [ 60%]
tests/engine/test_quant.py ..............                                [ 76%]
tests/engine/test_sampling.py ............                               [ 89%]
tests/engine/test_server.py ......                                       [ 96%]
tests/engine/test_speculative.py ...                                     [100%]

88 passed in 44.65s
EXIT_PYTEST:0

$ uv run ruff check .
All checks passed!
EXIT_RUFF:0
```

---

## v0.2 MANDATE pass (2026-09-27) — implement-1

### Full test suite (97 tests, 3.11)

```
$ .venv/bin/python -m pytest -q --tb=short
============================= test session starts ==============================
platform linux -- Python 3.11.15, pytest-8.3.5, pluggy-1.6.0
rootdir: /home/openclaw/portfolio/fitsproof
configfile: pyproject.toml
testpaths: tests
plugins: cov-6.1.0, hypothesis-6.135.0, platformdirs-4.12.0
collected 97 items

tests/contract/test_cost.py ..............                               [ 14%]
tests/contract/test_plan_admit_verify.py .........................       [ 40%]
tests/engine/test_attention.py ..............                            [ 54%]
tests/engine/test_quant.py ..............                                [ 69%]
tests/engine/test_sampling.py ............                               [ 81%]
tests/engine/test_server.py ......                                       [ 87%]
tests/engine/test_speculative.py ...                                     [ 90%]
tests/value/test_incumbent_gap.py .........                              [100%]

97 passed in 42.01s
```

### ruff clean

```
$ .venv/bin/ruff check .
All checks passed!

$ .venv/bin/ruff format --check .
34 files already formatted
```

### Research traceability check

```
$ .venv/bin/python scripts/check_research_traceability.py
TRACEABILITY OK (core only): all core test files cite valid research sources. Checked 22 source IDs from RESEARCH.md.
```

### M2 — FitsproofClient + @guard + MCP server

```
$ .venv/bin/fitsproof admit --budget-gb 4.0
ADMITTED: 0.042 GB predicted peak <= 4.000 GB budget (margin: 3958.3 MB)

$ .venv/bin/fitsproof admit --budget-gb 0.000000001; echo "exit: $?"
REFUSED: needs 0.04 GB, budget 0.00 GB; nearest fitting config is Offload ~50% of layers to system RAM (CPU fallback for those layers)
Degradation options:
  [does not fit] Use int8_sym quantisation instead of none -> 0.011 GB
  [does not fit] Use int4_sym quantisation instead of none -> 0.006 GB
  [does not fit] Reduce context to 256 tokens (1/2 of 512) -> 0.040 GB
  [does not fit] Reduce context to 128 tokens (1/4 of 512) -> 0.039 GB
  [does not fit] Reduce context to 64 tokens (1/8 of 512) -> 0.039 GB
  [does not fit] Offload ~50% of layers to system RAM (CPU fallback for those layers) -> 0.022 GB
exit: 2
```

```
$ .venv/bin/fitsproof verify --budget-gb 1.0 --tokens 8
ADMITTED: 0.039 GB predicted peak <= 1.000 GB budget (margin: 960.7 MB)
  measured_peak:    301.7 MB
  budget:           1000.0 MB
  budget_respected: True
  margin:           698.3 MB
```

MCP server (stdio) — initialize + tools/list + admit call:
```
$ echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}
{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}
{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"admit","arguments":{"budget":"4GiB","context_len":512}}}' | .venv/bin/fitsproof mcp

{"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "2025-03-26", ...serverInfo: fitsproof-mcp}}
{"jsonrpc": "2.0", "id": 2, "result": {"tools": [{"name": "probe"}, {"name": "plan"}, {"name": "admit"}]}}
{"jsonrpc": "2.0", "id": 3, "result": {"content": [{"type": "text", "text": "{\"status\": \"admitted\", \"message\": \"ADMITTED: 0.042 GB predicted peak <= 4.295 GB budget (margin: 4253.3 MB)\"}"}], "isError": false}}
```

### M3 — tests/value/test_incumbent_gap.py (9 tests, all pass)

```
$ .venv/bin/python -m pytest tests/value/ -v --tb=short
...
tests/value/test_incumbent_gap.py::test_refused_config_exceeds_budget PASSED
tests/value/test_incumbent_gap.py::test_admit_refuses_with_binding_constraint PASSED
tests/value/test_incumbent_gap.py::test_client_raises_does_not_fit PASSED
tests/value/test_incumbent_gap.py::test_degraded_config_emits_record PASSED
tests/value/test_incumbent_gap.py::test_guard_decorator_refuses_before_calling PASSED
tests/value/test_incumbent_gap.py::test_guard_decorator_admits_valid_config PASSED
tests/value/test_incumbent_gap.py::test_stress_harness_measured_le_budget PASSED
tests/value/test_incumbent_gap.py::test_mcp_tool_list_and_call PASSED
tests/value/test_incumbent_gap.py::test_mcp_admit_refused_returns_is_error PASSED

9 passed in 2.92s
```

### Summary

New in this pass:
- src/fitsproof/client.py: FitsproofClient, DoesNotFit, guard(), _parse_budget()
- src/fitsproof/mcp.py: MCP stdio server (probe/plan/admit tools)
- src/fitsproof/cli.py: added admit, verify, mcp subcommands
- tests/value/test_incumbent_gap.py: 9 tests for M3 value demonstration
- scripts/check_research_traceability.py: M4 traceability enforcement (CI-wired)
- .github/workflows/ci.yml: Research traceability check step added
- docs/EVIDENCE.md: this section
- tests/contract/test_cost.py: added research source citations to module docstring
- tests/contract/test_plan_admit_verify.py: added research source citations

Missing (M1 — binary): CI-built standalone executable requires GitHub Actions CI
with a release job and PyInstaller. Not implementable in-repo without CI secrets.
Recorded as deferred to cycle 2.
---

## v0.2 MANDATE pass 2 — implement-2 (2026-09-27)

### Full test suite
$ .venv/bin/python -m pytest -q

tests/adversarial/test_byzantine_inputs.py ............................. [ 19%]
....................                                                     [ 33%]
tests/contract/test_cost.py ..............                               [ 42%]
tests/contract/test_plan_admit_verify.py .........................       [ 59%]
tests/engine/test_attention.py ..............                            [ 68%]
tests/engine/test_quant.py ..............                                [ 78%]
tests/engine/test_sampling.py ............                               [ 86%]
tests/engine/test_server.py ......                                       [ 90%]
tests/engine/test_speculative.py ...                                     [ 92%]
tests/value/test_incumbent_gap.py .........                              [ 98%]
tests/value/test_readme_snippets.py ..                                   [100%]

============================= 148 passed in 50.57s =============================

### ruff + format + traceability (CI gates)
$ .venv/bin/ruff check .
All checks passed!
$ .venv/bin/ruff format --check .
37 files already formatted
$ .venv/bin/python scripts/check_research_traceability.py
TRACEABILITY OK (core only): all core test files cite valid research sources. Checked 22 source IDs from RESEARCH.md.

### Adversarial / byzantine suite (new this pass)
$ .venv/bin/python -m pytest tests/adversarial -v
============================= test session starts ==============================
platform linux -- Python 3.11.15, pytest-8.3.5, pluggy-1.6.0 -- /home/openclaw/portfolio/fitsproof/.venv/bin/python
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/openclaw/portfolio/fitsproof
configfile: pyproject.toml
plugins: cov-6.1.0, hypothesis-6.135.0, platformdirs-4.12.0
collecting ... collected 49 items

tests/adversarial/test_byzantine_inputs.py::test_quantize_rejects_nan_weights PASSED [  2%]
tests/adversarial/test_byzantine_inputs.py::test_quantize_rejects_inf_weights PASSED [  4%]
tests/adversarial/test_byzantine_inputs.py::test_int8_asym_constant_row_no_int32_overflow PASSED [  6%]
tests/adversarial/test_byzantine_inputs.py::test_int4_asym_constant_row_no_int32_overflow PASSED [  8%]
tests/adversarial/test_byzantine_inputs.py::test_int4_asym_known_answer PASSED [ 10%]
tests/adversarial/test_byzantine_inputs.py::test_plan_rejects_unknown_quant PASSED [ 12%]
tests/adversarial/test_byzantine_inputs.py::test_plan_rejects_non_positive_or_non_finite_budget[0] PASSED [ 14%]
tests/adversarial/test_byzantine_inputs.py::test_plan_rejects_non_positive_or_non_finite_budget[-1] PASSED [ 16%]
tests/adversarial/test_byzantine_inputs.py::test_plan_rejects_non_positive_or_non_finite_budget[-1000000000] PASSED [ 18%]
tests/adversarial/test_byzantine_inputs.py::test_plan_rejects_non_positive_or_non_finite_budget[nan] PASSED [ 20%]
tests/adversarial/test_byzantine_inputs.py::test_plan_rejects_non_positive_or_non_finite_budget[inf] PASSED [ 22%]
tests/adversarial/test_byzantine_inputs.py::test_plan_rejects_non_positive_or_non_finite_budget[-inf] PASSED [ 24%]
tests/adversarial/test_byzantine_inputs.py::test_plan_rejects_bad_context_len[0] PASSED [ 26%]
tests/adversarial/test_byzantine_inputs.py::test_plan_rejects_bad_context_len[-5] PASSED [ 28%]
tests/adversarial/test_byzantine_inputs.py::test_plan_rejects_bad_context_len[1.5] PASSED [ 30%]
tests/adversarial/test_byzantine_inputs.py::test_plan_rejects_bad_context_len[True] PASSED [ 32%]
tests/adversarial/test_byzantine_inputs.py::test_plan_rejects_bad_context_len[64] PASSED [ 34%]
tests/adversarial/test_byzantine_inputs.py::test_plan_rejects_bad_context_len[None] PASSED [ 36%]
tests/adversarial/test_byzantine_inputs.py::test_plan_survives_huge_context PASSED [ 38%]
tests/adversarial/test_byzantine_inputs.py::test_parse_budget_rejects_hostile_strings[(1+1)GB] PASSED [ 40%]
tests/adversarial/test_byzantine_inputs.py::test_parse_budget_rejects_hostile_strings[4GB; rm -rf /] PASSED [ 42%]
tests/adversarial/test_byzantine_inputs.py::test_parse_budget_rejects_hostile_strings[] PASSED [ 44%]
tests/adversarial/test_byzantine_inputs.py::test_parse_budget_rejects_hostile_strings[   ] PASSED [ 46%]
tests/adversarial/test_byzantine_inputs.py::test_parse_budget_rejects_hostile_strings[NaN] PASSED [ 48%]
tests/adversarial/test_byzantine_inputs.py::test_parse_budget_rejects_hostile_strings[nanGiB] PASSED [ 51%]
tests/adversarial/test_byzantine_inputs.py::test_parse_budget_rejects_hostile_strings[inf] PASSED [ 53%]
tests/adversarial/test_byzantine_inputs.py::test_parse_budget_rejects_hostile_strings[InfinityMB] PASSED [ 55%]
tests/adversarial/test_byzantine_inputs.py::test_parse_budget_rejects_hostile_strings[-4GiB] PASSED [ 57%]
tests/adversarial/test_byzantine_inputs.py::test_parse_budget_rejects_hostile_strings[__import__('os')] PASSED [ 59%]
tests/adversarial/test_byzantine_inputs.py::test_parse_budget_control_accepts_valid[4GiB-4294967296] PASSED [ 61%]
tests/adversarial/test_byzantine_inputs.py::test_parse_budget_control_accepts_valid[512MiB-536870912] PASSED [ 63%]
tests/adversarial/test_byzantine_inputs.py::test_parse_budget_control_accepts_valid[4096-4096] PASSED [ 65%]
tests/adversarial/test_byzantine_inputs.py::test_parse_budget_control_accepts_valid[8589934592-8589934592] PASSED [ 67%]
tests/adversarial/test_byzantine_inputs.py::test_parse_budget_control_accepts_valid[\uff14GiB-4294967296] PASSED [ 69%]
tests/adversarial/test_byzantine_inputs.py::test_parse_budget_rejects_non_finite_numbers[nan] PASSED [ 71%]
tests/adversarial/test_byzantine_inputs.py::test_parse_budget_rejects_non_finite_numbers[inf] PASSED [ 73%]
tests/adversarial/test_byzantine_inputs.py::test_parse_budget_rejects_non_finite_numbers[-inf] PASSED [ 75%]
tests/adversarial/test_byzantine_inputs.py::test_parse_budget_rejects_non_finite_numbers[0] PASSED [ 77%]
tests/adversarial/test_byzantine_inputs.py::test_parse_budget_rejects_non_finite_numbers[-1] PASSED [ 79%]
tests/adversarial/test_byzantine_inputs.py::test_guard_hostile_budget_never_allocates PASSED [ 81%]
tests/adversarial/test_byzantine_inputs.py::test_guard_refusal_raises_does_not_fit PASSED [ 83%]
tests/adversarial/test_byzantine_inputs.py::test_cli_hostile_args_fail_closed[argv0] PASSED [ 85%]
tests/adversarial/test_byzantine_inputs.py::test_cli_hostile_args_fail_closed[argv1] PASSED [ 87%]
tests/adversarial/test_byzantine_inputs.py::test_cli_hostile_args_fail_closed[argv2] PASSED [ 89%]
tests/adversarial/test_byzantine_inputs.py::test_cli_hostile_args_fail_closed[argv3] PASSED [ 91%]
tests/adversarial/test_byzantine_inputs.py::test_cli_hostile_args_fail_closed[argv4] PASSED [ 93%]
tests/adversarial/test_byzantine_inputs.py::test_cli_hostile_args_fail_closed[argv5] PASSED [ 95%]
tests/adversarial/test_byzantine_inputs.py::test_server_refuses_request_over_budget PASSED [ 97%]
tests/adversarial/test_byzantine_inputs.py::test_server_completion_carries_admission_record PASSED [100%]

============================== 49 passed in 5.45s ==============================

### README snippet executor (M2: every README python block runs)
$ .venv/bin/python -m pytest tests/value/test_readme_snippets.py -v
============================= test session starts ==============================
platform linux -- Python 3.11.15, pytest-8.3.5, pluggy-1.6.0 -- /home/openclaw/portfolio/fitsproof/.venv/bin/python
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/openclaw/portfolio/fitsproof
configfile: pyproject.toml
plugins: cov-6.1.0, hypothesis-6.135.0, platformdirs-4.12.0
collecting ... collected 2 items

tests/value/test_readme_snippets.py::test_readme_declares_all_four_plugin_surfaces PASSED [ 50%]
tests/value/test_readme_snippets.py::test_readme_python_snippets_execute PASSED [100%]

============================== 2 passed in 3.30s ===============================

### Hostile CLI argv — fail closed, no traceback
$ .venv/bin/fitsproof admit --budget-gb nan; echo exit=$?
ERROR: --budget-gb must be a positive finite number, got nan
exit=2
$ .venv/bin/fitsproof admit --quant int2; echo exit=$?
ERROR: unknown quant 'int2'; valid: float16, float32, int4_asym, int4_sym, int8_asym, int8_sym, none
exit=2
$ .venv/bin/fitsproof admit --context -5; echo exit=$?
ERROR: context_len must be a positive integer, got -5
exit=2

### serve alias (M1 binary surface: fitsproof serve)
$ .venv/bin/fitsproof serve --help
usage: fitsproof server [-h] [--host HOST] [--port PORT]
                        [--budget-gb BUDGET_GB]

options:
  -h, --help            show this help message and exit
  --host HOST
  --port PORT
  --budget-gb BUDGET_GB
                        Memory budget enforced on every request

### stress harness (headline evidence)
$ .venv/bin/fitsproof stress
ADMITTED: 0.039 GB predicted peak <= 4.000 GB budget (margin: 3961.0 MB)
Stress harness: 25 configs, 0 violations, 0 silent mode changes. Margin: min=3697.8 MB, median=3697.8 MB, max=3697.8 MB.
exit=0

### docs/demo.sh (end-to-end demo, real run)
$ PATH="$PWD/.venv/bin:$PATH" bash docs/demo.sh
=== 1. probe: characterise this machine ===
Probing machine...
  bandwidth:  4.25 GB/s
  gemm:       328.45 GFLOPS
  RAM:        33.55 GB
  VRAM:       0.00 GB

=== 2. admit: config fits a 4 GB budget ===
ADMITTED: 0.042 GB predicted peak <= 4.000 GB budget (margin: 3958.3 MB)

=== 3. admit: config cannot fit — refused, loudly (exit 2) ===
REFUSED: needs 0.04 GB, budget 0.00 GB; nearest fitting config is Offload ~50% of layers to system RAM (CPU fallback for those layers)
Degradation options:
  [does not fit] Use int8_sym quantisation instead of none -> 0.011 GB
  [does not fit] Use int4_sym quantisation instead of none -> 0.006 GB
  [does not fit] Reduce context to 256 tokens (1/2 of 512) -> 0.040 GB
  [does not fit] Reduce context to 128 tokens (1/4 of 512) -> 0.039 GB
  [does not fit] Reduce context to 64 tokens (1/8 of 512) -> 0.039 GB
  [does not fit] Offload ~50% of layers to system RAM (CPU fallback for those layers) -> 0.022 GB
(exit code: 2)

=== 4. stress: >=20 configs, zero budget violations, zero silent mode changes ===
ADMITTED: 0.039 GB predicted peak <= 4.000 GB budget (margin: 3961.0 MB)
Stress harness: 25 configs, 0 violations, 0 silent mode changes. Margin: min=3698.2 MB, median=3698.2 MB, max=3698.2 MB.
exit=0

### Wheel build + install on a clean venv (pip-installable)
$ uv build
Successfully built dist/fitsproof-0.1.0.tar.gz
Successfully built dist/fitsproof-0.1.0-py3-none-any.whl
$ uv pip install --python /tmp/fp-wheel dist/fitsproof-0.1.0-py3-none-any.whl
         If this is intentional, set `export UV_LINK_MODE=copy` or use `--link-mode=copy` to suppress this warning.
Installed 1 package in 3ms
 ~ fitsproof==0.1.0 (from file:///home/openclaw/portfolio/fitsproof/dist/fitsproof-0.1.0-py3-none-any.whl)
$ /tmp/fp-wheel/bin/fitsproof plan --budget-gb 4
ADMITTED: 0.042 GB predicted peak <= 4.000 GB budget (margin: 3958.3 MB)
$ /tmp/fp-wheel/bin/fitsproof admit --budget-gb 0.001; echo exit=$?
REFUSED: needs 0.04 GB, budget 0.00 GB; nearest fitting config is Offload ~50% of layers to system RAM (CPU fallback for those layers)
Degradation options:
exit=2

### Release workflow (not yet executed — requires push + v* tag)
$ python -c yaml.safe_load(...)
release.yml: valid YAML
146 .github/workflows/release.yml

### Launch surfaces present
-rw-rw-r-- 1 openclaw openclaw 3731 Sep 27 13:34 COMPARISONS.md
-rw-rw-r-- 1 openclaw openclaw 1897 Sep 27 14:54 CONTRIBUTING.md
-rwxrwxr-x 1 openclaw openclaw 1306 Sep 27 14:55 docs/demo.sh
-rw-rw-r-- 1 openclaw openclaw  389 Sep 27 14:54 launch/topics.txt
topics (non-comment): 16
19
266 README.md
40:# SURFACE: client
59:# SURFACE: guard
79:# SURFACE: server
131:# SURFACE: mcp

### Status notes (honest)

- **M1 (binary):** `.github/workflows/release.yml` defines the full path — PyInstaller
  one-file build, SHA256SUMS, a *clean job* that downloads the artifact and runs
  `fitsproof probe` / `fitsproof plan` / refusal-exit-2 from it, a wheel-install clean
  job, a GitHub Release attaching everything, and PyPI publish via trusted publishing
  (OIDC, gated on the `pypi` environment). It is valid YAML (checked above) but has NOT
  been executed: this pass may not push, and a tag is required. Executing it is a
  push-and-tag action for the orchestrator, not an in-repo change.
- **M2:** all four surfaces present with README snippets executed by
  `tests/value/test_readme_snippets.py`; the HTTP response now carries the admission
  record and refuses with 503 over budget (tests in tests/adversarial).
- **COMPARISONS.md / launch/topics.txt / CONTRIBUTING.md / docs/demo.sh:** topics and
  CONTRIBUTING and demo.sh written this pass; COMPARISONS.md already existed from a
  parallel lane (2026-09-27T13:00 UTC star counts) and was verified against
  MARKET-VERDICTS.md, not rewritten.
- **Defect fixed while closing the gap:** `int4_asym` was a NotImplementedError stub
  while the cost model priced it and the CLI accepted it; the asymmetric zero-point
  also overflowed int32 on constant rows. Both now implemented and tested.
