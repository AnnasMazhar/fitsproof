# ADVERSARIAL REVIEW — fitsproof

Independent review across two passes:
- Pass 1 (`c4-p10-adversarial-1`): Claims audit, citation audit, fault injection
- Pass 2 (`c4-p11-adversarial-2`): Attack the property — defeat the safety contract

Reviewer lane: kiro:claude-opus-4.5.

---

## Pass 1 — Claims Audit and Citation Verification

Baseline:
```
$ pytest tests/ -q --tb=no
189 passed in 138.63s
```

### 1.1 Claims Audit — The 3 Most Load-Bearing README Claims

**C1 (stress harness):** Verified — 25 configs, exit 0 on pass, exit 1 on violation.

**C2 (refusal with binding constraint):** Verified — exit 2, constraint named.

**C3 (MAPE):** Minor finding — observed 63.8% vs documented ~46–62% (ADV-01).


### 1.2 Citation Audit

15 critical URLs tested. All resolve (1 ACM DOI bot-blocked but DOI-verified).
Claims match sources on spot-check of 5 design-driving papers.


### 1.3 Fault Injection (6 tests sampled)

- 3 tests detected injected faults correctly
- 3 tests showed self-consistency patterns (compute expected from implementation)

---

## Pass 2 — Attack the Property

The core property is: **"the contract must refuse loudly or degrade explicitly
— never allow a silent OOM or mode change."**

### Attack 1 — Bypass admit() by constructing a lying Plan

**Attack vector:** Construct a `Plan` with `verdict=FITS` but `predicted > budget`.

```
$ python -c "
from fitsproof.contract.plan import Plan, Verdict
from fitsproof.contract.admit import admit

malicious_plan = Plan(
    verdict=Verdict.FITS,  # Lie
    predicted_peak_bytes=8_000_000_000,  # 8 GB
    predicted_peak_ci=(7_500_000_000, 8_500_000_000),
    predicted_tok_s=10.0,
    predicted_tok_s_ci=(5.0, 15.0),
    budget_bytes=4_000_000_000,  # 4 GB
    quant='none',
    context_len=512,
    degradations=[],
    binding_constraint='',
)
record = admit(malicious_plan)
print(f'Status: {record.status}')
print(f'Message: {record.message}')
"

# BEFORE FIX:
Status: AdmitStatus.ADMITTED
Message: ADMITTED: 8.000 GB predicted peak <= 4.000 GB budget (margin: -4000.0 MB)
>>> ATTACK SUCCEEDED: admitted despite predicted > budget!

# AFTER FIX:
Status: AdmitStatus.REFUSED
Message: REFUSED (inconsistent plan): predicted 8.000 GB > budget 4.000 GB
>>> FIX CONFIRMED: attack blocked
```

**Finding ADV-05 (blocker):** `admit()` trusted `plan.verdict` without validating
`predicted_peak_bytes <= budget_bytes`. **FIXED.**


### Attack 2 — Guard decorator bypass attempt

```
$ python -c "
from fitsproof.client import guard, DoesNotFit
called = False
@guard(budget='1MiB')
def load_model():
    global called
    called = True
try:
    load_model()
except DoesNotFit:
    print(f'called={called}')
"
called=False
>>> Attack failed: Guard prevented the call
```


### Attack 3 — Lying degradation.fits_budget flag

```
$ python -c "
from fitsproof.contract.plan import Plan, Verdict, DegradationStep
from fitsproof.contract.admit import admit

bad_degradation = DegradationStep(
    kind='lower_quant',
    description='int4 (malicious)',
    predicted_peak_bytes=10_000_000_000,  # 10 GB
    predicted_tok_s=50.0,
    fits_budget=True,  # Lie
)
malicious_plan = Plan(
    verdict=Verdict.FITS_WITH_DEGRADATION,
    predicted_peak_bytes=15_000_000_000,
    predicted_peak_ci=(14_000_000_000, 16_000_000_000),
    predicted_tok_s=5.0,
    predicted_tok_s_ci=(4.0, 6.0),
    budget_bytes=4_000_000_000,
    quant='none',
    context_len=512,
    degradations=[bad_degradation],
    binding_constraint='',
)
record = admit(malicious_plan)
print(f'Status: {record.status}')
"

# BEFORE FIX:
Status: AdmitStatus.DEGRADED
>>> ATTACK SUCCEEDED: accepted degradation where predicted > budget!

# AFTER FIX:
Status: AdmitStatus.REFUSED
>>> FIX CONFIRMED: lying degradation blocked
```

**Finding ADV-06 (blocker):** `admit()` trusted `fits_budget` without verifying
`degradation.predicted_peak_bytes <= budget`. **FIXED.**


### Attack 4 — Server with impossible budget

```
HTTP 503: Service Unavailable
fitsproof_refused
>>> Attack failed: server refused with proper error code
```


### Attack 5 — MCP injection with negative budget

```
{'isError': True} — negative budget rejected
>>> Attack failed
```


### Attack 6 — Determinism attack

```
Identical: True
>>> Attack failed: greedy determinism holds
```


### Attack 7 — Integer overflow

Python handles arbitrary precision integers gracefully.


### Attack 8 — plan() internal consistency

```
plan() is internally consistent
```

---

## Fixes Applied

### Fix 1: admit() validates predicted <= budget regardless of verdict

**File:** `src/fitsproof/contract/admit.py`

Added validation at the start of `admit()`:
```python
# SECURITY: Re-validate regardless of verdict — do not trust external Plans
if plan.predicted_peak_bytes > plan.budget_bytes:
    if plan.verdict == Verdict.FITS:
        return AdmitRecord(status=AdmitStatus.REFUSED, ...)
```


### Fix 2: admit() validates degradation.predicted_peak_bytes <= budget

Changed degradation filter:
```python
# SECURITY: Do not trust fits_budget flag — verify predicted <= budget
fitting = next(
    (d for d in plan.degradations
     if d.fits_budget and d.predicted_peak_bytes <= plan.budget_bytes),
    None,
)
```


### Tests Added

`tests/contract/test_plan_admit_verify.py::TestAdmitTrustBoundary`:
- `test_lying_verdict_fits_rejected`
- `test_lying_degradation_fits_budget_rejected`
- `test_honest_plan_still_admitted`

---

## Findings Table

| ID | Severity | Finding | Evidence | Status |
|----|----------|---------|----------|--------|
| ADV-05 | blocker | admit() trusted verdict without validating predicted <= budget | Attack 1 | **fixed** |
| ADV-06 | blocker | admit() trusted fits_budget without validating degradation predicted <= budget | Attack 3 | **fixed** |
| ADV-01 | minor | MAPE variance exceeds documented range | 63.8% vs ~46–62% | open |
| ADV-02 | minor | Several KATs compute expected from implementation constants | Self-consistency | limitation |
| ADV-03 | N/A | silent_mode_changes counter hardcoded False | c1-p10 | limitation |
| ADV-04 | N/A | RSS measurement is process-lifetime HWM | c1-p10 | limitation |

---

## Verification After Fixes

```
$ pytest tests/ -q --tb=no
192 passed in 161.96s

$ ruff check . && ruff format --check .
All checks passed!
```

Both attack vectors now blocked:
- Attack 1: `REFUSED (inconsistent plan): predicted 8.000 GB > budget 4.000 GB`
- Attack 3: `REFUSED (internal inconsistency): budget=4.000 GB, predicted=15.000 GB`

---

## Summary

**Attacks attempted:** 8
**Attacks succeeded (before fix):** 2 (blocker severity)
**Attacks failed:** 6

**Blockers fixed:** 2 (ADV-05, ADV-06)
**Remaining findings:** 4 (2 minor, 2 documented limitations)

The core safety property is now robust against external Plan injection attacks.
All blockers fixed with tests. Pass `c4-p11-adversarial-2` complete.
