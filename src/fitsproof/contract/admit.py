"""
fitsproof.contract.admit — The enforcement point.

Given a Plan, either:
  (a) admits it unchanged (Plan.verdict == FITS, margin >= SAFETY_MARGIN_BYTES),
  (b) admits with a boundary warning (Plan.verdict == FITS, margin < SAFETY_MARGIN_BYTES),
  (c) applies the cheapest fitting degradation and emits a DegradedRecord, or
  (d) refuses with an explicit message naming the binding constraint.

SAFETY MARGIN: When predicted_peak is within SAFETY_MARGIN_BYTES of budget, the
admission is flagged as NEAR_BOUNDARY.  The proof harness may still pass or fail
depending on measurement noise, but the caller is warned that the margin is small
and the tool cannot guarantee the budget will be respected at runtime.  The CLI
exits non-zero (exit 1) on NEAR_BOUNDARY so the build will fail unless the caller
explicitly handles the warning.

INVARIANT: there is no code path that changes execution mode without
emitting a record. This is the central guarantee of fitsproof.

Every mode change goes through admit(); callers must not bypass it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from fitsproof.contract.plan import DegradationStep, Plan, Verdict

# When the predicted peak is within this many bytes of the declared budget,
# admit() returns NEAR_BOUNDARY instead of ADMITTED.  The value is chosen to
# cover a 25% prediction overrun on a typical small model (~50 MB weights):
#   50 MB * 1.25 = 62.5 MB overshoot if the predictor underestimates by 25%.
# Any budget whose margin is smaller than this value cannot be guaranteed by
# the predictor alone; the proof harness must be run to confirm.
SAFETY_MARGIN_BYTES: int = 50 * 1024 * 1024  # 50 MB


class AdmitStatus(str, Enum):
    ADMITTED = "admitted"
    NEAR_BOUNDARY = "near_boundary"
    DEGRADED = "degraded"
    REFUSED = "refused"


@dataclass
class AdmitRecord:
    """
    Record of an admission decision. Always emitted; never suppressed.

    Fault detected: if admit() can return without producing a record,
    a silent mode change can occur. The test injects a plan that triggers
    degradation and asserts the returned record is not None.
    """

    status: AdmitStatus
    plan: Plan
    applied_degradation: DegradationStep | None
    refusal_reason: str
    message: str


def admit(plan: Plan) -> AdmitRecord:
    """
    Enforce the resource contract defined by *plan*.

    Returns an AdmitRecord in all cases. The record must be inspected
    by the caller; silent discarding of a REFUSED record is a contract violation.

    Logic:
      - FITS:                   → AdmitRecord(ADMITTED, no degradation)
      - FITS_WITH_DEGRADATION:  → AdmitRecord(DEGRADED, cheapest fitting degradation)
      - DOES_NOT_FIT:           → AdmitRecord(REFUSED, binding_constraint)

    Security: admit() does NOT trust plan.verdict blindly. It re-validates
    that predicted_peak_bytes <= budget_bytes before admitting. A Plan
    constructed outside plan() with a lying verdict will be refused.

    Fault detected: the DOES_NOT_FIT branch must emit REFUSED, not DEGRADED;
    a bug that emits DEGRADED with None degradation would pass the config
    silently and allow an OOM.
    """
    # SECURITY: Re-validate regardless of verdict — do not trust external Plans
    if plan.predicted_peak_bytes > plan.budget_bytes:
        # The plan claims FITS but the numbers don't add up — refuse
        if plan.verdict == Verdict.FITS:
            return AdmitRecord(
                status=AdmitStatus.REFUSED,
                plan=plan,
                applied_degradation=None,
                refusal_reason=(
                    f"Plan inconsistency: verdict=FITS but predicted "
                    f"({plan.predicted_peak_bytes / 1e9:.3f} GB) > budget "
                    f"({plan.budget_bytes / 1e9:.3f} GB). Refusing for safety."
                ),
                message=(
                    f"REFUSED (inconsistent plan): predicted "
                    f"{plan.predicted_peak_bytes / 1e9:.3f} GB > budget "
                    f"{plan.budget_bytes / 1e9:.3f} GB"
                ),
            )

    if plan.verdict == Verdict.FITS:
        margin = plan.budget_bytes - plan.predicted_peak_bytes
        if margin < SAFETY_MARGIN_BYTES:
            return AdmitRecord(
                status=AdmitStatus.NEAR_BOUNDARY,
                plan=plan,
                applied_degradation=None,
                refusal_reason="",
                message=(
                    f"WARNING (near boundary): {plan.predicted_peak_bytes / 1e9:.3f} GB "
                    f"predicted peak <= {plan.budget_bytes / 1e9:.3f} GB budget "
                    f"(margin: {margin / 1e6:.1f} MB < safety margin: "
                    f"{SAFETY_MARGIN_BYTES / 1e6:.0f} MB). "
                    "Prediction error may exceed the remaining margin. "
                    "Run `fitsproof verify` to measure actual RSS, or increase the budget."
                ),
            )
        return AdmitRecord(
            status=AdmitStatus.ADMITTED,
            plan=plan,
            applied_degradation=None,
            refusal_reason="",
            message=(
                f"ADMITTED: {plan.predicted_peak_bytes / 1e9:.3f} GB predicted peak "
                f"<= {plan.budget_bytes / 1e9:.3f} GB budget "
                f"(margin: {margin / 1e6:.1f} MB)"
            ),
        )

    elif plan.verdict is Verdict.FITS_WITH_DEGRADATION:
        # Find the cheapest degradation that actually fits (re-validate fits_budget)
        # SECURITY: Do not trust fits_budget flag — verify predicted <= budget
        fitting = next(
            (
                d
                for d in plan.degradations
                if d.fits_budget and d.predicted_peak_bytes <= plan.budget_bytes
            ),
            None,
        )
        if fitting is None:
            # Either no degradation exists, or all marked fits_budget are lying
            return AdmitRecord(
                status=AdmitStatus.REFUSED,
                plan=plan,
                applied_degradation=None,
                refusal_reason=(
                    "Internal inconsistency: verdict=FITS_WITH_DEGRADATION but no "
                    "degradation actually fits the budget. Refusing for safety."
                ),
                message=(
                    f"REFUSED (internal inconsistency): "
                    f"budget={plan.budget_bytes / 1e9:.3f} GB, "
                    f"predicted={plan.predicted_peak_bytes / 1e9:.3f} GB"
                ),
            )
        return AdmitRecord(
            status=AdmitStatus.DEGRADED,
            plan=plan,
            applied_degradation=fitting,
            refusal_reason="",
            message=(
                f"DEGRADED: base config needs {plan.predicted_peak_bytes / 1e9:.3f} GB "
                f"> budget {plan.budget_bytes / 1e9:.3f} GB. "
                f"Applying: {fitting.description}. "
                f"New predicted peak: {fitting.predicted_peak_bytes / 1e9:.3f} GB."
            ),
        )

    else:  # DOES_NOT_FIT
        return AdmitRecord(
            status=AdmitStatus.REFUSED,
            plan=plan,
            applied_degradation=None,
            refusal_reason=plan.binding_constraint,
            message=(f"REFUSED: {plan.binding_constraint}"),
        )
