"""
fitsproof.contract.admit — The enforcement point.

Given a Plan, either:
  (a) admits it unchanged (Plan.verdict == FITS),
  (b) applies the cheapest fitting degradation and emits a DegradedRecord, or
  (c) refuses with an explicit message naming the binding constraint.

INVARIANT: there is no code path that changes execution mode without
emitting a record. This is the central guarantee of fitsproof.

Every mode change goes through admit(); callers must not bypass it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from fitsproof.contract.plan import DegradationStep, Plan, Verdict


class AdmitStatus(str, Enum):
    ADMITTED = "admitted"
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

    Fault detected: the DOES_NOT_FIT branch must emit REFUSED, not DEGRADED;
    a bug that emits DEGRADED with None degradation would pass the config
    silently and allow an OOM.
    """
    if plan.verdict == Verdict.FITS:
        return AdmitRecord(
            status=AdmitStatus.ADMITTED,
            plan=plan,
            applied_degradation=None,
            refusal_reason="",
            message=(
                f"ADMITTED: {plan.predicted_peak_bytes / 1e9:.3f} GB predicted peak "
                f"<= {plan.budget_bytes / 1e9:.3f} GB budget "
                f"(margin: {(plan.budget_bytes - plan.predicted_peak_bytes) / 1e6:.1f} MB)"
            ),
        )

    elif plan.verdict == Verdict.FITS_WITH_DEGRADATION:
        # Find the cheapest degradation that fits
        fitting = next((d for d in plan.degradations if d.fits_budget), None)
        if fitting is None:
            # Defensive: verdict says degradation exists but none fit — refuse
            return AdmitRecord(
                status=AdmitStatus.REFUSED,
                plan=plan,
                applied_degradation=None,
                refusal_reason=(
                    "Internal inconsistency: verdict=FITS_WITH_DEGRADATION but no "
                    "degradation fits the budget. Refusing for safety."
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
            message=(
                f"REFUSED: {plan.binding_constraint}"
            ),
        )
