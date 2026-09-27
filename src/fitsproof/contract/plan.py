"""
fitsproof.contract.plan — Resource plan: predict and classify configurations.

Given (model, quant, context, machine, budget), computes a Plan describing
whether the configuration fits, fits with degradation, or does not fit, and
what the predicted peak memory usage is.

Degradations are an ordered list (cheapest first):
  1. Lower quantisation (int8 → int4)
  2. Shorter context window
  3. Offload layers to system RAM

Each degradation step comes with its predicted peak bytes and tok/s penalty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from fitsproof.contract.cost import estimate, kv_cache_bytes, weight_bytes
from fitsproof.contract.probe import MachineProfile
from fitsproof.engine.model import ModelConfig


class Verdict(str, Enum):
    FITS = "fits"
    FITS_WITH_DEGRADATION = "fits_with_degradation"
    DOES_NOT_FIT = "does_not_fit"


QUANT_ORDER = ["none", "int8_sym", "int4_sym"]  # increasingly aggressive


@dataclass
class DegradationStep:
    """
    One concrete degradation option with its predicted resource impact.

    Fault detected: if a degradation step is listed but actually exceeds
    the budget, admit.py would accept a plan that violates the contract.
    This is checked in test_plan.py and test_admit.py.
    """

    kind: Literal["lower_quant", "shorter_context", "offload_layers"]
    description: str
    predicted_peak_bytes: int
    predicted_tok_s: float
    fits_budget: bool


@dataclass
class Plan:
    """
    Full resource plan for a given configuration.

    predicted_peak_bytes: best estimate of peak memory usage.
    predicted_peak_ci: (lower, upper) 95% confidence interval (filled by calibrate).
    verdict: fits | fits_with_degradation | does_not_fit.
    degradations: ordered list of alternatives if the base config doesn't fit.
    binding_constraint: human-readable string naming what would be violated.
    """

    verdict: Verdict
    predicted_peak_bytes: int
    predicted_peak_ci: tuple[int, int]
    predicted_tok_s: float
    predicted_tok_s_ci: tuple[float, float]
    budget_bytes: int
    quant: str
    context_len: int
    degradations: list[DegradationStep] = field(default_factory=list)
    binding_constraint: str = ""


def plan(
    cfg: ModelConfig,
    machine: MachineProfile,
    context_len: int,
    budget_bytes: int,
    quant: str = "none",
    bandwidth_utilisation: float = 0.6,
) -> Plan:
    """
    Produce a resource plan for (cfg, machine, context_len, budget_bytes, quant).

    Algorithm:
      1. Compute predicted_peak_bytes for the requested config.
      2. If it fits, return Verdict.FITS.
      3. Otherwise, enumerate degradations in ascending cost order and
         find the first one that fits.
      4. If no degradation fits, return Verdict.DOES_NOT_FIT with the
         binding constraint named.

    Fault detected: if predicted_peak_bytes is computed with the wrong
    formula (e.g. omitting KV cache), the plan will give FITS for a config
    that actually OOMs; tested with configs chosen to exceed the budget
    only via the KV cache term.
    """
    cost = estimate(cfg, machine, context_len, quant, bandwidth_utilisation)
    predicted_peak = cost.total_peak_bytes
    predicted_tok_s = cost.predicted_tok_s

    # Simple CI: ±20% of predicted (replaced by fitted CI when calibrate is used)
    peak_ci = (int(predicted_peak * 0.8), int(predicted_peak * 1.2))
    tok_s_ci = (predicted_tok_s * 0.7, predicted_tok_s * 1.3)

    if predicted_peak <= budget_bytes:
        return Plan(
            verdict=Verdict.FITS,
            predicted_peak_bytes=predicted_peak,
            predicted_peak_ci=peak_ci,
            predicted_tok_s=predicted_tok_s,
            predicted_tok_s_ci=tok_s_ci,
            budget_bytes=budget_bytes,
            quant=quant,
            context_len=context_len,
        )

    # Try degradations
    degradations: list[DegradationStep] = []

    # 1. Lower quantisation
    current_quant_idx = QUANT_ORDER.index(quant) if quant in QUANT_ORDER else 0
    for q in QUANT_ORDER[current_quant_idx + 1 :]:
        dq_cost = estimate(cfg, machine, context_len, q, bandwidth_utilisation)
        fits = dq_cost.total_peak_bytes <= budget_bytes
        degradations.append(
            DegradationStep(
                kind="lower_quant",
                description=f"Use {q} quantisation instead of {quant}",
                predicted_peak_bytes=dq_cost.total_peak_bytes,
                predicted_tok_s=dq_cost.predicted_tok_s,
                fits_budget=fits,
            )
        )

    # 2. Shorter context (halve context, then quarter)
    for divisor in [2, 4, 8]:
        shorter = max(1, context_len // divisor)
        sc_cost = estimate(cfg, machine, shorter, quant, bandwidth_utilisation)
        fits = sc_cost.total_peak_bytes <= budget_bytes
        degradations.append(
            DegradationStep(
                kind="shorter_context",
                description=f"Reduce context to {shorter} tokens (1/{divisor} of {context_len})",
                predicted_peak_bytes=sc_cost.total_peak_bytes,
                predicted_tok_s=sc_cost.predicted_tok_s,
                fits_budget=fits,
            )
        )

    # 3. Offload some layers (reduce effective in-memory weight cost)
    # Simplified model: offloading half the layers saves ~50% of layer weight bytes
    w = weight_bytes(cfg, quant)
    kv = kv_cache_bytes(cfg, context_len, quant)
    from fitsproof.contract.cost import activation_bytes

    act = activation_bytes(cfg)
    # Offload half layers: saves ~50% of attention+FFN weight bytes
    offload_saving_factor = 0.5
    offloaded_peak = int(w * offload_saving_factor + kv + act)
    fits_offload = offloaded_peak <= budget_bytes
    degradations.append(
        DegradationStep(
            kind="offload_layers",
            description="Offload ~50% of layers to system RAM (CPU fallback for those layers)",
            predicted_peak_bytes=offloaded_peak,
            predicted_tok_s=predicted_tok_s * 0.3,  # significant tok/s penalty
            fits_budget=fits_offload,
        )
    )

    # Find the first degradation that fits
    fitting_degradation = next((d for d in degradations if d.fits_budget), None)

    if fitting_degradation is not None:
        verdict = Verdict.FITS_WITH_DEGRADATION
        binding = ""
    else:
        verdict = Verdict.DOES_NOT_FIT
        binding = (
            f"needs {predicted_peak / 1e9:.2f} GB, budget {budget_bytes / 1e9:.2f} GB; "
            f"nearest fitting config is "
            + (f"{degradations[-1].description}" if degradations else "none found")
        )

    return Plan(
        verdict=verdict,
        predicted_peak_bytes=predicted_peak,
        predicted_peak_ci=peak_ci,
        predicted_tok_s=predicted_tok_s,
        predicted_tok_s_ci=tok_s_ci,
        budget_bytes=budget_bytes,
        quant=quant,
        context_len=context_len,
        degradations=degradations,
        binding_constraint=binding,
    )
