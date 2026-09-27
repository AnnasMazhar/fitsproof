"""
fitsproof.client — Python client and guard decorator for the resource contract.

Provides:
  - DoesNotFit: raised by @guard and by FitsproofClient.admit() on refusal.
  - FitsproofClient: in-process client wrapping plan/admit/probe/metrics.
  - guard(budget, ...): decorator that enforces the contract before the
    wrapped callable allocates.

Integration pattern (M2 from v0.2 MANDATE):
  from fitsproof.client import FitsproofClient, guard, DoesNotFit

  client = FitsproofClient()
  plan = client.plan(context_len=512, budget_bytes=4 * 1024**3)
  record = client.admit(plan)

  @guard(budget="6GiB")
  def load_model():
      ...  # never called if config does not fit
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

from fitsproof.contract.admit import AdmitRecord, AdmitStatus
from fitsproof.contract.admit import admit as _admit
from fitsproof.contract.plan import Plan
from fitsproof.contract.plan import plan as _plan
from fitsproof.contract.probe import MachineProfile
from fitsproof.contract.probe import probe as _probe
from fitsproof.engine.model import REFERENCE_CONFIG, ModelConfig


class DoesNotFit(RuntimeError):  # noqa: N818
    """
    Raised when a configuration is refused by the resource contract.

    N818 suppressed: DoesNotFit is the domain term; "DoesNotFitError" is less clear.

    Fault detected: a guard that raises a different exception type (or does
    not raise at all) would silently allow the wrapped callable to allocate
    more memory than the declared budget.
    """

    def __init__(self, record: AdmitRecord) -> None:
        self.record = record
        super().__init__(record.message)


def _parse_budget(budget: str | int | float) -> int:
    """
    Parse a human-readable budget string or numeric bytes into integer bytes.

    Accepts: int/float bytes, or strings like "4GiB", "512MiB", "4GB", "512MB".

    Fault detected: a parser that silently truncates (e.g. "4GiB" → 4) would
    pass an impossibly small budget and refuse every valid configuration.
    """
    if isinstance(budget, int | float):
        return int(budget)
    s = budget.strip()
    for suffix, multiplier in [
        ("GiB", 1024**3),
        ("MiB", 1024**2),
        ("KiB", 1024),
        ("GB", 10**9),
        ("MB", 10**6),
        ("KB", 10**3),
    ]:
        if s.endswith(suffix):
            return int(float(s[: -len(suffix)]) * multiplier)
    return int(s)  # bare integer string


class FitsproofClient:
    """
    In-process client for the fitsproof resource contract.

    Wraps probe/plan/admit to give callers a clean API without importing
    internal modules directly. The server variant (HTTP client for
    fitsproof serve) is separate; this variant is in-process only.

    Fault detected: a client that caches a stale MachineProfile and uses
    it for plans after hardware changes would mis-plan. Re-probe on demand
    by passing machine=None.
    """

    def __init__(self, model_cfg: ModelConfig | None = None) -> None:
        self._cfg = model_cfg or REFERENCE_CONFIG
        self._machine: MachineProfile | None = None

    @property
    def machine(self) -> MachineProfile:
        """Return cached MachineProfile, probing once on first access."""
        if self._machine is None:
            self._machine = _probe()
        return self._machine

    def reprobe(self) -> MachineProfile:
        """Force a fresh machine probe and update the cache."""
        self._machine = _probe()
        return self._machine

    def plan(
        self,
        context_len: int = 512,
        budget_bytes: int | str = 4 * 1024**3,
        quant: str = "none",
    ) -> Plan:
        """
        Return a Plan for (context_len, budget_bytes, quant) on this machine.

        budget_bytes may be an integer (bytes) or a string like "4GiB".
        """
        budget = _parse_budget(budget_bytes)
        return _plan(self._cfg, self.machine, context_len, budget, quant)

    def admit(self, p: Plan) -> AdmitRecord:
        """
        Enforce the Plan. Returns an AdmitRecord; raises DoesNotFit if refused.

        Fault detected: a client.admit() that swallows REFUSED status and
        returns the record silently would allow callers to proceed past a
        refusal. This implementation raises unconditionally on REFUSED.
        """
        record = _admit(p)
        if record.status == AdmitStatus.REFUSED:
            raise DoesNotFit(record)
        return record

    def metrics(self) -> dict[str, Any]:
        """
        Return a summary dict of machine metrics from the current profile.

        Useful for agents querying the contract via the Python surface before
        deciding whether to attempt a load.
        """
        m = self.machine
        return {
            "memory_bandwidth_gb_s": m.memory_bandwidth_bps / 1e9,
            "gemm_throughput_gflops": m.gemm_throughput_flops / 1e9,
            "ram_gb": m.memory_bytes / 1e9,
            "vram_gb": m.gpu_memory_bytes / 1e9,
            "hostname": m.hostname,
        }


def guard(
    budget: str | int | float,
    context_len: int = 512,
    quant: str = "none",
    model_cfg: ModelConfig | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator: enforce the resource contract before the wrapped callable runs.

    Raises DoesNotFit **before** the wrapped callable is invoked, so the
    caller never allocates the model on a refusing configuration.

    Usage:
      @guard(budget="4GiB", context_len=1024)
      def load_and_run():
          ...

    Fault detected: a guard that calls the wrapped function before checking
    the contract would allow OOM on refused configs. The test asserts the
    wrapped callable is never invoked when the plan is refused.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            client = FitsproofClient(model_cfg)
            p = client.plan(context_len=context_len, budget_bytes=budget, quant=quant)
            record = _admit(p)
            if record.status == AdmitStatus.REFUSED:
                raise DoesNotFit(record)
            # record may be ADMITTED or DEGRADED — proceed
            return fn(*args, **kwargs)

        return wrapper

    return decorator
