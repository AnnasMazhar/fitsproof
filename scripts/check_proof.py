#!/usr/bin/env python3
"""check_proof.py — validate that fitsproof satisfies its acceptance criteria.

The repo's proof is:
1. All tests pass (pytest -q exits 0)
2. The stress harness runs with 0 violations, 0 silent mode changes
3. An admit command can refuse a configuration that exceeds budget
4. EVIDENCE.md exists with documented proof output

This checker is falsifiable: removing EVIDENCE.md, breaking a test, or making the
stress harness report violations will cause the checker to fail and print which
check broke.

Exits 0 and prints PROOF_COMPLETE on success.
Exits 1 with a clear message on any failure.

Usage:
    python scripts/check_proof.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EVIDENCE_MD = _REPO_ROOT / "docs" / "EVIDENCE.md"
_VENV_PYTEST = _REPO_ROOT / ".venv" / "bin" / "pytest"
_VENV_FITSPROOF = _REPO_ROOT / ".venv" / "bin" / "fitsproof"

MIN_TEST_COUNT = 150  # Must have at least this many tests
MIN_STRESS_CONFIGS = 20  # Stress harness must test at least this many configs


def check_evidence_exists() -> list[str]:
    """Check that docs/EVIDENCE.md exists and is non-empty."""
    if not _EVIDENCE_MD.exists():
        return [f"MISSING: {_EVIDENCE_MD.relative_to(_REPO_ROOT)} does not exist"]
    if _EVIDENCE_MD.stat().st_size < 1000:
        return [f"INCOMPLETE: {_EVIDENCE_MD.relative_to(_REPO_ROOT)} is too small (< 1000 bytes)"]
    return []


def check_venv_exists() -> list[str]:
    """Check that the venv and required tools exist."""
    fails = []
    if not _VENV_PYTEST.exists():
        fails.append(f"MISSING: {_VENV_PYTEST} — run: uv venv && uv pip install -e '.[dev]'")
    if not _VENV_FITSPROOF.exists():
        fails.append(f"MISSING: {_VENV_FITSPROOF} — run: uv venv && uv pip install -e '.[dev]'")
    return fails


def check_tests_pass() -> list[str]:
    """Run pytest and verify all tests pass with at least MIN_TEST_COUNT tests."""
    if not _VENV_PYTEST.exists():
        return ["SKIP: pytest not installed (venv missing)"]
    
    result = subprocess.run(
        [str(_VENV_PYTEST), "-q", "--tb=no"],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        timeout=600,  # 10 min timeout for the full suite
    )
    
    if result.returncode != 0:
        lines = (result.stdout + result.stderr).strip().splitlines()
        summary = [l for l in lines if "passed" in l.lower() or "failed" in l.lower()]
        return [f"TESTS FAILED: pytest exited {result.returncode}"] + summary[-3:]
    
    # Parse test count from output (e.g., "235 passed in 371.93s")
    output = result.stdout + result.stderr
    match = re.search(r"(\d+)\s+passed", output)
    if match:
        count = int(match.group(1))
        if count < MIN_TEST_COUNT:
            return [f"INSUFFICIENT TESTS: {count} < {MIN_TEST_COUNT} required"]
    
    return []


def check_stress_harness() -> list[str]:
    """Run the stress harness and verify 0 violations, 0 silent mode changes."""
    if not _VENV_FITSPROOF.exists():
        return ["SKIP: fitsproof not installed (venv missing)"]
    
    result = subprocess.run(
        [str(_VENV_FITSPROOF), "stress"],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        timeout=300,  # 5 min timeout
    )
    
    output = result.stdout + result.stderr
    
    if result.returncode != 0:
        lines = output.strip().splitlines()
        return [f"STRESS FAILED: fitsproof stress exited {result.returncode}"] + lines[-5:]
    
    # Parse the output for violations and mode changes
    # Expected: "Stress harness: 25 configs, 0 violations, 0 silent mode changes."
    match = re.search(r"(\d+)\s+configs,\s+(\d+)\s+violations?,\s+(\d+)\s+silent", output)
    if not match:
        return ["STRESS PARSE ERROR: could not find config/violation counts in output"]
    
    configs = int(match.group(1))
    violations = int(match.group(2))
    silent_changes = int(match.group(3))
    
    fails = []
    if configs < MIN_STRESS_CONFIGS:
        fails.append(f"INSUFFICIENT STRESS CONFIGS: {configs} < {MIN_STRESS_CONFIGS} required")
    if violations > 0:
        fails.append(f"STRESS VIOLATIONS: {violations} budget violations detected")
    if silent_changes > 0:
        fails.append(f"SILENT MODE CHANGES: {silent_changes} silent mode changes detected")
    
    return fails


def check_admit_refuses() -> list[str]:
    """Verify that admit can refuse a configuration that exceeds budget."""
    if not _VENV_FITSPROOF.exists():
        return ["SKIP: fitsproof not installed (venv missing)"]
    
    # Try to admit with a very small budget that should be refused
    result = subprocess.run(
        [str(_VENV_FITSPROOF), "admit", "--budget-gb", "0.001"],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )
    
    # Should exit 2 (refused) or 1 (error), not 0 (admitted)
    output = result.stdout + result.stderr
    if result.returncode == 0:
        return ["ADMIT SHOULD REFUSE: admit with 0.001 GB budget should not succeed"]
    
    # Check that it says REFUSED, not just errored out
    if "REFUSED" not in output.upper() and result.returncode != 2:
        # Might be an error, not a refusal
        if result.returncode == 1 and "budget" not in output.lower():
            return [f"ADMIT UNCLEAR: exited {result.returncode} but no refusal message"]
    
    return []


def main() -> int:
    all_failures: list[str] = []
    
    print("check_proof: validating fitsproof proof...")
    
    # Quick checks first
    print("  [1/5] Checking docs/EVIDENCE.md exists...")
    all_failures.extend(check_evidence_exists())
    
    print("  [2/5] Checking venv exists...")
    all_failures.extend(check_venv_exists())
    
    # If venv is missing, skip the rest
    if any("venv missing" in f or "uv venv" in f for f in all_failures):
        print("check_proof: FAIL (venv not set up)")
        for f in all_failures:
            print(f"  {f}")
        return 1
    
    print("  [3/5] Running pytest (this may take several minutes)...")
    all_failures.extend(check_tests_pass())
    
    print("  [4/5] Running stress harness...")
    all_failures.extend(check_stress_harness())
    
    print("  [5/5] Verifying admit can refuse over-budget configs...")
    all_failures.extend(check_admit_refuses())
    
    if all_failures:
        print("check_proof: FAIL")
        for f in all_failures:
            print(f"  {f}")
        return 1
    
    print("check_proof: all checks passed.")
    print("PROOF_COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
