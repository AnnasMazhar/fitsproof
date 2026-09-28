"""
Packaging and versioning coherence.

Faults detected:
  test_pyproject_version_matches_package_version:
      Fault: a release built from pyproject.toml carries a different version
      than fitsproof.__version__, so `pip show fitsproof` and
      `fitsproof --version` disagree — bug reports and release tags become
      untraceable.

  test_cli_reports_version:
      Fault: no way for a user to ask which version they are running;
      support and bug reports cannot be answered without it.

  test_calibration_demo_runs:
      Fault: scripts/calibration_demo.py is referenced in the README as the
      reproducible command for the prediction-accuracy benchmark. If it fails
      to run or omits the required fields (MAPE, bandwidth, CI) the README
      claim of "reproduce: python scripts/calibration_demo.py" is false.
      The test asserts all required output fields appear and the script
      exits 0.
"""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

from fitsproof import __version__

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_version_matches_package_version() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    assert pyproject["project"]["version"] == __version__, (
        f"pyproject version {pyproject['project']['version']!r} != "
        f"fitsproof.__version__ {__version__!r}"
    )


def test_cli_reports_version() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "fitsproof.cli", "--version"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, f"--version must exit 0, got {proc.returncode}: {proc.stderr}"
    assert __version__ in proc.stdout, f"--version must print {__version__}, got: {proc.stdout!r}"


def test_calibration_demo_runs() -> None:
    """
    The README references `python scripts/calibration_demo.py` as the
    reproducible calibration benchmark command. This test verifies that:
    - the script exits 0
    - all required output fields are present in stdout (bandwidth, MAPE, CI,
      n_train, n_held_out)
    A missing field means the README claim is not reproducible.
    """
    demo = REPO_ROOT / "scripts" / "calibration_demo.py"
    assert demo.exists(), f"calibration_demo.py not found at {demo}"
    proc = subprocess.run(
        [sys.executable, str(demo)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"calibration_demo.py must exit 0; got {proc.returncode}. stderr: {proc.stderr}"
    )
    out = proc.stdout
    for field in ("bandwidth:", "MAPE (held-out):", "CI (95%):", "n_train=", "n_held_out="):
        assert field in out, (
            f"calibration_demo.py output missing required field {field!r}. Full output:\n{out}"
        )
