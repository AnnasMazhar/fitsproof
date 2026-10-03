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

  test_calibration_demo_mape_is_parseable_float:
      Fault: ADV-01 (c4-p10) — MAPE line present in output but value outside
      the physically plausible range (0, 100). Catches:
        - broken calibration returning 0.0% (divide-by-zero or trivial fit)
        - negative MAPE from a sign error in the formula
        - NaN or "None" text if calibration raises and is silently swallowed
        - MAPE > 100% indicating a calculation bug (MAPE is a percentage of
          the true value; values above 100 are possible in principle but would
          represent a worse-than-random predictor on the reference fixtures)
      The documented range is ~30–65% across observed sessions (see F-3 in
      docs/ADOPTION.md). This test does NOT enforce that range — MAPE varies
      with machine load at measurement time. It enforces that the calibration
      produces a valid, non-trivial, finite result. The README's range claim
      is documented prose that must be updated when new observations land
      outside it (ADV-01 root cause).
"""

from __future__ import annotations

import re
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
    - when n_held_out=1, the output explains the degenerate CI (the point
      interval [X%, X%] that strangers mistake for a broken CI implementation)
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
    # When n_held_out=1 (the default for the in-repo 3-measurement run),
    # the output must explain that the CI is a degenerate point, not a range.
    # This prevents strangers from thinking [X%, X%] is a broken CI implementation.
    # We check for the word "degenerate" — which only appears when the annotation
    # is present, not from the bare `n_held_out=1` line itself.
    import re as _re

    held_match = _re.search(r"n_held_out=(\d+)", out)
    if held_match and int(held_match.group(1)) < 2:
        assert "degenerate" in out.lower(), (
            "When n_held_out=1, calibration_demo.py must explain the degenerate CI "
            "(the [X%, X%] point interval). "
            "A stranger seeing identical lower/upper bounds will think the CI is broken. "
            "Add a note like: 'n_held_out=1: degenerate interval — see docs/ADOPTION.md F-3'. "
            f"Full output:\n{out}"
        )


def test_calibration_demo_mape_is_parseable_float() -> None:
    """
    ADV-01 regression guard: MAPE must be a finite float in (0, 100).

    The adversarial reviewer (c4-p10) observed MAPE=63.8%, which exceeded
    the then-documented range of ~46-62%.  A follow-up session produced
    31.3%.  Root cause: the documented range was too narrow and the test
    only verified the label existed, not that the value was valid.

    This test extracts the numeric MAPE value and asserts it is:
      - parseable as a float (not "None", "nan", "N/A", or missing)
      - strictly positive (0.0 would mean the calibration returned a
        trivially perfect fit — impossible on the reference fixtures whose
        held-out config differs from the training configs)
      - less than 100 (MAPE above 100 indicates a calculation error; the
        reference fixtures are not that far apart in scale)

    The test does NOT enforce the documented range (~30-65%) because MAPE
    varies with machine load at measurement time (see docs/ADOPTION.md F-3).
    The prose range must be updated when new observations land outside it;
    this test guards against a broken calibration script, not a narrow range.

    Fault injections that this test kills:
      - divide-by-zero in _mape returning 0.0 or NaN
      - sign error returning a negative percentage
      - calibration silently returning None (printed as "None")
      - formula inversion producing MAPE >> 100
    """
    demo = REPO_ROOT / "scripts" / "calibration_demo.py"
    proc = subprocess.run(
        [sys.executable, str(demo)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"calibration_demo.py exited {proc.returncode}: {proc.stderr}"
    # Parse: "MAPE (held-out):       46.1%"
    match = re.search(r"MAPE \(held-out\):\s+([\d.]+)%", proc.stdout)
    assert match is not None, (
        f"Could not parse MAPE value from calibration_demo.py output.\nOutput was:\n{proc.stdout}"
    )
    mape = float(match.group(1))
    assert 0.0 < mape < 100.0, (
        f"MAPE={mape:.1f}% is outside the physically plausible range (0, 100). "
        f"A value of 0 indicates a trivial/broken fit; a value >= 100 indicates "
        f"a formula error. See docs/ADOPTION.md F-3 for the documented range. "
        f"(ADV-01: this test was added to catch exactly this class of breakage.)"
    )
