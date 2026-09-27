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
