"""
README snippet execution tests — every ```python block in README.md must run.

Sources: [1] Roofline Model, [2] Memory-Bandwidth-Bound Decode (the contract
surfaces documented in the README are the CLI/API face of the roofline
admission model).

Faults detected:
  test_readme_declares_all_four_plugin_surfaces:
      Fault: one of the four M2 surfaces (client / guard / server / mcp)
      missing from the README, so users never learn it exists.
  test_readme_python_snippets_execute:
      Fault: a broken or stale README code snippet shipped as marketing
      (M2: "a snippet that has never run is marketing, not documentation").
      Any snippet that raises fails the test; each snippet also carries its
      own assert of observable behaviour (admission record, refusal-before-
      allocation, MCP tool list).
"""

from __future__ import annotations

import contextlib
import io
import re
from pathlib import Path

README = Path(__file__).resolve().parents[2] / "README.md"

REQUIRED_SURFACES = ("client", "guard", "server", "mcp")


def _python_snippets() -> list[str]:
    text = README.read_text()
    return re.findall(r"```python\n(.*?)```", text, flags=re.DOTALL)


def test_readme_declares_all_four_plugin_surfaces() -> None:
    text = README.read_text()
    for surface in REQUIRED_SURFACES:
        marker = f"# SURFACE: {surface}"
        assert marker in text, f"README is missing the {marker} snippet (M2 surface)"


def test_readme_python_snippets_execute() -> None:
    snippets = _python_snippets()
    marked = [s for s in snippets if "# SURFACE:" in s]
    assert len(marked) >= 4, f"expected >=4 surface snippets, found {len(marked)}"
    for i, snippet in enumerate(snippets):
        ns: dict = {}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(compile(snippet, f"README.md#block{i}", "exec"), ns)  # noqa: S102
        # The snippet's own asserts are the behaviour check; reaching here
        # without an exception means the documented code actually runs.
