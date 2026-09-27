#!/usr/bin/env python3
"""
scripts/check_research_traceability.py — M4: enforce that core tests cite research sources.

Two checks are performed:

1. PAPER-TRACEABILITY.md check: every row marked "IMPLEMENTED" in
   docs/PAPER-TRACEABILITY.md must have a non-empty "Our implementation" column
   and a non-empty "Experiment" column. A row with placeholder text or empty cells
   fails this check.

2. Test-citation check: every test file under tests/contract/ and tests/value/
   must contain a reference to a source ID from docs/RESEARCH.md in its module
   docstring. This ensures tests are anchored to a cited paper.

Wire into CI:
  - python scripts/check_research_traceability.py

A source reference is a string matching:
  - "Source: [N]" or "Sources: [N], [M]" (brackets are optional but conventional)
  - "Source N" where N is an integer that appears as "### N." in RESEARCH.md
  - Any occurrence of "[1]", "[2]", ... "[N_max]" in the docstring

Usage:
  python scripts/check_research_traceability.py [--strict]

  --strict: also require tests in tests/engine/ to cite sources.

Exit codes:
  0 — all checks pass
  1 — one or more checks fail
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path


def _extract_source_ids(research_path: Path) -> set[int]:
    """
    Parse docs/RESEARCH.md and return the set of numeric source IDs defined.

    Looks for lines matching "### N." (e.g. "### 1.", "### 12.") which is the
    format used in the RESEARCH.md source list.
    """
    ids: set[int] = set()
    header_re = re.compile(r"^###\s+(\d+)\.")
    for line in research_path.read_text().splitlines():
        m = header_re.match(line)
        if m:
            ids.add(int(m.group(1)))
    return ids


def _get_module_docstring(source: str) -> str:
    """Extract the top-level module docstring from Python source code."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ""
    if tree.body and isinstance(tree.body[0], ast.Expr):
        node = tree.body[0].value
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
    return ""


def _has_source_citation(docstring: str, valid_ids: set[int]) -> bool:
    """
    Return True if the docstring contains at least one valid source ID citation.

    Accepts patterns:
      [N]          — bracket-style reference, e.g. "[1]", "[12]"
      Source: N    — inline reference, e.g. "Source: 1", "Sources: 1, 2"
      source N     — case-insensitive, e.g. "source 3"
    """
    # Bracket-style: [N] where N is a valid ID
    bracket_re = re.compile(r"\[(\d+)\]")
    for m in bracket_re.finditer(docstring):
        if int(m.group(1)) in valid_ids:
            return True

    # Source: N or Sources: N, M
    source_re = re.compile(r"[Ss]ource[s]?\s*:?\s*([\d,\s]+)")
    for m in source_re.finditer(docstring):
        for num_str in re.findall(r"\d+", m.group(1)):
            if int(num_str) in valid_ids:
                return True

    return False


def check_traceability(
    repo_root: Path,
    strict: bool = False,
) -> list[tuple[Path, str]]:
    """
    Check all core test files for source citations.

    Returns a list of (path, reason) pairs for files that fail the check.
    Empty list means all files pass.
    """
    research_path = repo_root / "docs" / "RESEARCH.md"
    if not research_path.exists():
        return [(research_path, "docs/RESEARCH.md not found — cannot verify citations")]

    valid_ids = _extract_source_ids(research_path)
    if not valid_ids:
        return [
            (research_path, "No source IDs found in docs/RESEARCH.md (expected '### N.' headers)")
        ]

    # Core test directories that MUST have citations
    core_dirs = [
        repo_root / "tests" / "contract",
        repo_root / "tests" / "value",
    ]
    if strict:
        core_dirs.append(repo_root / "tests" / "engine")

    failures: list[tuple[Path, str]] = []
    for test_dir in core_dirs:
        if not test_dir.exists():
            continue
        for test_file in sorted(test_dir.glob("test_*.py")):
            source = test_file.read_text()
            docstring = _get_module_docstring(source)
            if not docstring:
                failures.append(
                    (
                        test_file,
                        "Missing module docstring — add one citing the research source IDs "
                        "(e.g. 'Source: [1] Roofline...')",
                    )
                )
                continue
            if not _has_source_citation(docstring, valid_ids):
                failures.append(
                    (
                        test_file,
                        f"Module docstring contains no valid source ID citation. "
                        f"Valid IDs: {sorted(valid_ids)}. "
                        f"Add e.g. 'Source: [1]' or 'Sources: [2], [7]' to the docstring.",
                    )
                )

    return failures


def check_paper_traceability_table(repo_root: Path) -> list[tuple[Path, str]]:
    """
    Validate docs/PAPER-TRACEABILITY.md: every row marked IMPLEMENTED must have
    non-empty 'Our implementation' and 'Experiment' columns.

    Table format (Markdown pipe-delimited):
      | # | Paper | Mechanism | Our implementation (file:symbol) | Experiment (file::test) | ... | Status |

    Returns a list of (path, reason) pairs for failures.
    """
    traceability_path = repo_root / "docs" / "PAPER-TRACEABILITY.md"
    if not traceability_path.exists():
        return [(traceability_path, "docs/PAPER-TRACEABILITY.md not found")]

    content = traceability_path.read_text()
    failures: list[tuple[Path, str]] = []

    # Find table rows: lines with | that contain IMPLEMENTED or PARTIAL
    for lineno, line in enumerate(content.splitlines(), 1):
        if "| IMPLEMENTED" not in line and "| PARTIAL" not in line:
            continue
        # Parse pipe-delimited columns
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        # Expected columns: #, Paper, Mechanism, Implementation, Experiment, Claim, Status
        if len(parts) < 7:
            failures.append(
                (
                    traceability_path,
                    f"Line {lineno}: row has {len(parts)} columns (expected ≥7): {line[:80]}",
                )
            )
            continue
        source_num = parts[0]
        impl_col = parts[3]  # "Our implementation (file:symbol)"
        experiment_col = parts[4]  # "Experiment (file::test)"
        status_col = parts[6]

        if status_col not in {"IMPLEMENTED", "PARTIAL"}:
            continue  # skip rows with other statuses

        empty_markers = {"", "—", "-", "N/A", "TODO", "TBD"}

        if impl_col in empty_markers:
            failures.append(
                (
                    traceability_path,
                    f"Source {source_num}: 'Our implementation' column is empty for a {status_col} row. "
                    f"Add a file:symbol reference or change status.",
                )
            )
        if experiment_col in empty_markers:
            failures.append(
                (
                    traceability_path,
                    f"Source {source_num}: 'Experiment' column is empty for a {status_col} row. "
                    f"Add a file::test reference or change status.",
                )
            )

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check that core test files cite research sources from RESEARCH.md."
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Also require tests/engine/ files to have citations.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Repo root directory (default: parent of this script's directory).",
    )
    args = parser.parse_args()

    repo_root = args.root or Path(__file__).parent.parent

    # Check 1: PAPER-TRACEABILITY.md table completeness
    table_failures = check_paper_traceability_table(repo_root)
    if table_failures:
        print("TRACEABILITY CHECK FAILED — docs/PAPER-TRACEABILITY.md has incomplete rows:\n")
        for path, reason in table_failures:
            rel = path.relative_to(repo_root) if path.is_relative_to(repo_root) else path
            print(f"  {rel}")
            print(f"    {reason}\n")
        print(
            "Every IMPLEMENTED row in PAPER-TRACEABILITY.md must name a file:symbol "
            "implementation and a file::test experiment. "
            "Papers with no implementation or test must be removed from the bibliography."
        )
        return 1

    # Check 2: test-file citation check
    failures = check_traceability(repo_root, strict=args.strict)

    if failures:
        print(
            "TRACEABILITY CHECK FAILED — the following test files have no research source citation:\n"
        )
        for path, reason in failures:
            rel = path.relative_to(repo_root) if path.is_relative_to(repo_root) else path
            print(f"  {rel}")
            print(f"    {reason}\n")
        print(
            "Every core test file must cite the specific source ID(s) from docs/RESEARCH.md "
            "that its assertions derive from. This is required by the QUALITY-CONTRACT (§4, M4)."
        )
        return 1

    research_path = repo_root / "docs" / "RESEARCH.md"
    valid_ids = _extract_source_ids(research_path)
    mode = "(strict)" if args.strict else "(core only)"
    print(
        f"TRACEABILITY OK {mode}: all core test files cite valid research sources. "
        f"Checked {len(valid_ids)} source IDs from RESEARCH.md. "
        f"PAPER-TRACEABILITY.md table validated ({sum(1 for line in (repo_root / 'docs' / 'PAPER-TRACEABILITY.md').read_text().splitlines() if '| IMPLEMENTED' in line)} IMPLEMENTED rows)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
