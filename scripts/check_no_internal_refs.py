#!/usr/bin/env python3
"""
check_no_internal_refs.py — scan the repo for committed host paths.

Looks for absolute paths that reference the developer's home directory or
any path component that is machine-specific and should not be committed.

Checked locations:
  - reports/          (eval artefacts — historically contained /home/... paths)
  - docs/             (evidence, release notes — must use /build/ redaction)
  - src/              (source code — no host paths in strings or comments)
  - scripts/          (utility scripts — must not embed host dirs)
  - tests/            (test files — no host paths in expected output)

Exits 0 if no violations found, 1 if any violation is found.

Add new patterns to FORBIDDEN_PATTERNS below.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Patterns that must never appear in committed files.
# Each pattern is a regex; any match is a violation.
FORBIDDEN_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"/home/[a-zA-Z0-9_\-]+/"),  # /home/<username>/
    re.compile(r"/Users/[a-zA-Z0-9_\-]+/"),  # macOS /Users/<username>/
]

# File extensions to check
CHECKED_EXTENSIONS = {".py", ".json", ".toml", ".md", ".txt", ".yml", ".yaml", ".sh", ".rst"}

# Directories to check (relative to repo root)
CHECKED_DIRS = [
    "reports",
    "docs",
    "src",
    "scripts",
    "tests",
]

# Files to skip entirely (e.g. lock files, generated artefacts)
SKIP_FILES = {
    "uv.lock",
    "poetry.lock",
    "package-lock.json",
}


def check_repo(root: Path) -> list[tuple[Path, int, str]]:
    """
    Walk checked directories and return a list of (path, line_number, line) violations.
    """
    violations: list[tuple[Path, int, str]] = []

    for dir_name in CHECKED_DIRS:
        target = root / dir_name
        if not target.exists():
            continue
        for path in sorted(target.rglob("*")):
            if not path.is_file():
                continue
            if path.name in SKIP_FILES:
                continue
            if path.suffix not in CHECKED_EXTENSIONS:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for lineno, line in enumerate(text.splitlines(), start=1):
                for pattern in FORBIDDEN_PATTERNS:
                    if pattern.search(line):
                        violations.append((path, lineno, line.rstrip()))
                        break  # one violation per line is enough

    return violations


def main() -> int:
    root = Path(__file__).parent.parent.resolve()
    violations = check_repo(root)

    if violations:
        print(f"FAIL: {len(violations)} internal path reference(s) found:\n")
        for path, lineno, line in violations:
            rel = path.relative_to(root)
            print(f"  {rel}:{lineno}: {line}")
        print("\nReplace host paths with /build/ (see existing reports/ for the pattern).")
        return 1

    print(f"OK: no internal path references found in {', '.join(CHECKED_DIRS)}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
