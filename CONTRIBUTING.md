# Contributing to fitsproof

Thanks for your interest. This repo has one rule that overrides the others:

**Every test must name the fault it detects.** The top-of-file docstring of each test
module lists, per test, the specific fault it would catch. If you cannot name the fault,
delete the test — a test that can only agree with the implementation is worse than none.

## Development setup

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
pytest -q
```

Works offline: no GPU, no CUDA toolkit, no model downloads (the reference model is
generated in-repo from a fixed seed).

## Before you open a PR

Run the same gates CI runs:

```bash
ruff check .
ruff format --check .
python scripts/check_research_traceability.py
pytest -q
```

- Python 3.11+ with annotations on every public function; docstrings on the public API.
- No network in tests. Fixtures and synthetic data only. Deterministic: explicit seeds,
  no wall-clock or dict-ordering dependence in assertions.
- Known-answer tests for numerical routines must derive the expected value *outside*
  this codebase (hand computation shown in a comment, or a published value), and cite
  the source id from `docs/RESEARCH.md` in the module docstring. CI enforces the
  citation for `tests/contract/` and `tests/value/`.
- Fail closed where the contract is involved: never let a mode change, a refusal, or a
  degradation happen without an explicit record. That is the whole point of the repo.

## Scope

- Do not claim speed superiority over llama.cpp, vLLM, or KTransformers. See
  `COMPARISONS.md` — the claim is the contract, not the benchmark.
- One focused change per PR, conventional commits (`feat:`, `fix:`, `refactor:`,
  `docs:`, `test:`, `chore:`).

## Good first issues

Issues labelled `good first issue` are scoped, self-contained, and point at the exact
file and test to extend. Start there.
