"""
Tests for fitsproof.engine.speculative.

Faults detected by each test:
  test_speculative_equals_greedy:
    ACCEPTANCE CRITERION 4: speculative decoding under greedy (temperature=0)
    must produce identical output to non-speculative greedy generation.

    Fault detected: if the verification step accepts a wrong draft token,
    the speculative output will diverge from the target greedy output.
    This test is the hardest possible check on the speculative decoder.

  test_speculative_respects_max_tokens:
    speculative_generate must return exactly max_new_tokens tokens.
    Fault: loop termination error causes over- or under-generation.

  test_speculative_is_deterministic:
    Same seed must produce same output.
    Fault: undeclared global state in the speculative loop.
"""

from __future__ import annotations

import numpy as np
import pytest

from fitsproof.engine.model import generate_reference_model
from fitsproof.engine.transformer import Transformer


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    path = tmp_path_factory.mktemp("spec_model")
    generate_reference_model(path, seed=42)
    from fitsproof.engine.model import load_bundle

    return load_bundle(path)


@pytest.fixture(scope="module")
def target(bundle):
    cfg, weights = bundle
    return Transformer(cfg, weights)


@pytest.fixture(scope="module")
def draft(tmp_path_factory):
    """
    ADV-05 fix: the draft model uses a DIFFERENT random seed from the target.

    Using draft=target makes the test vacuous — all draft proposals are trivially
    accepted (they're the greedy target proposals), so the verification step never
    exercises the rejection path and 'accept a wrong draft token' cannot be observed.

    With a different seed, the draft produces different logit distributions, so:
    - Some draft proposals will DIFFER from the target's greedy choice
    - The verifier must REJECT those and emit the target's token instead
    - The speculative output must still equal the target greedy output

    This makes the test non-trivial: it exercises both the accept path and the
    reject-then-correct path.
    """
    path = tmp_path_factory.mktemp("draft_model")
    generate_reference_model(path, seed=999)  # different seed from target (42)
    from fitsproof.engine.model import load_bundle

    cfg, weights = load_bundle(path)
    return Transformer(cfg, weights)


def test_speculative_equals_greedy(target, draft) -> None:
    """
    ACCEPTANCE CRITERION 4: speculative output must be identical to greedy target output.

    Method: run target.generate(greedy) and speculative_generate(target, draft, greedy),
    compare outputs token by token.

    When draft=target and temperature=0, every draft token should be accepted
    (it's the same as the greedy target), so speculative degrades to the
    pure target greedy pass — output must match exactly.
    """
    from fitsproof.engine.sampling import Sampler
    from fitsproof.engine.speculative import speculative_generate

    cfg = target.cfg
    rng = np.random.default_rng(7)
    prompt = rng.integers(0, cfg.vocab_size, size=4, dtype=np.int64).tolist()

    # Reference greedy output
    greedy_out = target.generate(prompt, max_new_tokens=8, temperature=0.0, sampler=Sampler(0))

    # Speculative output (draft = target, so all proposals are correct)
    spec_out = speculative_generate(
        target=target,
        draft=draft,
        prompt_ids=prompt,
        max_new_tokens=8,
        speculation_len=4,
        seed=0,
    )

    assert spec_out == greedy_out, f"Speculative output {spec_out} differs from greedy {greedy_out}"


def test_speculative_respects_max_tokens(target, draft) -> None:
    """
    Fault: off-by-one in the speculative loop returns wrong count.
    """
    from fitsproof.engine.speculative import speculative_generate

    cfg = target.cfg
    rng = np.random.default_rng(13)
    prompt = rng.integers(0, cfg.vocab_size, size=3, dtype=np.int64).tolist()

    for n in [1, 4, 8]:
        out = speculative_generate(target, draft, prompt, max_new_tokens=n, speculation_len=2)
        assert len(out) == n, f"Expected {n} tokens, got {len(out)}"


def test_speculative_is_deterministic(target, draft) -> None:
    """
    Fault: if the speculative sampler uses global state, outputs differ across calls.
    """
    from fitsproof.engine.speculative import speculative_generate

    cfg = target.cfg
    rng = np.random.default_rng(99)
    prompt = rng.integers(0, cfg.vocab_size, size=4, dtype=np.int64).tolist()

    out1 = speculative_generate(target, draft, prompt, max_new_tokens=6, seed=0)
    out2 = speculative_generate(target, draft, prompt, max_new_tokens=6, seed=0)
    assert out1 == out2, "Speculative generate is not deterministic"
