"""
fitsproof.engine.speculative — Speculative decoding (draft-then-verify).

A draft model proposes multiple tokens; the target model verifies them in
one forward pass. Accepted tokens are identical to greedy target output.

The critical property tested:
    speculative_generate(prompt) == greedy_generate(prompt)
under greedy decoding (temperature=0). This is the acceptance criterion.

Source:
  - Leviathan et al. 2023 (Fast Inference from Transformers via Speculative Decoding),
    https://arxiv.org/abs/2211.17192
"""

from __future__ import annotations

import numpy as np

from fitsproof.engine.sampling import Sampler
from fitsproof.engine.transformer import Transformer


def speculative_generate(
    target: Transformer,
    draft: Transformer,
    prompt_ids: list[int],
    max_new_tokens: int = 20,
    speculation_len: int = 4,
    seed: int = 0,
) -> list[int]:
    """
    Speculative decoding: draft proposes *speculation_len* tokens, target verifies.

    Under greedy decoding (temperature=0), accepted tokens are identical to
    non-speculative greedy generation from *target* alone.

    Algorithm (Leviathan et al. 2023):
      1. Draft proposes gamma tokens greedily.
      2. Target runs one forward pass over (context + draft_tokens).
      3. For each draft token at position i, check if target's greedy choice
         at position (context_len + i - 1) equals draft_tokens[i].
         - If yes: accept it.
         - If no: reject it, emit target's token instead, start over.
      4. If all gamma tokens accepted: emit one bonus target token at the end.

    This is tested in tests/engine/test_speculative.py to produce the same
    output as target.generate(prompt_ids, max_new_tokens, temperature=0.0).

    Fault detected: if the verification step accepts tokens that don't match
    target greedy, the equality property is broken and the test fails.
    """
    sampler = Sampler(seed=seed)
    generated: list[int] = []
    context = list(prompt_ids)

    while len(generated) < max_new_tokens:
        remaining = max_new_tokens - len(generated)
        gamma = min(speculation_len, remaining)

        # 1. Draft proposes gamma tokens greedily
        draft_tokens: list[int] = []
        draft_context = list(context)
        for _ in range(gamma):
            dt = draft.generate(
                draft_context,
                max_new_tokens=1,
                temperature=0.0,
                sampler=Sampler(seed=seed),
            )
            draft_tokens.append(dt[0])
            draft_context.append(dt[0])

        # 2. Target verifies: run forward over context + all draft tokens at once
        #    The verify sequence is context ++ draft_tokens.
        #    Output at position i (0-indexed) predicts token at position i+1.
        verify_ids = context + draft_tokens
        verify_arr = np.array(verify_ids, dtype=np.int64)[np.newaxis, :]
        target_logits = target.forward_reference(verify_arr)  # (1, len(verify_ids), vocab)

        # 3. Accept/reject draft tokens
        # target_logits[0, j, :] predicts the token *after* position j.
        # Draft token i (= draft_tokens[i]) is at position len(context)+i in verify_ids.
        # The target's prediction for that token comes from position len(context)+i-1.
        n_context = len(context)
        n_accepted = 0

        for i, draft_tok in enumerate(draft_tokens):
            # The prediction position: verify_ids has context at [0..n_context-1],
            # then draft_tokens at [n_context..n_context+gamma-1].
            # To predict draft_tokens[i] we look at logits at n_context + i - 1.
            pred_pos = n_context + i - 1
            greedy_target = int(np.argmax(target_logits[0, pred_pos]))
            if greedy_target == draft_tok:
                n_accepted += 1
                context.append(draft_tok)
                generated.append(draft_tok)
                if len(generated) >= max_new_tokens:
                    break
            else:
                # Reject: emit target's correct token instead
                context.append(greedy_target)
                generated.append(greedy_target)
                break  # restart speculative loop with updated context

        # 4. If all gamma tokens accepted, emit one bonus target token
        if n_accepted == gamma and len(generated) < max_new_tokens:
            # The bonus token comes from logits at position n_context + gamma - 1
            bonus_pos = n_context + gamma - 1
            bonus_tok = int(np.argmax(target_logits[0, bonus_pos]))
            context.append(bonus_tok)
            generated.append(bonus_tok)

    return generated[:max_new_tokens]
