#!/usr/bin/env bash
# fitsproof demo — the contract end to end, ~15 seconds.
#
# Requires `fitsproof` on PATH (pip install, or activate the dev venv first:
#   . .venv/bin/activate ).
#
# --- Recording instructions (asciinema -> GIF) -------------------------------
# 1. Record (10-15 s of real terminal):
#      asciinema rec --title "fitsproof — prove it fits, or refuse loudly" \
#        -c "bash docs/demo.sh" demo.cast
# 2. Convert to GIF (install agg once: cargo install agg):
#      agg --speed 1.5 demo.cast docs/demo.gif
#    Or upload for a shareable link: asciinema upload demo.cast
# 3. Reference docs/demo.gif from the README (hero media above the fold).
# ----------------------------------------------------------------------------
set -euo pipefail

command -v fitsproof >/dev/null 2>&1 || {
    echo "fitsproof not on PATH — activate a venv with it installed" >&2
    exit 1
}

echo "=== 1. probe: characterise this machine ==="
fitsproof probe

echo
echo "=== 2. plan: inspect the prediction (exit 0 — no enforcement) ==="
fitsproof plan --budget-gb 4

echo
echo "=== 3. admit: config fits a 4 GB budget ==="
fitsproof admit --budget-gb 4

echo
echo "=== 4. admit: config cannot fit — refused, loudly (exit 2) ==="
fitsproof admit --budget-gb 0.001 || echo "(exit code: $?)"

echo
echo "=== 5. stress: >=20 configs, zero budget violations, zero silent mode changes ==="
fitsproof stress
