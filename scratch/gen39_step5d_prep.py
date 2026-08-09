#!/usr/bin/env python3
"""gen39 step 5D: author a SECOND qwenthink16 curriculum, identical in every pinned respect,
differing only in the search's own stochasticity (the proposal loop samples at temperature 0.9,
so a fresh run is a fresh draw). Pre-registered in `experiments/gen39_concealment.md`
(step-5d section, written before any call).

This is a THIN WRAPPER, not a fork: it rebinds only the two output paths of
`scratch/gen39_step5c_prep.py` and calls its `main()`, so the search, prompt, budget, doctrine,
operating point and verification are the pinned originals, byte for byte. The instrument is not
edited, per the standing rule.

Resumable: progress persists per field, so an interrupted run continues where it stopped.

    PYTHONPATH=. OMP_NUM_THREADS=1 ../sacred/.venv/bin/python scratch/gen39_step5d_prep.py
"""
from pathlib import Path

import scratch.gen39_step5c_prep as p5c

OUT = Path("models/runs/gen39_step5/curricula_qwenthink2.json")
PROGRESS = Path("models/runs/gen39_step5/qwenthink2_progress.json")

if __name__ == "__main__":
    if OUT.exists():
        raise SystemExit(f"{OUT} already exists; refusing to overwrite a banked curriculum")
    p5c.OUT = OUT
    p5c.PROGRESS = PROGRESS
    print(f"[step-5d] second qwenthink16 curriculum -> {OUT}\n"
          f"[step-5d] progress -> {PROGRESS} (resumable per field)", flush=True)
    p5c.main()
