#!/usr/bin/env python3
"""gen39 step 5E: the THIRD qwenthink16 curriculum (authoring roll 3 of the grid), the
step-5d wrapper pattern verbatim with roll-3 output paths. Pre-registered in the ledger's
step-5e section before any call.

    PYTHONPATH=. OMP_NUM_THREADS=1 ../sacred/.venv/bin/python scratch/gen39_step5e_qwen3_prep.py
"""
from pathlib import Path

import scratch.gen39_step5c_prep as p5c

OUT = Path("models/runs/gen39_step5/curricula_qwenthink3.json")
PROGRESS = Path("models/runs/gen39_step5/qwenthink3_progress.json")

if __name__ == "__main__":
    if OUT.exists():
        raise SystemExit(f"{OUT} already exists; refusing to overwrite a banked curriculum")
    p5c.OUT = OUT
    p5c.PROGRESS = PROGRESS
    print(f"[step-5e] third qwenthink16 curriculum -> {OUT}", flush=True)
    p5c.main()
