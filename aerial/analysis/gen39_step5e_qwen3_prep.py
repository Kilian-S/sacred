#!/usr/bin/env python3
"""Authors the third qwenthink16 curriculum for gen39 step 5e.

    PYTHONPATH=. OMP_NUM_THREADS=1 ../sacred/.venv/bin/python analysis/gen39_step5e_qwen3_prep.py
"""
from pathlib import Path

import analysis.gen39_step5c_prep as p5c

OUT = Path("models/runs/gen39_step5/curricula_qwenthink3.json")
PROGRESS = Path("models/runs/gen39_step5/qwenthink3_progress.json")

if __name__ == "__main__":
    if OUT.exists():
        raise SystemExit(f"{OUT} already exists; refusing to overwrite an existing curriculum")
    p5c.OUT = OUT
    p5c.PROGRESS = PROGRESS
    print(f"[step-5e] third qwenthink16 curriculum -> {OUT}", flush=True)
    p5c.main()
