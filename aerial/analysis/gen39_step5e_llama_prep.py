#!/usr/bin/env python3
"""Authors a fresh llama llm16 curriculum for gen39 step 5e, reusing the step-5 search
unchanged so a roll differs only by the search's own stochasticity. The output carries all four
existing family keys unaltered and puts the fresh roll under "llm16r", so the trainer's test set,
which reads the family keys from the same file, is preserved. Per-field progress persists and an
existing roll is never overwritten.

    PYTHONPATH=. OMP_NUM_THREADS=1 ../sacred/.venv/bin/python \
        analysis/gen39_step5e_llama_prep.py --roll 2
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

import analysis.gen39_step5_prep as p5
from scripts.train_gen39_conceal import TEST_FIELDS, TRAIN_FIELDS, narva_base
from src.envs.aerial_conceal import resample_field
from analysis.gen39_phase1f import map_digest

BANKED = Path("models/runs/gen39_step5/curricula.json")


def main():
    import multiprocessing as mp_
    ap = argparse.ArgumentParser()
    ap.add_argument("--roll", type=int, required=True, choices=(2, 3))
    a = ap.parse_args()
    out = Path(f"models/runs/gen39_step5/curricula_llama{a.roll}.json")
    progress = Path(f"models/runs/gen39_step5/llama{a.roll}_progress.json")
    if out.exists():
        raise SystemExit(f"{out} already exists; refusing to overwrite an existing curriculum")
    base = narva_base()
    pp0 = base.lethality(resample_field(base.coords, 1000), hidden_leth=1.0)
    digest = map_digest(base, pp0)
    banked = json.loads(BANKED.read_text())
    fields = list(TRAIN_FIELDS) + list(TEST_FIELDS)
    new: dict = json.loads(progress.read_text()) if progress.exists() else {}
    if new:
        print(f"  resuming: {len(new)} fields already done", flush=True)
    t0 = time.time()
    with mp_.get_context("spawn").Pool(9, initializer=p5._init) as P:
        for field in fields:
            if str(field) in new:
                continue
            h = sorted(p5.llm16(base, digest, P, field), key=lambda x: -x[1])[:p5.KEEP]
            if not h:
                raise RuntimeError(f"field {field}: zero usable proposals - transport or "
                                   f"model failure, not a curriculum")
            new[str(field)] = [[list(map(int, t)), float(v)] for t, v in h]
            progress.write_text(json.dumps(new, indent=1))
            print(f"  field {field}: llama roll {a.roll} best {new[str(field)][0][1]:.4f} "
                  f"(kept {len(h)})  [{(time.time()-t0)/60:.1f} min]", flush=True)
    outd = dict(banked)
    outd["llm16r"] = new
    out.write_text(json.dumps(outd, indent=1))
    chk = json.loads(out.read_text())
    assert all(chk[k] == banked[k] for k in ("llm16", "local16", "random16", "tuned"))
    tr = np.median([new[str(f)][0][1] for f in TRAIN_FIELDS])
    print(f"\nllama roll {a.roll} train-field median best {tr:.4f}\n[written] {out} "
          f"(all four existing family keys byte-identical; llm16r key = THIS roll)")


if __name__ == "__main__":
    main()
