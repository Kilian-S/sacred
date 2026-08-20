#!/usr/bin/env python3
"""gen39 step 5E zero-shot pass: score the twelve NEW grid defenders (qwen rolls 2-3, llama
rolls 2-3, seeds 0-2) on the SAME saved fresh test sets as roll 1 (zeroshot2_build.json,
laydowns saved). Thin wrapper: rebinds only the checkpoint enumeration and the output path
of the pinned zeroshot2 harness; the scoring loop, cell definition and test objects are the
originals byte for byte. Pre-registered in the step-5e section.

    PYTHONPATH=. OMP_NUM_THREADS=1 ../sacred/.venv/bin/python analysis/gen39_step5e_zeroshot.py
"""
from __future__ import annotations

import json
from pathlib import Path

import analysis.gen39_zeroshot2 as z2

PREFIXES = ("qwenthink16_r2", "qwenthink16_r3", "llm16r2", "llm16r3")
OUT = Path("models/runs/gen39_step5/step5e_zeroshot.json")


def load_ckpts_grid():
    import torch

    from src.agents.sac import ProtagonistSAC
    out = []
    for prefix in PREFIXES:
        for s in (0, 1, 2):
            p = z2.STEP5 / f"{prefix}_seed{s}.json"
            run = json.loads(p.read_text())
            srt = min(run["history"], key=lambda h: h["val"])["sortie"]
            ck = z2.STEP5 / f"{prefix}_seed{s}_ckpts" / f"actor_ep{srt}.pt"
            prot = ProtagonistSAC(node_in_dim=14, edge_in_dim=5, hidden_dim=64, num_layers=2,
                                  heads=4, reward_scale=1.0, device="cpu", role_alpha=True)
            prot.actor.follow_w = torch.nn.Parameter(torch.tensor(1.0))
            prot.actor.route_feat_w = torch.nn.Parameter(torch.zeros(3))
            prot.actor.route_feats = None
            prot.actor.load_state_dict(torch.load(ck, weights_only=True))
            out.append((prefix, s, prot))
    return out


if __name__ == "__main__":
    z2.load_ckpts_all = load_ckpts_grid
    print(f"[step-5e] scoring 12 grid defenders on the saved fresh sets -> {OUT}", flush=True)
    z2.score(workers=6, out_path=str(OUT))
