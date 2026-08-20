"""Zero-shot K/N rows: evaluation only, with no retraining.

A frozen multi-city generalist trained at N=3, K=1 is evaluated on held-out Gdansk ODs at a
shifted adversary budget K=2 and fleet size N=5, scored against each (OD, K, N) cell's own oracle
equilibrium. The policy conditions on the map (edge vulnerability and per-route features) rather
than on K or N, so this tests whether the learned hedge survives a budget or fleet shift.
"""
from __future__ import annotations

import argparse
import glob
import json
import re

import numpy as np
import torch

from scripts.train_generalist import CITY_PATHS, Instance, exact_ratio, sample_instances
from src.agents.sac import ProtagonistSAC
from src.baselines.multiconvoy_oracle import best_response_attacker_multi


def _mkprot(state=None):
    prot = ProtagonistSAC(node_in_dim=14, edge_in_dim=5, hidden_dim=64, num_layers=2, heads=4,
                          device="cpu", role_alpha=True)
    prot.actor.follow_w = torch.nn.Parameter(torch.tensor(1.0))
    prot.actor.route_feat_w = torch.nn.Parameter(torch.zeros(2)); prot.actor.route_feats = None
    if state is not None:
        prot.actor.load_state_dict(state)
    return prot


def tap_ratio(states, it):
    """Score the averaged exact occupancy distributions of a set of checkpoints.

    Averaging is needed because a single checkpoint is noisy under fictitious play.
    """
    ds = []
    for st in states:
        _, d = exact_ratio(_mkprot(st), it)
        ds.append(d)
    tap = np.mean(ds, axis=0)
    _, expl = best_response_attacker_multi(it.env.obj_matrix, tap)
    return float(expl) / it.eq, float(expl)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("actor")
    ap.add_argument("--json-out", default="models/runs/zst_kn_rows.json"); a = ap.parse_args()
    torch.set_num_threads(4)
    # Average over the three checkpoints centred on the selected best checkpoint, which is the
    # deployable object, rather than over the last three, which are drifted final iterates.
    ck_dir = a.actor.rsplit("/", 1)[0]
    cks = {int(re.search(r"ep(\d+)", c).group(1)): c
           for c in glob.glob(f"{ck_dir}/actor_ep*.pt")}
    best_at = int(re.search(r"ep(\d+)", a.actor).group(1))
    eps = sorted(cks)
    ci = eps.index(best_at) if best_at in eps else len(eps) - 1
    window = eps[max(0, ci - 1):ci + 2]  # best-1, best, best+1
    sel = [torch.load(cks[e], map_location="cpu") for e in window]
    rnd_state = _mkprot().actor.state_dict()

    out = {}
    for (N, K) in [(3, 1), (3, 2), (5, 1)]:
        insts = sample_instances(6, N, K, (0.15, 0.95), 8, 0, city="gdansk")
        gr = [tap_ratio(sel, it)[0] for it in insts]
        rr = [exact_ratio(_mkprot(rnd_state), it)[0] for it in insts]
        beats = [tap_ratio(sel, it)[0] * it.eq < it.loss_det for it in insts]
        out[f"N{N}K{K}"] = {"gen_mean": float(np.mean(gr)), "rand_mean": float(np.mean(rr)),
                            "beats_loss_det": int(sum(beats)), "n": len(insts),
                            "gen": [round(x, 2) for x in gr]}
        tag = "(train regime, sanity)" if (N, K) == (3, 1) else "(zero-shot budget/fleet shift)"
        print(f"N={N} K={K} {tag}: gen(TAP) {np.mean(gr):.2f}x vs rand {np.mean(rr):.2f}x | "
              f"beats loss_det {sum(beats)}/{len(insts)} | per-OD {[round(x,2) for x in gr]}")
    json.dump(out, open(a.json_out, "w"), indent=2)
    print(f"[written] {a.json_out}")


if __name__ == "__main__":
    main()
