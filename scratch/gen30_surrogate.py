#!/usr/bin/env python3
"""gen30 Component C (eval-only): the Obj-4 metamodel rider. Fits the repo's SurrogateMLP on
CHEAP pre-solve design features -> exact joint-equilibrium security value (v_joint), over the
pooled gen30 design rows (single-depot sites + two-depot redundant designs, all seeded draws).

Rows: Kaliningrad pool (kal_primary + kal_random_s30/31/32) split 70/30 (seeded) for the
held-out accuracy row; the Gdansk rows (gdansk_s30/31) are scored ZERO-SHOT by the
Kaliningrad-trained model (disclosed exploratory transfer row; features are z-scored on the
training set only). Metrics: held-out RMSE, Spearman rho, argmin regret (true suboptimality of
the surrogate's chosen design within the held-out pool)."""
from __future__ import annotations

import json

import numpy as np
import torch
from scipy.stats import spearmanr

from src.sbo.surrogate import train_surrogate

FEATS = ["cost_total", "cost_max", "cost_min", "mincut_min", "mincut_sum", "routes_sum",
         "E_union", "vuln_mean", "vuln_max", "dist_pair", "jaccard_pair", "is_pair"]


def rows_of(tag):
    d = json.load(open(f"models/runs/gen30_secure_flp_{tag}.json"))
    out = []
    for r in d["component_a"]:
        f = dict(r["feat"])
        f.update(dist_pair=0.0, jaccard_pair=1.0, is_pair=0.0)
        out.append((f, r["v_joint"]))
    for r in d["component_b"]:
        f = dict(r["feat"])
        f["is_pair"] = 1.0
        out.append((f, r["red"]["v_joint"]))
    return out


def mat(rows):
    X = np.array([[fr[k] for k in FEATS] for fr, _ in rows], dtype=np.float32)
    y = np.array([v for _, v in rows], dtype=np.float32)
    return X, y


def score(model, X, y, mu, sd, label):
    with torch.no_grad():
        p = model(torch.tensor((X - mu) / sd)).squeeze(-1).numpy()
    rmse = float(np.sqrt(np.mean((p - y) ** 2)))
    rho, pv = spearmanr(p, y)
    i_pred = int(p.argmin())
    regret = float((y[i_pred] - y.min()) / max(y.min(), 1e-9))
    rank = int((y < y[i_pred]).sum()) + 1
    print(f"  {label:22s} n={len(y):3d} | RMSE {rmse:.4f} | Spearman {rho:.3f} (p={pv:.1e}) | "
          f"argmin regret {100*regret:.1f}% (chosen design true rank {rank}/{len(y)})")
    return dict(n=len(y), rmse=rmse, spearman=float(rho), regret=regret, rank=rank)


def main():
    torch.manual_seed(30)
    np.random.seed(30)
    kal = sum((rows_of(t) for t in
               ("kal_primary", "kal_random_s30", "kal_random_s31", "kal_random_s32")), [])
    gda = sum((rows_of(t) for t in ("gdansk_s30", "gdansk_s31")), [])
    X, y = mat(kal)
    idx = np.random.permutation(len(y))
    n_tr = int(0.7 * len(y))
    tr, te = idx[:n_tr], idx[n_tr:]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    model, _ = train_surrogate((X[tr] - mu) / sd, y[tr], epochs=200, lr=5e-3,
                               batch_size=16, hidden_dim=32)
    model.eval()
    print(f"[gen30-C] pooled Kaliningrad designs {len(y)} (train {n_tr}); "
          f"features = {len(FEATS)} cheap pre-solve columns, no LP at query time")
    res = {"train": score(model, X[tr], y[tr], mu, sd, "train (in-sample)"),
           "heldout_kal": score(model, X[te], y[te], mu, sd, "HELD-OUT Kaliningrad"),
           "zeroshot_gdansk": score(model, *mat(gda), mu, sd, "zero-shot Gdansk")}
    json.dump(res, open("models/runs/gen30_surrogate.json", "w"), indent=1)
    print("[written] models/runs/gen30_surrogate.json")


if __name__ == "__main__":
    main()
