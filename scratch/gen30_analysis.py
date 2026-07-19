#!/usr/bin/env python3
"""gen30 analysis + figures (eval-only; reads the sweep artefacts, prints the ledger-ready
numbers, writes assets/gen30_*.png)."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

RUNS = "models/runs/gen30_secure_flp_{}.json"
TAGS_A = ["kal_primary", "kal_random_s30", "kal_random_s31", "kal_random_s32",
          "gdansk_s30", "gdansk_s31"]
TAGS_B = ["kal_primary", "gdansk_s30", "gdansk_s31"]


def load(tag):
    return json.load(open(RUNS.format(tag)))


def a_stats(d):
    A = d["component_a"]
    cost = np.array([r["cost"] for r in A])
    vj = np.array([r["v_joint"] for r in A])
    vc = np.array([r["v_cap"] for r in A])
    i_c, i_s = int(cost.argmin()), int(vj.argmin())
    cn = (cost - cost.min()) / max(cost.max() - cost.min(), 1e-9)
    vn = (vj - vj.min()) / max(vj.max() - vj.min(), 1e-9)
    par = set(d["pareto_sites"])
    kidx = [i for i, r in enumerate(A) if r["site"] in par]
    knee = min(kidx, key=lambda i: float(np.hypot(cn[i], vn[i])))
    rho, _ = spearmanr(vj, vc)
    return dict(A=A, cost=cost, vj=vj, i_c=i_c, i_s=i_s, knee=knee, rho=float(rho),
                premium=float((vj[i_c] - vj[i_s]) / vj[i_s]),
                knee_cost_extra=float((cost[knee] - cost[i_c]) / cost[i_c]),
                knee_premium=float((vj[knee] - vj[i_s]) / vj[i_s]),
                spread=float(vj.max() / vj.min()),
                gapcap_med=float(np.median([r["gap_vs_cap"] for r in A])))


def print_all():
    print("== Component A across draws ==")
    for tag in TAGS_A:
        d = load(tag)
        s = a_stats(d)
        print(f"  {tag:16s} targets {'-'.join(d['targets']):13s} sites {len(s['A']):3d} | "
              f"premium {100*s['premium']:5.1f}% | knee +{100*s['knee_cost_extra']:.0f}% cost "
              f"-> {100*s['knee_premium']:.0f}% premium | site spread x{s['spread']:.1f} | "
              f"Spearman(vj,vcap) {s['rho']:.3f} | med gap-vs-cap {100*s['gapcap_med']:.0f}%")
    print("== Component B across draws ==")
    for tag in TAGS_B:
        d = load(tag)
        B = d["component_b"]
        g = np.array([r["gain_rel"] for r in B])
        gc = np.array([r["gain_vs_cap_rel"] for r in B])
        rgc = np.array([r["red"]["gap_vs_cap"] for r in B])
        jac = np.array([r["jaccard"] for r in B])
        rj, _ = spearmanr(jac, g)
        opc = np.array([(r["cls_op_len"] - r["cls_cost"]) / r["cls_cost"] for r in B])
        opr = np.array([(r["red_op_len"] - r["cls_cost"]) / r["cls_cost"] for r in B])
        print(f"  {tag:16s} pairs {len(B):3d} | eq gain med {100*np.median(g):4.0f}% "
              f"(>=5% on {int((g >= .05).sum())}/{len(B)}, max {100*g.max():.0f}%) | "
              f"napkin-red beats perfect-cls {int((gc > 0).sum())}/{len(B)} "
              f"(med {100*np.median(gc):.0f}%) | red gap-vs-cap med {100*np.median(rgc):.0f}% | "
              f"gain~jac rho {rj:.2f} | op premium cls {100*np.median(opc):.0f}% / "
              f"red {100*np.median(opr):.0f}%")


def fig_frontier(tag, fname, title):
    d = load(tag)
    s = a_stats(d)
    cost, vj = s["cost"], s["vj"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), width_ratios=[3, 2])
    sc = ax.scatter(cost, vj, c=[r["gap_vs_cap"] for r in s["A"]], cmap="viridis",
                    s=28, alpha=0.85, zorder=2)
    plt.colorbar(sc, ax=ax, label="gap vs m-pairing cap (napkin suboptimality)")
    par = set(d["pareto_sites"])
    pf = sorted([(r["cost"], r["v_joint"]) for r in s["A"] if r["site"] in par])
    ax.plot([p[0] for p in pf], [p[1] for p in pf], "-", c="tab:red", lw=1.4,
            zorder=3, label="Pareto frontier")
    for i, mk, lbl in ((s["i_c"], "s", "p-median (cost-optimal)"),
                       (s["i_s"], "*", "security-optimal"),
                       (s["knee"], "D", "knee")):
        ax.scatter([cost[i]], [vj[i]], marker=mk, s=150 if mk == "*" else 80,
                   facecolor="none", edgecolor="k", lw=1.6, zorder=4, label=lbl)
    if tag == "kal_primary":
        a = [r for r in s["A"] if r["site"] == d["anchor_source"]]
        if a:
            ax.scatter([a[0]["cost"]], [a[0]["v_joint"]], marker="o", s=90,
                       facecolor="none", edgecolor="tab:orange", lw=1.6, zorder=4,
                       label="gen29 anchor (147)")
    ax.set_xlabel("classical service cost (sum of shortest paths)")
    ax.set_ylabel("mission exploitability at the joint equilibrium")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper right")
    ax2.hist(vj, bins=24, color="tab:blue", alpha=0.75)
    for i, c, lbl in ((s["i_c"], "k", "p-median"), (s["i_s"], "tab:red", "security-opt")):
        ax2.axvline(vj[i], color=c, ls="--", lw=1.2, label=lbl)
    ax2.set_xlabel("design security (v_joint)")
    ax2.set_ylabel("sites")
    ax2.set_title(f"prevalence over all {len(vj)} sites (x{s['spread']:.1f} spread)")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fname, dpi=160)
    print(f"[fig] {fname}")


def fig_overlap(tag, fname, title):
    d = load(tag)
    B = d["component_b"]
    g = np.array([r["gain_rel"] for r in B]) * 100
    gc = np.array([r["gain_vs_cap_rel"] for r in B]) * 100
    jac = np.array([r["jaccard"] for r in B])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax.scatter(jac, g, c="tab:blue", s=34, alpha=0.85)
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(np.median(g), color="tab:blue", ls="--", lw=1,
               label=f"median {np.median(g):.0f}%")
    ax.set_xlabel("depot corridor overlap (candidate-edge Jaccard)")
    ax.set_ylabel("equilibrium value of dual-servability (%)")
    ax.set_title("value of the redundancy classical FLP prunes")
    ax.legend(fontsize=8)
    fig.suptitle(title, fontsize=11)
    ax2.scatter(g, gc, c="tab:red", s=34, alpha=0.85)
    lim = [min(gc.min(), 0) - 5, g.max() + 5]
    ax2.plot(lim, lim, "k:", lw=0.8, label="napkin deployment keeps all value")
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xlabel("redundancy value at the coordinated optimum (%)")
    ax2.set_ylabel("value under m<=4 napkin deployment (%)")
    ax2.set_title("deployment-conditionality (above 0 = survives napkin play)")
    ax2.legend(fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(fname, dpi=160)
    print(f"[fig] {fname}")


if __name__ == "__main__":
    print_all()
    fig_frontier("kal_primary", "assets/gen30_frontier.png",
                 "Kaliningrad, primary targets: the (cost, security) design frontier")
    fig_overlap("kal_primary", "assets/gen30_overlap_value.png",
                "Kaliningrad, primary targets")
    fig_frontier("gdansk_s30", "assets/gen30_frontier_gdansk.png",
                 "Gdansk (held-out city, unscreened targets): frontier replication")
    fig_overlap("gdansk_s30", "assets/gen30_overlap_value_gdansk.png",
                "Gdansk (held-out city)")
