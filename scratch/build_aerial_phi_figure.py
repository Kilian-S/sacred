"""Build assets/aerial_phi_boundary.png from models/runs/gen28_screen.json (GAME V2 screen:
curved routes, line-integral exposure, dense grid; 2026-07-17).

Panel A: the naive-lane gap (best-naive / equilibrium) vs coverage phi = 2Kr/W on the open
sector, one series per hazard radius r (the lane-count quantisation is the driver, so phi does
not collapse the family: that IS the finding), banded-effectiveness variants as hollow markers,
the two pinch headline cells annotated. Panel B: the full baseline ladder at the two headline-
candidate cells. Palette: dataviz categorical slots 1-4 (light), validated order; direct labels;
recessive grid; single-mode print asset for the thesis.
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SLOT = {0.8: "#2a78d6", 1.2: "#008300", 1.6: "#e87ba4", 2.0: "#eda100"}
INK, MUT = "#333333", "#8a8a8a"

_data = json.load(open("models/runs/gen28_screen.json"))
cells = _data["cells"] if isinstance(_data, dict) else _data
by = {c["tag"]: c for c in cells}

fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.5, 4.8), width_ratios=[1.15, 1.0])
fig.subplots_adjust(left=0.06, right=0.985, top=0.86, bottom=0.13, wspace=0.30)

# --- Panel A ---------------------------------------------------------------
for r, col in SLOT.items():
    base = sorted([c for c in cells if c["tag"].startswith("base_") and c["r"] == r],
                  key=lambda c: c["phi"])
    xs = [c["phi"] for c in base]; ys = [c["best_naive_over_eq"] for c in base]
    ax.plot(xs, ys, "-o", color=col, lw=2, ms=7, zorder=3)
    ax.annotate(f"r={r}", (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                color=col, fontsize=9, fontweight="bold", va="center")
    band = sorted([c for c in cells if c["tag"].startswith("banded_") and c["r"] == r],
                  key=lambda c: c["phi"])
    if band:
        ax.plot([c["phi"] for c in band], [c["best_naive_over_eq"] for c in band],
                "o", mfc="white", mec=col, mew=1.8, ms=7, zorder=3)
for tag, dy in (("pinch_banded_K1_r1.2", 8), ("pinch_K1_r1.2", -12)):
    c = by[tag]
    ax.plot(c["phi"], c["best_naive_over_eq"], "*", color=INK, ms=13, zorder=4)
    ax.annotate("pinch" + (" + banded" if "banded" in tag else ""),
                (c["phi"], c["best_naive_over_eq"]), xytext=(8, dy),
                textcoords="offset points", fontsize=9, color=INK)
ax.axhline(1.0, color=MUT, lw=1, ls="--", zorder=1)
ax.text(1.52, 1.005, "lane heuristic = optimal", color=MUT, fontsize=8, va="bottom")
ax.set_xlabel("coverage fraction  φ = 2Kr / W"); ax.set_ylabel("best naive stack / equilibrium")
ax.set_title("Open sector: the gap decays with coverage but never closes (1.03-1.59x)\n(filled = uniform p_max, hollow = banded; ★ = pinch cells)",
             fontsize=10, loc="left")

# --- Panel B ---------------------------------------------------------------
arms = [("shortest_det", "shortest path (det)"), ("uniform_lane", "uniform-lane stack"),
        ("invrisk_lane", "inv-risk-lane stack"), ("uniform_full", "uniform-full stack"),
        ("invrisk_full", "inv-risk-full stack"), ("tabular_fp", "tabular smooth FP"),
        ("eq", "equilibrium")]
tags = [("pinch_banded_K1_r1.2", "#2a78d6", "pinch + banded, K=1 r=1.2 (φ=0.30)"),
        ("base_K1_r1.2", "#008300", "open sector, K=1 r=1.2 (φ=0.30)")]
ypos = np.arange(len(arms))[::-1]
for off, (tag, col, label) in zip((0.16, -0.16), tags):
    c = by[tag]
    vals = [c[k] for k, _ in arms]
    bx.plot(vals, ypos + off, "o", color=col, ms=8, zorder=3, label=label)
    for v, y in zip(vals, ypos + off):
        bx.annotate(f"{v:.2f}", (v, y), xytext=(0, 6), textcoords="offset points",
                    fontsize=7.5, color=INK, ha="center")
bx.set_yticks(ypos, [n for _, n in arms])
bx.set_xlabel("worst-case interception probability")
bx.set_xlim(0, 1.0)
bx.set_title("The pre-registered ladder at the headline candidates", fontsize=10, loc="left")
bx.legend(loc="lower right", fontsize=8, frameon=False)

for a in (ax, bx):
    a.spines[["top", "right"]].set_visible(False)
    a.grid(axis="y" if a is ax else "x", color="#e6e6e6", lw=0.6, zorder=0)
    a.tick_params(colors=INK, labelsize=9)
fig.suptitle("gen28 aerial screen: where calibrated mixing beats the lane heuristic (oracle-exact, no training)",
             fontsize=11, x=0.06, ha="left", color=INK)
fig.savefig("assets/aerial_phi_boundary.png", dpi=170)
print("wrote assets/aerial_phi_boundary.png")
