"""gen33 metric-1 figure: the LLM force populations against the anchored ladder.

Small multiples (2 phases x 3 theatres), one dot per force (n=8 per model per cell, mean over
field seeds 5100-5102, sigma0 = 8 km), population mean tick, and the three anchor references
(random floor +/- sd band, doctrine-heuristic bar, oracle ceiling) on a SHARED y-axis.
Output: assets/gen33_metric1_ladder.png (+ .pdf).
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

SCORES = json.load(open("models/runs/gen33_force_scores.json"))
SCREEN = json.load(open("models/runs/gen33_score_screen.json"))

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE, SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"
C_LLAMA, C_QWEN = "#2a78d6", "#eb6834"
MODELS = (("llama-3.3-70b", "Llama 3.3 70B", C_LLAMA), ("qwen3-27b", "Qwen3 27B", C_QWEN))
THEATRES = (("kgd", "Kaliningrad"), ("ukraine", "Ukraine"), ("narva", "Narva"))
PHASES = (("single", "Phase 1: single (K = 1)"), ("coordinated", "Phase 2: coordinated (K = 3)"))

plt.rcParams.update({"font.family": "sans-serif", "font.size": 8.5,
                     "axes.edgecolor": BASE, "axes.linewidth": 0.8,
                     "xtick.color": MUTED, "ytick.color": MUTED,
                     "text.color": INK, "axes.labelcolor": INK2})

fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.4), sharey=True, facecolor=SURF)
rng = np.random.default_rng(3)
for r, (phase, phase_lab) in enumerate(PHASES):
    for c, (name, th_lab) in enumerate(THEATRES):
        ax = axes[r, c]
        ax.set_facecolor(SURF)
        anc = SCREEN["anchors"][name][phase]
        ax.axhspan(anc["random_mean"] - anc["random_sd"], anc["random_mean"] + anc["random_sd"],
                   color=GRID, alpha=0.75, lw=0, zorder=1)
        ax.axhline(anc["random_mean"], color=MUTED, lw=1.0, ls=(0, (1, 2)), zorder=2)
        ax.axhline(anc["heuristic"], color=INK, lw=1.3, zorder=2)
        ax.axhline(anc["oracle"], color=INK2, lw=1.1, ls=(0, (5, 2)), zorder=2)
        for i, (mkey, mlab, col) in enumerate(MODELS):
            cell = SCORES["cells"][f"{mkey}|{name}|{phase}"]
            x0 = 0.30 + 0.40 * i
            xs = x0 + np.clip(rng.normal(0, 0.045, len(cell["values"])), -0.11, 0.11)
            ax.scatter(xs, cell["values"], s=26, color=col, alpha=0.9, lw=0.8,
                       edgecolor=SURF, zorder=4)
            ax.hlines(cell["mean"], x0 - 0.13, x0 + 0.13, color=col, lw=2.6, zorder=5)
            beat = cell["mean"] > anc["heuristic"]
            ax.annotate("beats bar" if beat else "below bar",
                        (x0, -0.006), ha="center", va="top", fontsize=7,
                        color=INK2 if beat else MUTED,
                        fontweight="bold" if beat else "normal", annotation_clip=False)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.108)
        ax.set_xticks([])
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines["bottom"].set_position(("data", 0))
        if r == 0:
            ax.set_title(th_lab, fontsize=10, color=INK, pad=8)
        if c == 0:
            ax.set_ylabel(f"{phase_lab}\ninduced game value", fontsize=8.5)
            ax.yaxis.set_tick_params(labelsize=7.5)

handles = [
    Line2D([], [], marker="o", ls="", ms=6, mfc=C_LLAMA, mec=SURF, label="Llama 3.3 70B (8 forces)"),
    Line2D([], [], marker="o", ls="", ms=6, mfc=C_QWEN, mec=SURF, label="Qwen3 27B (8 forces)"),
    Line2D([], [], color=INK, lw=1.3, label="doctrine heuristic (the bar)"),
    Patch(facecolor=GRID, alpha=0.75, label="random floor (mean ± sd)"),
    Line2D([], [], color=INK2, lw=1.1, ls=(0, (5, 2)), label="oracle ceiling"),
]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=5,
           frameon=False, fontsize=8, handletextpad=0.5, columnspacing=1.3)
fig.suptitle("gen33 metric 1: LLM red-force hardness against the anchored ladder",
             fontsize=12, color=INK, x=0.5, y=0.985, fontweight="bold")
fig.text(0.5, 0.008,
         "Each dot = one generated force (mean best-response damage over field seeds 5100–5102, "
         "σ₀ = 8 km); thick tick = population mean. Bar = beat the heuristic; per model, no pooling.",
         ha="center", fontsize=7.5, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 0.90))
for ext in ("png", "pdf"):
    fig.savefig(f"assets/gen33_metric1_ladder.{ext}", dpi=220, facecolor=SURF,
                bbox_inches="tight")
print("[written] assets/gen33_metric1_ladder.png/.pdf")
