#!/usr/bin/env python3
"""Block R: the K-to-min-cut boundary map figure (assets/k_boundary_map.png).

Two small multiples (one per instance; one shared y meaning, no dual axes): mission
exploitability vs interdiction budget K, for SACRED (trained), the two disjoint heuristics,
and the exact equilibrium where it exists. Yardstick: exact at K <= 3, greedy BR past the wall
(fidelity <= 1.8% at K <= 3, gen26 step 2). Data: gen26 ledger + r0_screen + oracle rows.
Palette: dataviz reference slots 1-3 (validated) + neutral for the equilibrium reference.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, GREEN, MAGENTA, NEUTRAL = "#2a78d6", "#008300", "#e87ba4", "#8a8a85"
INK, MUTED = "#1a1a19", "#6e6d66"

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
fig.patch.set_facecolor("white")

# ---- Panel A: 35-159 (m = 4) --------------------------------------------------------------
A = axes[0]
A.plot([1, 2, 3], [0.206, 0.412, 0.604], ls="--", lw=1.6, color=NEUTRAL, zorder=1)
A.plot([1, 2, 3, 4, 5], [0.250, 0.494, 0.738, 0.966, 0.985], lw=2, color=GREEN,
       marker="o", ms=6, zorder=3)
A.plot([1, 2, 3], [0.241, 0.493, 0.737], lw=2, color=MAGENTA, marker="s", ms=6, zorder=3)
A.errorbar([1, 3], [0.256, 0.664], yerr=[[0.010, 0.018], [0.010, 0.018]], lw=2.2, color=BLUE,
           marker="o", ms=7, capsize=3, zorder=4)
A.plot([2], [0.500], marker="o", ms=7, mfc="white", mec=BLUE, mew=1.8, ls="none", zorder=4)
A.axvline(4, color=MUTED, lw=1, ls=":", zorder=0)
A.text(4.02, 0.06, "min-cut m = 4", rotation=90, fontsize=8, color=MUTED, va="bottom")
A.set_title("35–159  (m = 4, exact yardstick; greedy past K = 3)", fontsize=10, color=INK)
# direct labels
A.text(3.06, 0.63, "SACRED", fontsize=9, color=BLUE, fontweight="bold")
A.text(4.4, 0.90, "uniform-\ndisjoint", fontsize=8, color=GREEN, ha="center")
A.text(1.62, 0.455, "heuristics (uniform ≈ inv-vuln)", fontsize=8, color=GREEN, rotation=39)
A.text(2.1, 0.295, "equilibrium (exact, to the wall)", fontsize=8, color=NEUTRAL, rotation=33)

# ---- Panel B: 71-33 (m = 6) --------------------------------------------------------------
B = axes[1]
B.plot([1, 2, 3], [0.128, 0.255, 0.383], ls="--", lw=1.6, color=NEUTRAL, zorder=1)
B.plot([1, 2, 3, 4, 5, 6], [0.167, 0.329, 0.468, 0.602, 0.705, 0.800], lw=2, color=GREEN,
       marker="o", ms=6, zorder=3)
B.plot([1, 2, 3, 4, 5, 6], [0.128, 0.298, 0.456, 0.511, 0.638, 0.766], lw=2, color=MAGENTA,
       marker="s", ms=6, zorder=3)
# full-menu uniform-stack (the STRONGEST naive baseline, admits shared routes) - orange
B.plot([1, 2, 3, 4, 5, 6], [0.167, 0.329, 0.468, 0.590, 0.666, 0.739], lw=1.8, color="#eb6834",
       ls=(0, (4, 2)), marker="^", ms=5, zorder=2)
B.errorbar([5, 6], [0.667, 0.733], yerr=[[0.016, 0.015], [0.016, 0.015]], lw=2.2, color=BLUE,
           marker="o", ms=7, capsize=3, zorder=4)
B.plot([5, 6], [0.667, 0.733], lw=2.2, color=BLUE, zorder=3)
B.axvline(6, color=MUTED, lw=1, ls=":", zorder=0)
B.text(6.02, 0.06, "min-cut m = 6", rotation=90, fontsize=8, color=MUTED, va="bottom")
B.set_title("71–33  (m = 6; exact to K = 3, greedy K ≥ 4: past the LP wall)",
            fontsize=10, color=INK)
B.text(4.62, 0.66, "SACRED", fontsize=9, color=BLUE, fontweight="bold", ha="right")
B.text(5.5, 0.83, "uniform-disjoint", fontsize=8, color=GREEN, ha="center")
B.text(3.0, 0.60, "full-menu stack (best naive)", fontsize=7.5, color="#eb6834", rotation=22)
B.text(3.55, 0.485, "inv-vuln", fontsize=8, color=MAGENTA, rotation=18)
B.text(2.6, 0.30, "equilibrium", fontsize=8, color=NEUTRAL)

for ax in axes:
    ax.set_xlabel("interdiction budget K", fontsize=9, color=INK)
    ax.grid(axis="y", color="#eceae4", lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c7bd")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_ylim(0, 1.02)
axes[0].set_ylabel("mission exploitability (lower = better)", fontsize=9, color=INK)
axes[0].set_xticks([1, 2, 3, 4, 5]); axes[1].set_xticks([1, 2, 3, 4, 5, 6])

fig.suptitle("Where learning pays: trained calibration vs the max-flow heuristics as the "
             "interdiction budget approaches the min-cut", fontsize=11.5, color=INK, y=1.0)
fig.text(0.5, -0.04,
         "Filled SACRED markers: n ≥ 3 seeds (± pop. std); open markers: single seed. "
         "K ≥ 4 columns scored by the certified greedy best response (measured fidelity "
         "≤ 1.8% at K ≤ 3). Sources: gen26 ledger; r0_screen.json.",
         ha="center", fontsize=8, color=MUTED)
fig.tight_layout()
fig.savefig("assets/k_boundary_map.png", dpi=150, bbox_inches="tight", facecolor="white")
print("[written] assets/k_boundary_map.png")
