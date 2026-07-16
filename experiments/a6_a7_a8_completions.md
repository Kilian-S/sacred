# Block A completions: A6 retrieval baseline, A7 gap-closure ladder, A8 prevalence figure

- **status: PRE-REGISTERED 2026-07-12 (NEXT_STEPS_MASTER Block A; all EVAL/ORACLE-ONLY, no
  training); results appended per item below.**
- **git SHA:** the commit landing this ledger + the three scripts.

## A6: the retrieval baseline for the ZST act (`scratch/a6_retrieval.py`)

**Why:** a trivial amortiser might match the generalist: find the most similar TRAINING instance
and play ITS equilibrium mixture on the new menu. If naive retrieval ties gen16, the network adds
little beyond feature matching; if gen16 wins clearly, the act strengthens. Complements gen24
(retrieval = memory without generalisation; distillation = labels without interaction; SACRED =
interaction).

**Design (fixed before looking):** train pool = the 18 gen16 training instances; test = the 6
held-out Gdansk ODs. Scale-free instance features (route count, candidate-edge count,
cost_min/mean, cost_std/mean, vuln mean/max, mean route-overlap Jaccard, harmonic vulnerability
aggregate, pmax min/mean), z-normalised over the train pool; nearest neighbour by Euclidean
distance. The neighbour's equilibrium STACKED mixture is mapped onto the test menu by COST RANK
(k-th cheapest gets k-th cheapest's mass; unmatched mass renormalised); secondary variant maps by
WORST-VULNERABILITY rank. Score = stacked occupancy distribution under the test OD's oracle BR,
ratio to its equilibrium.

> **Pre-committed reading:** retrieval mean > 1.88 (= clearly worse than gen16's 1.733) =>
> the generalist adds real value beyond feature matching. Retrieval in [1.58, 1.88] => partial:
> report as "a surprisingly strong trivial baseline". Retrieval < 1.58 => retrieval matches or
> beats the trained generalist; the ZST act's network contribution is re-scoped honestly.

### A6 RESULT (2026-07-12): retrieval MATCHES the trained generalist; the transfer act's ladder is now fully bounded

| arm (held-out Gdansk, 6 ODs) | mean ratio |
|---|---|
| vanilla (cost-trained) generalist (gen21) | 2.338 |
| random-init net | ~1.99 |
| uniform-stack heuristic (computed this session) | 1.989 |
| adversarial generalist (gen16) | 1.733 select-on-train / 1.677 select-on-test / 1.761 val-stopped |
| **retrieval, cost-rank (this item)** | **1.676** [1.36, 1.51, 1.40, 2.12, 1.55, 2.10] |
| retrieval, vulnerability-rank (secondary) | 1.783 |
| distillation + val early stop (gen24) | 1.555 |
| equilibrium | 1.000 |

**Pre-committed branch fired: the middle one, at its extreme.** Nearest-training-instance
equilibrium retrieval (mapped by cost rank; no test-side solve; labels on the 18 train instances
only) lands at **1.676 = the trained generalist's own level**. Combined with gen24 (distillation
1.555) and zst_map_robustness (map insensitivity), the honest synthesis for the ZST act is now
unambiguous:

1. **The transferable content at this instance family is COARSE** (a spread hedge shaped by
   cost/vulnerability structure): several cheap amortisers reach ~1.55-1.78, and everything
   threat-aware beats random/uniform (~1.99) and cost-training (2.34).
2. **The adversarial generalist's distinction is NOT superior zero-shot transfer.** Its unique,
   measured properties are: label-freeness (retrieval and distillation both consume train-side
   equilibrium labels; self-play needs none), self-stopping (gen24: no overfitting without a
   validation signal), and threat-field robustness (zst_map_robustness). Those, plus the
   past-the-enumeration-wall regime where labels cannot exist, are what the act may claim.
3. Per-OD spread is real in every arm (retrieval hits 2.1 on two ODs); n=6 ODs, descriptive.

## A7: gap-closure restatement + the transfer-decay figure (`scratch/a7_gap_closure.py`)

**Why (CRITIQUE_12-07-26 §3.2):** ratio-to-equilibrium flatters thin-headroom cells. Gap closure
= (loss_det - policy) / (loss_det - equilibrium) measures the fraction of the
deterministic-to-equilibrium gap actually closed (1 = equilibrium play, 0 = no better than the
deterministic optimum, negative = worse).

**Design:** recompute per-OD gap closure EXACTLY from saved artefacts at the standing deployable
read of each act: gen15 (held-out ODs), gen16 (held-out Gdansk), gen22 (held-out Istanbul), the
whole-Kyiv row, plus the two headline instances (gen14 numbers). Produce
`assets/transfer_gap_closure.png` (gap closure vs transfer distance) and per-act tables here. No
bars: this is a metric restatement, reported as measured.

### A7 RESULT (2026-07-12): the gap-closure ladder decays 0.90 -> 0.04 across transfer distance; Istanbul/Kyiv close little to no gap

Exact per-OD recomputation from the saved JSONs at the select-on-train read
(artefacts `models/runs/a7_gap_closure.json`, `a7_aggregate_closure.json`; figure
`assets/transfer_gap_closure.png`):

| act | per-cell mean | median | cells <= 0 | **headroom-weighted aggregate** |
|---|---|---|---|---|
| trained MC 35-159 (gen14, n=10) | 0.899 | - | 0/1 | **0.899** |
| trained SC 33-71 (gen14, n=10) | 0.828 | - | 0/1 | **0.828** |
| held-out ODs, same graph (gen15, 18 cells) | 0.541 | 0.625 | 1 | **0.539** |
| held-out CITY Gdansk (gen16, 18 cells) | 0.427 | 0.380 | 1 | **0.450** |
| rotation Istanbul (gen22, 18 cells) | -0.140 | -0.089 | **11** | **0.199** |
| whole-Kyiv scale row (5 cells) | -0.494 | -1.029 | 3 | **0.036** |

Reading rules: the unweighted per-cell mean explodes on thin-headroom cells (denominator det-eq
tiny), so the fair single number per act is the HEADROOM-WEIGHTED aggregate
(sum(det - policy)/sum(det - eq)); both are reported. **What changes (binding for the storyline):**
(1) the transfer-difficulty ladder restated in calibration content: **0.90 (trained) -> 0.83 ->
0.54 (held-out OD) -> 0.45 (held-out city) -> 0.20 (Istanbul) -> 0.04 (Kyiv)**: an honest, clean,
figure-worthy decay. (2) gen22's "PASS" and the Kyiv "partial pass" must be worded as
"beats random-init; closes little (0.20) to essentially none (0.04) of the
deterministic-to-equilibrium gap": on the majority of Istanbul cells (11/18) the deployed policy
does not beat the deterministic optimum. (3) The near-end claims (gen15/gen16, ~0.5 closure)
survive restatement comfortably; the far-end claims are randomisation-level protection, not
calibrated hedging. No banked number changes; the METRIC changes what may be said.

## A8: the prevalence figure (`scratch/a8_prevalence.py`)

**Why (CRITIQUE_EXAMINER §3.8):** answers "how often does calibrated mixing matter, and were the
headline instances cherry-picked?" in one figure.

**Design:** over sampled high-connectivity OD pairs in all four cities (deg >= 3, 3-6 base
routes, k8 menus, R in [10, 14], the standing screen; target ~40 per city), compute per OD:
loss_det / equilibrium (the headroom for calibrated play) and uniform-stack / equilibrium (the
headroom over naive randomisation), N=3 K=1 mission. Plot both distributions
(`assets/prevalence.png`), mark 35-159 and 62-97 on them, and report quartiles here. No bars:
descriptive.

### A8 RESULT (2026-07-12): headroom is PREVALENT (69% of ODs at det/eq >= 2); the headlines sit in the top decile BY SCREEN DESIGN

160 ODs (40 per city, standing screen), N=3 K=1 mission (artefacts
`models/runs/a8_prevalence.json`, `assets/prevalence.png`):

| quantity | 10% | 25% | median | 75% | 90% | headline 35-159 | headline 62-97 |
|---|---|---|---|---|---|---|---|
| loss_det / equilibrium | 1.74 | 1.92 | 2.35 | 2.64 | 2.85 | **3.39** | **3.23** |
| uniform-stack / equilibrium | 1.55 | 1.80 | 2.05 | 2.28 | 2.43 | 2.14 | 3.00 |

- **69% of high-connectivity ODs have det/eq >= 2** (material calibration headroom); **93% have
  uniform-stack >= 1.5x eq** (naive randomisation clearly suboptimal almost everywhere).
- The headline instances sit at the ~top decile of det/eq: exactly what their PRE-REGISTERED
  screening criterion (ratio >= 3) selected for. The honest thesis sentence writes itself: *the
  headline instances were screened to be favourable (top decile), from a population in which
  material headroom is the norm (69% at >= 2x), so the phenomenon is prevalent and the screen
  bought margin, not existence.* One figure answers the cherry-picking question for the whole
  thesis.

### DISJOINT-BASELINE APPENDIX (2026-07-16, Block R0; oracle/eval-only)

> **Context (binding wording rule; CRITIQUE_16-07-26.md §1; probes
> `scratch/disjoint_baseline_probe.py`, `scratch/r0_screen.py`, artefacts
> `models/runs/r0_screen.json`):** the candidate menus' first routes ARE the max-flow
> decomposition, and "uniform-stack over the edge-disjoint routes" (2 lines) is the strongest
> NAIVE baseline. No comparative sentence in this ledger may claim SACRED beats "every
> uncalibrated strategy class" or that its transfer is something "standard algorithms cannot
> achieve" without the rows below beside it.

- **A6 ladder amendment:** the retrieval/distillation/generalist ladder gains its floor:
  **uniform-disjoint-stack 1.134 < distill+val 1.555 < retrieval 1.676 < adversarial
  1.733 < uniform-menu-stack 1.989 ~ random ~1.99 < DR 2.056 < vanilla 2.354.** Every amortiser
  sits ABOVE the 2-line heuristic at K=1; the act's honest content is the label-free/
  self-stopping taxonomy plus where each method's regime ENDS (labels at the wall; the
  heuristic at K >= m-1 and under adaptation).
- **A8 companion row (population, same 160-OD sample):** disjoint-stack/eq quantiles
  (10/25/50/75/90) = **1.011 / 1.057 / 1.117 / 1.260 / 1.376; only 5% of ODs >= 1.5x** (vs
  uniform-menu-stack's 93% >= 1.5x). The prevalence figure's honest caption: at K=1, calibrated
  mixing beyond naive disjointness buys 6-26% on the interquartile population — the material
  headroom lives at K >= m-1 (gen26), under adaptation (gen19/gen27), and multi-OD (B4).
