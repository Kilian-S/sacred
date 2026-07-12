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

### A7 RESULT (appended after the run)

## A8: the prevalence figure (`scratch/a8_prevalence.py`)

**Why (CRITIQUE_EXAMINER §3.8):** answers "how often does calibrated mixing matter, and were the
headline instances cherry-picked?" in one figure.

**Design:** over sampled high-connectivity OD pairs in all four cities (deg >= 3, 3-6 base
routes, k8 menus, R in [10, 14], the standing screen; target ~40 per city), compute per OD:
loss_det / equilibrium (the headroom for calibrated play) and uniform-stack / equilibrium (the
headroom over naive randomisation), N=3 K=1 mission. Plot both distributions
(`assets/prevalence.png`), mark 35-159 and 62-97 on them, and report quartiles here. No bars:
descriptive.

### A8 RESULT (appended after the run)
