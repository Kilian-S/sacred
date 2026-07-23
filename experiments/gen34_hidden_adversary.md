# Generation: gen34_hidden_adversary (Phase-1 frontier hardening, point 1: the enemy is no longer a single hand-chosen rule)

**status: PRE-REGISTERED 2026-07-23.** Mandate: Kilian's 2026-07-23 direction ("make the current
frontier more complex, without damaging the claims that have already been made"; Phase-1 points
1-3 agreed in-conversation). Ledger committed BEFORE any trainer code exists; results are
appended below the RESULTS line and nothing above it changes. Training launch requires Kilian's
explicit go.

**git SHA:** the commit landing this ledger (pre-registration); build and result steps pin their
own SHAs at their records.

## Why

The gen27/gen31/gen32 dynamic positives share one standing legitimacy caveat (recorded in all
three ledgers): the adaptive enemy is a SINGLE hand-chosen rule, and in the aerial acts it was
designed under full freedom explicitly biased toward the method winning. The honest reading of
those acts is "a learnable niche exists and the policy captures it". gen34 removes the caveat at
its root: the enemy TYPE is drawn per episode, hidden, from a doctrine family; no defender tuned
to one type can be robust to the family, and the exact value of type-INFERENCE is computable.
This lifts the gen27 claim structure one level: gen27's policy beat the best static object by
conditioning on its own window; gen34's policy must beat the best TYPE-BLIND object (the exact
blind cap) by conditioning on observed enemy behaviour.

The design probe (`scratch/gen34_family_probe.py`, oracle-exact, 2026-07-23,
`models/runs/gen34_family_probe.json`) established the game is non-degenerate:

| instance | OMNI cap (type known) | BLIND cap (exact, type-blind optimum) | inference gap | fitted Bayes-MAP (playbook) | anti-repeat rule | iid_eq mixture |
|---|---|---|---|---|---|---|
| kaliningrad 35-159 | 0.0527 | 0.0717 | 1.36x | 0.0564 | 0.0757 | 0.1469 |
| kaliningrad 62-97 | 0.0487 | 0.0756 | 1.55x | 0.0546 | 0.0756 | 0.1552 |
| gdansk 249-95 (held-out class) | 0.0589 | 0.1198 | **2.04x** | 0.0690 | 0.1767 | 0.2257 |

The specialist cross-tables (in the JSON) show every single-type counter-doctrine blows up
off-diagonal (up to ~28x its diagonal): no fixed rule survives the family. The playbook row
(Bayes-MAP over KNOWN member forms, then the member's exact specialist policy) captures ~80% of
the inference gap and is the disclosed fitted anchor: the trained policy gets the OBSERVATIONS
but not the playbook.

**Yardstick provenance (binding):** every exact reference in this act is computed by
`scratch/dyn_exact.py` (Karp minimum-mean-cycle, cross-checked by damped RVI). The undamped RVI
`history_opt` of `oracle_refs` is NOT used anywhere (see the 2026-07-23 corrected-yardstick
appendices in the gen19/gen27 ledgers and `models/runs/gen35_mmc_check.json`).

## The game (pinned)

Everything not stated here is the gen27 recipe verbatim (`scripts/train_dyn_generalist.py`
defaults; N=3, K=1, band (0.15,0.95), k_extra=8, fleet-route menu-select, mission objective,
interception_loss 10.0, episode = 40 sorties, w=3, gamma 0.95).

- **Member family (uniform draw per episode, hidden from the defender), on each instance's
  stacked loss matrix L** (exact forms as implemented in `scratch/gen34_family_probe.py`
  `member_fns`, which is the normative definition):
  - M1 `reactive`: softmax-BR tau=0.15 to the trailing-window route counts (the gen19/27 incumbent).
  - M2 `sharp`: softmax-BR tau=0.05.
  - M3 `anticipatory`: softmax-BR tau=0.15 to the anti-repeat prediction (uniform over routes
    with zero window count; fallback uniform).
  - M4 `doctrine`: window-independent softmax-BR tau=0.15 to the instance's static equilibrium
    mixture.
  - M5 `scattergun`: uniform over interdiction sets.
- **Defender observation:** the gen27 route-feature head gains TWO placement-observation
  columns (the realised interdiction set j_t is revealed after each sortie, as convoys/recon
  would report an ambush): col 4 = minmax over routes of L[r, j_{t-1}] (last observed
  placement's damage-if-taken); col 5 = minmax of an EWMA (decay 0.8, reset per episode) of
  L[r, j_s] over the episode's observed placements. Window feature (col 3) unchanged. The
  causal control (`--no-intel`) zeroes cols 4-5 ONLY (window stays).
- **Pools:** gen27 verbatim: train = kaliningrad + east_london + istanbul, 6 ODs each,
  pool-seed 0; held-out = gdansk, 6 ODs, pool-seed 0 (never tuned on).
- **Per-instance pre-computed refs (stored in the run JSON):** per-member omni values, blind
  cap, inference gap, anti-repeat/rotation/iid_eq rule rows, fitted Bayes-MAP row - all via
  `scratch/dyn_exact.py` + the probe's member machinery.
- **Trainer:** NEW script `scripts/train_family_generalist.py` (gen27 trainer + member sampling
  per episode + the two intel columns + per-member refs). ADDITIVE ONLY: no existing file's
  behaviour changes; suite green with raw output pasted at the build record before any launch.

## Baseline family (pre-registered, non-negotiable; all exact except the MC playbook row)

Type-blind (all bounded below by the BLIND cap, by construction): the blind-optimal window
policy (the cap itself), anti-repeat over disjoint routes, best-of-20-orders rotation, iid_eq
static mixture, uniform/inv-vuln disjoint static stacks, best fixed route. Type-informed
(disclosed as playbook-fitted): Bayes-MAP + specialist (MC, 3000 episodes, MAP threshold 0.6).
Plus the worst-case committing-adversary row (policy marginal vs oracle one-shot BR, vs the
one-shot v_eq), reported ungated as in gen27.

## Decision metric (PRE-REGISTERED)

Deployable value per held-out instance = episodic mixture value: mean per-sortie expected loss
over 40-sortie episodes with the member resampled uniformly per episode (the training regime;
analytic expected-loss estimator as gen19/27), at the select-on-TRAIN best checkpoint
(select-on-test dual-reported as the optimistic bound). Ratio = value / that instance's BLIND
cap.

> **PRIMARY (the unique claim): pooled held-out ratio-to-blind-cap < 1.0, AND < 1.0 on >= 4/6
> held-out ODs, on >= 2/3 seeds.** Beating the blind cap is impossible for ANY object that does
> not use the placement observations - every static mixture, composed rule and window-only
> policy is bounded by it. This is the claim no naive baseline can share, whatever its form.
> **STRONG: pooled capture fraction >= 0.5** (value <= blind - 0.5 x (blind - omni), pooled).
> **CAUSAL CONTROL: the `--no-intel` arm (cols 4-5 zeroed, window kept) lands at ratio >= 0.95
> on the pooled held-out set** (it cannot beat the blind cap; expected ~1.0-1.1 with the same
> drift disclosure as gen27's control).
> **REPORTED ROWS (not gated):** per-member per-instance values; the playbook row beside the
> policy (expected AHEAD of it - disclosed, as gen27 disclosed its composed rule); the
> worst-case committing row; train-pool rows; final-iterate drift.
> **Branches (all writable):** PASS = the first trained result that beats every possible
> type-blind object by measured inference, zero-shot on a held-out city, with the enemy family
> disclosed and the playbook cap beside it. PARTIAL (pooled < 1.0 but per-OD/seed clauses miss)
> = the inference signal exists but is not robust; report per-clause. FAIL = the boundary
> finding: at this scale the policy class cannot convert placement observations into type
> inference; the exact inference-gap landscape (this probe) plus the brittleness cross-table
> stand as the act's oracle contribution, gen29-style.

## Design decisions ledgered (one line each)

1. K=1: the family axis is isolated from the budget axis (gen35 owns K>1); composing them is
   future work, not this act.
2. Member set of five: spans reactive/anticipatory/committed/indiscriminate archetypes while
   every member stays exactly solvable; no member was tuned against the defender (M1-M5 fixed
   at probe time, before any training).
3. Uniform member prior: the least-informative choice; a non-uniform prior would be a tunable
   lever (excluded).
4. Two intel columns only, computed from observables the game itself reveals: no oracle
   quantity enters the observation.
5. Episodic mixture value (40 sorties) is the deployable metric: long-run per-member stationary
   rows are reported but not gated (inference within a finite episode is the story).
6. The playbook row is IN the family and expected ahead: the act's claim is beating the blind
   cap without the playbook, not beating the playbook.
7. Numbers live only in this ledger and its JSONs; anchors above are pinned from the committed
   probe.
8. Thread caps: any multi-process launch exports OMP/VECLIB caps + torch threads per SYSTEM.md.

## Commands (pinned at build; launch = Kilian's explicit go)

```bash
# build gate (suite + refs reproduction) then, per seed S in 0 1 2 (3-parallel), + control:
OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. .venv/bin/python \
  scripts/train_family_generalist.py --sorties 12000 --eval-every 500 --seed $S --threads 3 \
  --json-out models/runs/gen34_hidden_adversary/seed$S.json \
  --ckpt-dir models/runs/gen34_hidden_adversary/seed${S}_ckpts
# control: --no-intel --seed 0, same budget
```

## Compute envelope

gen27's measured scale (~8 h/seed at 12k sorties, 3-parallel) + per-episode member sampling
(negligible) + pool-build refs (~2 min total). Plan: one overnight batch for 3 seeds, control
the following night. Hard ceiling: 2 nights; no chase beyond the pre-registered budget.

## RESULTS (appended per step; nothing above changes after launch)
