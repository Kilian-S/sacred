# Generation: gen27_dynamic_generalist (Block R2: zero-shot transferable DYNAMIC hedging — the rescued ZST act)

- **status: PRE-REGISTERED 2026-07-16 (Block R, `NEXT_STEPS_MASTER.md`; Kilian's full autonomous
  launch authority). Binding at launch; smoke gate before the batch.**
- **git SHA:** the commit landing this ledger + `scripts/train_dyn_generalist.py`.

## Why (CRITIQUE_16-07-26.md §6; the disjoint-baseline finding)

The static ZST act cannot claim superiority over naive methods at K=1 (the max-flow heuristic
transfers at 1.13x eq with zero training; R0a appendices). The DYNAMIC register is the one place
the aim's "policies that standard algorithms cannot achieve" is true BY CONSTRUCTION: against a
pattern-of-life adversary (softmax-BR to the defender's realised w-window), every STATIC object
— the disjoint heuristic, the LP equilibrium mixture, distilled or retrieved policies — is
mathematically capped at iid_eq; only history-conditioned play goes below it. gen19 proved the
mechanism single-instance (0.050 vs cap 0.147, causal no-window control, worst-case row); gen16
proved multi-city transfer of static hedging. gen27 composes them: ONE history-aware policy,
trained on three cities, evaluated zero-shot on a held-out city's dynamic games.

## Design (locked; gen19 + gen16 recipes verbatim, decisions recorded)

- **Game per instance:** fleet-route stacked; adversary = softmax-BR (tau=0.15) to the trailing
  w=3 realised-route window — the gen19 operating point, chosen from its banked sensitivity grid
  (tau=0.05 is degenerate-dodgeable; w=3 the non-degenerate point). Analytic expected-mission-
  failure reward; episodes = S=40 sorties chained with gamma=0.95; window cleared per episode.
- **Pools:** gen16 verbatim — kaliningrad + east_london + istanbul x 6 ODs (train),
  GDANSK x 6 held out entirely; pool-seed 0; N=3, K=1, band 0.15-0.95, k8 menus.
- **Conditioning:** per-instance menus + [cost, worst-vuln, window-frequency] route features on
  every observation (the gen15 per-transition plumbing + the gen19 third column), head-term lr
  3e-2. NO route_bias (identity does not transfer).
- **Yardsticks per instance (computed at pool build, exact):** static_det, iid_eq (the static
  cap), history_opt (RVI over the R^w window MDP) — the gen19 oracle, per instance.
- **Arms:** history-aware x seeds {0,1,2}; NO-WINDOW causal control x 1 seed (window feature
  zeroed; must land ~iid_eq as in gen19).
- **Selection:** select-on-TRAIN (mean train-instance ratio-to-iid_eq; standing default);
  select-on-test dual-reported as the optimistic bound; final iterate disclosed.
- **Budget:** 12,000 sorties (the gen16 budget), eval-every 500 (train n=400/instance, held-out
  n=1000/instance), per-eval ckpts. Timing probe + 240-sortie smoke gate BEFORE the batch
  (gate = plumbing sound + rw[2] trending negative, the gen19 anti-repeat signature; NOT a
  performance bar — smokes validate plumbing, not dynamics).

## Decision metric (PRE-REGISTERED)

Per held-out instance: stationary per-sortie expected mission failure vs ITS pattern-of-life
adversary (the gen19 estimator), ratio to ITS iid_eq.

> **PRIMARY (the unique claim): at the select-on-train checkpoint, held-out-GDANSK mean
> ratio-to-iid_eq < 1.0 AND < 1.0 on >= 4/6 ODs, on >= 2/3 seeds.** Beating iid_eq zero-shot =
> beating every static method (heuristic, LP, distillation, retrieval) by construction.
> **STRONG:** pooled held-out mean per-sortie loss <= 2x the per-instance history_opt mean.
> **CAUSAL CONTROL:** the no-window arm lands at ratio ~1.0 (the cap; gen19's landed 1.007).
> **REPORTED ROW (not gated):** worst-case = the marginal route mixture's single-shot
> exploitability under each OD's oracle BR vs its V_eq (gen19's premium was +6%; zero-shot will
> be looser; reported honestly).
> **Branches:** PASS = the crown jewel (zero-shot transferable dynamic hedging). PARTIAL
> (train-city ratios < 1, held-out >= 1) = dynamic hedging learned, transfer boundary measured.
> FAIL = gen19 stays the single-instance positive; the boundary is the result. All writable.

## Commands (pinned; batch via `scratch/gen27_batch.sh` after the smoke gate)

```bash
PYTHONPATH=. .venv/bin/python scripts/train_dyn_generalist.py \
  --sorties 12000 --eval-every 500 --seed $S --threads 3 \
  --json-out models/runs/gen27_dyn_generalist/seed$S.json \
  --ckpt-dir models/runs/gen27_dyn_generalist/seed${S}_ckpts
# control: --no-window --seed 0, json/ckpt paths suffixed _nowin
```

## RESULTS (appended per step; nothing above changes after launch)

### AMENDMENT (2026-07-16, DURING the batch, BEFORE any result was read; disclosed)

Two integrity additions, both decided before looking at any training outcome:
1. **Per-seed yardsticks + the LP-degeneracy wobble.** The three seeds' pool-prep logs show
   iid_eq differing by ~1-2% across processes on identical games (e.g. OD 193-278: 0.187 vs
   0.184): the equilibrium LP has degenerate optima and the solver's vertex choice is not
   process-stable. static_det and history_opt are identical (they do not depend on the vertex).
   Handling: each seed's ratios are scored against ITS OWN stored refs (as built); the wobble is
   disclosed and immaterial at the expected effect size (gen19: 66% below the cap).
2. **The "every static method" claim gets MEASURED rows, not by-construction wording.** iid_eq
   is the static EQUILIBRIUM mixture's value against this adversary — not the best static value.
   The pre-registered primary (beat iid_eq) therefore only certifies beating the LP mixture. At
   results time the following oracle-exact rows are added per held-out instance (eval-only,
   `scratch/gen27_static_rows.py`): the UNIFORM-DISJOINT heuristic's static value, the INV-VULN
   heuristic's static value, and a multi-start local-search STATIC OPTIMUM (disclosed as local).
   The act's claim is worded against whichever of these rows the results actually clear; the
   primary bar itself is unchanged (iid_eq).
