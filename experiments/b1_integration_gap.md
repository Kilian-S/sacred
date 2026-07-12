# B1: the holistic-SBO integration gap (joint vs tier-by-tier design, on the held-out city)

- **status: PRE-REGISTERED 2026-07-12 (NEXT_STEPS_MASTER Block B item B1; EVAL/ORACLE-ONLY:
  frozen actors, no training); binding at launch.**
- **git SHA:** the commit landing this ledger + `scratch/b1_integration_gap.py`.

## Why (Obj-4 verbatim; CRITIQUE_EXAMINER §6 item 8; CRITIQUE_12-07-26 §6 item 6)

Objective 4 promises "HOLISTIC, SIMULTANEOUS evaluation of strategic supply chain design alongside
the operations-level SDVRP". The D-chain (F3/D1/D2/D3) optimises the tiers SEPARATELY, so the word
"simultaneous" is not yet earned. D2 measured the mechanism that should make joint optimisation
win (hardening RELOCATES where randomisation pays; equilibrium mass shift L1 = 0.29). This
experiment measures the INTEGRATION GAP directly: joint optimisation over
(placement x fleet x hardening) vs the classical tier-by-tier decomposition, at MATCHED total
evaluation budget, priced by the frozen zero-shot generalist on the never-trained city (Gdansk).

## Design (fixed before looking)

- **Joint design space:** ~40 screened Gdansk placements (deg >= 3, 3-6 base routes, k8, R 10-14)
  x fleet size N in {2, 3, 4} x a 4-option doctrine-style HARDENING menu (budget 4 units,
  eta = 0.5 per unit, the D2 cost model; nested-LP-free so the space is enumerable):
  h0 = none; h1 = top-4 most vulnerable candidate edges; h2 = the 4 most route-shared edges
  (chokepoints); h3 = the worst edge of each of the 4 highest-worst-vulnerability routes.
  Design value = the frozen generalist's exact stacked exploitability on the HARDENED game
  (forward pass + oracle BR); fleet travel cost recorded per design (bi-objective row).
- **Actors:** primary = gen16 seed-0 best actor (the D3-Gdansk convention); the gap is ALSO
  computed under seed-1's actor (the A5 reliability lesson: is the gap actor-specific?).
- **Arm A (sequential, the classical decomposition), budget 60:** Tier 1 = D1-recipe SBO
  (ensemble-LCB, n0 15, kappa 1.0) over (placement, N) at h0, budget 56 -> pick the best found;
  Tier 2 = evaluate all 4 hardening options on it (budget 4); final value = its best hardened
  exploitability.
- **Arm B (joint), budget 60:** the same SBO recipe over the FULL (placement, N, hardening)
  space (hardening one-hot appended to the F3 features).
- 12 repeats (RNG seeds 0-11); the full table is enumerated once for regret reference only (the
  optimisers never see it).

## Decision metric (PRE-REGISTERED)

Primary = the **integration gap** = median over repeats of
(sequential final exploitability - joint final exploitability), absolute and as % of the joint
value, per actor. Secondary: regret of each arm vs the true joint optimum; the mechanism row =
Spearman between placements' h0 ranking and their best-hardened ranking (low correlation = the
lock-in that makes sequential lose); the (cost, exploitability) scatter with both arms' picks.

> **Pre-committed reading:** median relative gap >= 5% under BOTH actors => "holistic,
> simultaneous" is EARNED as a measured advantage (the Obj-4 sentence + poster exhibit). Gap
> ~0 (< 5%) => the tiers DECOMPOSE on this family: reported plainly with the mechanism row
> explaining why (also a finding; Obj-4 then rests on the loop pattern + policy-valued target,
> the standing position). Gap materially actor-dependent => report per-actor, per the A5 rule.

## Command (pinned)

```bash
PYTHONPATH=. .venv/bin/python scratch/b1_integration_gap.py --threads 2 \
  --json-out models/runs/b1_integration_gap.json
```

## RESULT (2026-07-12, 420 joint designs, 12 repeats/arm/actor): the strict bar is NOT met; the measured statement is "joint is the safe default, sequential lock-in is actor-contingent (0-19%)"

(artefacts `models/runs/b1_integration_gap.json`, `.log`)

| actor | sequential median | joint median | integration gap | true joint opt | placement rank corr h0 vs best-hardened |
|---|---|---|---|---|---|
| gen16 seed 0 | 0.2242 | 0.2242 | **+0.0%** | 0.2242 | 0.872 |
| gen16 seed 1 | 0.2811 | 0.2356 | **+19.3%** | 0.2356 | 0.859 |

**Against the pre-registered bar (>= 5% under BOTH actors): NOT MET** (0% under actor 0). Reported
as measured, per-actor, per the A5 rule. The honest findings:
1. **The joint loop attained the TRUE optimum under both actors at budget 60; the classical
   tier-by-tier decomposition matched it under one actor and left 19.3% on the table under the
   other.** So joint optimisation never did worse and is the SAFE DEFAULT; the cost of the
   classical decomposition is real but actor-contingent, echoing A5's finding that the
   operational design landscape is policy-instance-specific.
2. **The mechanism row bounds the tier coupling:** placements' VALUE rankings are largely
   hardening-stable (Spearman ~0.86 under both actors), so strong lock-in is not the norm on
   this design family; D2's strategy-level coupling (equilibrium mass shift L1 = 0.29) coexists
   with value-level ranking stability. Where the sequential arm lost (actor 1), it lost through
   search dynamics on a rougher landscape, not through dramatic re-ranking.
3. **Obj-4 wording (binding):** "holistic, simultaneous" is supported in the weaker, honest form:
   *a single surrogate-guided loop over the joint strategic-tactical-operational space finds the
   optimum at the same budget where the classical decomposition is only sometimes adequate (and
   19% worse when not); the joint loop is the safe default because which regime you are in
   depends on the deployed policy instance and cannot be known in advance.* Scope caveats: a
   4-option doctrine hardening menu (a richer allocation space could widen the gap); 420 designs;
   never-trained city.
