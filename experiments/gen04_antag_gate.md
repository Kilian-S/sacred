# Generation: gen04_antag_gate — can the learned adversary beat RANDOM once it can see motion?

- **git SHA:** _fill at launch_ (N1 fix: edge features 2→4 = directed truck occupancy + progress)
- **date opened:** 2026-07-04
- **status:** LEDGER OPEN

## Question (fixed before looking)

gen03 found the learned best-response attacker **weaker than uniform-random blocking** (D ≈
0.6–1.9k vs random ≈ 1.7–2.1k) and diagnosed missing motion observability (mid-edge trucks
invisible). After the N1 fix (edge_attr gains directed occupancy count + progress fraction):
**does a retrained best-response antagonist now clearly beat the random attacker?** This gates all
Phase-3 co-evolution CPU: if a dedicated best-response attacker still can't beat random even with
the motion state, ATLA co-evolution cannot produce meaningful adversarial pressure on this compute
budget, and the back-pocket option (scripted-adversarial training arm, D3 amendment recorded
2026-07-04) becomes the Phase-3 recommendation.

## Setup

One BR training, everything else held from gen03: frozen protagonist = **the same**
`vanilla_seed0/protagonist_ep450.pt` (node 13 / edge 2 — slices new features away, so the
defender is byte-identical to gen03's), fresh 13/4-dim antagonist, 300 episodes, dynassign
config. Gate eval: `vanilla_s0` × {none, random, targeted, br_fixed} × **16 validation instances**
(seeds 20_000_019…+15; test instances stay untouched).

## Decision criteria (PRE-REGISTERED)

- **PASS:** D(br_fixed) ≥ 1.25 × D(random), paired on the 16 validation instances.
- **STRONG PASS:** additionally D(br_fixed) ≥ 0.5 × D(targeted) — within 2× of the scripted
  heuristic.
- **FAIL:** co-evolution is parked for Phase 3 (vanilla + scripted-adversarial arms instead);
  the learned-adversary limitation becomes a documented thesis finding.

Reference points from gen03 (test instances, same frozen defender): D(random) ≈ 2108 ± 338,
D(br_old) ≈ 1083 ± 327, D(targeted) ≈ 5660 ± 384.

## Commands

```bash
PYTHONPATH=. python scripts/train_sacred.py --problem dynassign --train-antagonist-only \
  --protagonist-snapshot models/runs/gen03_robustness_dynassign/vanilla_seed0/snapshots/protagonist_ep450.pt \
  --episodes 300 --switch-every 50 --eval-every 0 --seed 0 --group gen04_antag_gate --tag br_fixed_vanilla_s0 --threads 4

PYTHONPATH=. python scripts/evaluate_portfolio.py --problem dynassign \
  --policy vanilla=models/runs/gen03_robustness_dynassign/vanilla_seed0/snapshots/protagonist_ep450.pt \
  --br fixed=models/runs/gen04_antag_gate/br_fixed_vanilla_s0_seed0/antagonist/actor.pt \
  --attackers none,random,targeted,br_fixed --instances 16 --seed-base 20000019 \
  --out experiments/gen04_gate.json
```

## Result

_(to be filled)_
