# Generation: gen04_antag_gate — can the learned adversary beat RANDOM once it can see motion?

- **git SHA:** `af056ac` (N1 fix: edge features 2→4 = directed truck occupancy + progress)
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

## Result (2026-07-04) — **GATE FAIL.** Observability was necessary but not sufficient.

On the 16 validation instances (paired, same frozen defender as gen03):

| attacker | D (degradation) |
|---|---|
| random | 1984 ± 447 |
| targeted (scripted) | 5868 ± 647 |
| **br_fixed (retrained, WITH motion features)** | **1663 ± 517** |

Ratios: br/random = **0.84** (PASS needed ≥ 1.25); br/targeted = 0.28 (STRONG needed ≥ 0.50).
Paired D(br) − D(random) = −321 ± 807. The seeing attacker is still statistically ≈ random.

**Training curve — the same failure signature as gen03, so the bottleneck is deeper than
observability:** true episode reward *fell* 8710→8120 while Q inflated 37→113 (critic
over-estimation), critic loss never converged (~250→265), and **α stayed pinned at 1.0 with
policy entropy ~2.1** — with a ~120-option flat action space and the 0.5·ln(N) entropy target,
the max-entropy objective *requires* a near-uniform attack policy at these advantage magnitudes.
A near-uniform policy over the mask ≈ the random attacker — which is exactly what both gates
measured. Contributing causes (per CRITIQUE.md): reward SNR (~1–2k controllable effect on an ~8k
uncontrollable queue baseline) and γ=0.99/tick myopia vs 100+-tick damage horizons.

**Consequence (pre-registered):** co-evolution is **parked**; the back-pocket
**scripted-adversarial arm is promoted** for Phase 3. Additional recommendation for Kilian: keep
one ATLA arm in the hybrid matrix anyway — in the hybrid arena the **route-reach mask itself aims
the attacks** (the gen03-era scripted route-reach attacker was literally "first maskable edge" and
cost greedy +40…+184%), so even a near-uniform learned antagonist applies real pressure there;
that keeps the thesis's namesake mechanism in the headline experiment with an honest mechanism
either way. Optional cheap follow-up (gen04b, ~2 h): one gate re-run with a lowered antagonist
entropy target to test the "entropy pinning" hypothesis directly.

Artifacts: `experiments/gen04_gate.json`, run `models/runs/gen04_antag_gate/br_fixed_vanilla_s0_seed0`.

## Decisions (2026-07-04, Kilian)

- **Scripted-adversarial arm: PROMOTED** — Phase 3 trains the protagonist against the scripted
  targeted attacker (`gen05_hybrid_matrix`).
- **ATLA co-evolution rider arm: BACK POCKET** — not in Phase 3; revisit only if the scripted
  arm's result motivates it (the hybrid route-reach mask argument stands recorded above).
- **gen04b (lowered antagonist entropy target re-gate): BACK POCKET** — the entropy-pinning
  hypothesis stays documented but untested for now; a ~2 h re-gate if the thesis's diagnosis
  chapter needs the direct counterfactual.
