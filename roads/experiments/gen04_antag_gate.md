# gen04: best-response attacker gate after the motion-observability fix
Registered 2026-07-04. Results 2026-07-04. Code `af056ac` (edge features 2 to 4: directed truck occupancy and progress fraction).
Artefacts: `experiments/gen04_gate.json`, run `models/runs/gen04_antag_gate/br_fixed_vanilla_s0_seed0`, `scripts/train_sacred.py`, `scripts/evaluate_portfolio.py`.

## Question
With directed truck occupancy and progress added to the edge features, does a retrained best-response antagonist clearly beat the random attacker?

## Game
- Arena: dynassign, everything except the edge features held from gen03.
- Frozen protagonist (the victim): `models/runs/gen03_robustness_dynassign/vanilla_seed0/snapshots/protagonist_ep450.pt`, node 13 / edge 2, which slices the new features away so the defender is byte-identical to gen03's.
- Antagonist: fresh 13/4-dim net, 300 episodes, `--train-antagonist-only`, switch-every 50, eval-every 0, seed 0, threads 4.
- Gate evaluation: `vanilla_s0` crossed with {none, random, targeted, br_fixed} on 16 validation instances (seeds 20_000_019 to +15). Test instances untouched.
- Reference points carried from gen03 (test instances, same frozen defender): D(random) about 2108 +/- 338, D(br_old) about 1083 +/- 327, D(targeted) about 5660 +/- 384.

## Criteria
W = mean total wait; D(a) = W(a) - W(none), paired per instance on the 16 validation instances.

- PASS: D(br_fixed) >= 1.25 x D(random).
- STRONG PASS: additionally D(br_fixed) >= 0.5 x D(targeted).
- FAIL: co-evolution is parked for Phase 3 and the scripted-adversarial training arm is used instead.

## Baselines
- `none`: no attacker, the clean baseline for D.
- `random`: scripted, uniform over maskable edges.
- `targeted`: scripted heuristic, blocks ahead of the truck nearest its goal.
- `br_fixed`: the retrained best-response antagonist, trained with the motion features present.

## Results
Paired on the 16 validation instances, same frozen defender as gen03:

| attacker | D (degradation) |
|---|---|
| random | 1984 +/- 447 |
| targeted (scripted) | 5868 +/- 647 |
| br_fixed (retrained, with motion features) | 1663 +/- 517 |

Ratios: br/random = 0.84 against the 1.25 PASS bar; br/targeted = 0.28 against the 0.50 STRONG bar. PASS criterion met 0/1. STRONG PASS criterion met 0/1.

Paired difference D(br) - D(random) = -321 +/- 807.

Antagonist training telemetry over the 300 episodes: true episode reward fell 8710 to 8120, Q inflated 37 to 113, critic loss moved from about 250 to 265, alpha stayed pinned at 1.0 and policy entropy at about 2.1, against an action space of about 120 options and a 0.5*ln(N) entropy target.

The gen03 result this gate follows up: learned best-response D of about 0.6k to 1.9k against random at about 1.7k to 2.1k.
