# gen05: robustness matrix on the fixed hybrid rung
Registered 2026-07-04. Results 2026-07-04 (primary and best-response rows), 2026-07-06 (seed-level sensitivity). Code `cd11f14` (build state), `324a644` (launch).
Artefacts: `experiments/gen05_portfolio.json`, runs under `models/runs/gen05_hybrid_matrix/`, `analysis/gen0506_seedlevel_stats.py`, `scripts/run_generation.py`, `scripts/train_sacred.py`, `scripts/evaluate_portfolio.py`.

## Question
Does adversarial training against a strong scripted adversary buy robustness to a held-out attack on the fixed hybrid rung?

## Game
- Arena: fixed hybrid rung (assignment plus next-hop routing, chokepoint geometry, route reach, full-block antagonist), budget 1500, max_ticks 800, `--update-every 8`, identical for both arms.
- Arms: `hybrid_vanilla` (no adversary during training) and `hybrid_scripted` (trained against the scripted `targeted` attacker), seeds {0, 1, 2}. Identical env, reward, nets and hyperparameters; only the training-time adversary differs.
- Training: 400 episodes, switch-every 25 (16 snapshots), batch size 32, hidden dim 64, device cpu, eval-every 50, threads 3.
- Evaluation: W = mean total wait over 24 paired rollout instances (static demand, so instance = episode seed). Greedy is deterministic and contributes a single trajectory.
- Selection rule: per arm, `evaluate_portfolio.py --select-best --problem hybrid` under the `targeted` attacker on validation rollout seeds 20_000_019 and up, 8 instances. The same selector is used for both arms; it is the training attack for one of them.
- Best-response attackers: one per arm, trained against the seed-0 selected checkpoint for about 300 episodes.

## Criteria
D(arm, a) = W(a) - W(none), paired per instance.

Primary: dD_gateway = D(vanilla, gateway) - D(scripted, gateway) per seed pairing (v_k against s_k), pooled across the 3 pairings. Success requires pooled dD_gateway > 0 with the paired 95% CI excluding 0, and at least 2/3 pairings individually positive.

Secondary (reported, not gating): dD under `random` and the best-response rows; the `targeted` row, which is in-distribution for the scripted arm and explicitly not claimable as held-out robustness; clean premium W(scripted, none) - W(vanilla, none), target no more than about +15%; greedy reference rows.

## Baselines
- `none`: no attacker, the clean baseline for D.
- `random`: scripted, undirected floor.
- `targeted`: scripted, the scripted arm's training attacker and the validation attacker for both arms.
- `gateway`: scripted first-maskable-edge attack under route reach. Held out from training and selection, the primary test attack.
- `br_vanilla_s0`, `br_scripted_s0`: learned best-response attackers, one per arm, seed 0 only.
- `greedy`: untrained reactive dispatcher, reference row.

## Results
Per-pairing D, paired over 24 rollout instances:

| arm | W(none) | D(random) | D(targeted) | D(gateway) |
|---|---|---|---|---|
| greedy | 847 | 1036 | 1154 | 714 |
| vanilla (s0/s1/s2) | 4739 / 4769 / 4845 | 593 / 585 / 407 | 978 / 841 / 725 | 476 / 366 / 320 |
| scripted (s0/s1/s2) | 4716 / 4605 / 4726 | 559 / 604 / 566 | 849 / 910 / 844 | 582 / 584 / 571 |

Primary: pooled dD_gateway = -192 +/- 181 (95% CI, n = 72). Criterion met 0/3 pairings (-106 +/- 313, -219 +/- 258, -251 +/- 368). The pooled CI excludes zero on the negative side.

Secondary: dD_targeted = -20 +/- 135.

Competence: learned-arm W(none) 4605 to 4845 against greedy's 847, about 5.6x worse; clean delivery about 0.5 within the horizon; Q_Spread about 0.1; deterministic-argmax evaluation delivers zero. The saturation ceiling is 6.4k (8 requests x 800 ticks).

Best-response rows, 24 paired instances:

| arm | D(br_vanilla) | D(br_scripted) |
|---|---|---|
| greedy | 1667 +/- 0 | 1667 +/- 0 |
| vanilla (s0/s1/s2) | 596 / 561 / 532 | 597 / 570 / 436 |
| scripted (s0/s1/s2) | 721 / 754 / 685 | 582 / 764 / 602 |

Against the learned arms the best-response rows fall in the same 430 to 760 band as the other attacks. Against greedy both best-response nets collapse to the identical deterministic attack trajectory at +1667, above scripted `targeted` (+1154) and `gateway` (+714).

Seed-level sensitivity, pairing as the unit (n = 3), from `analysis/gen0506_seedlevel_stats.py`: per-pairing dD_gateway {-106.2, -219.0, -251.2}, mean -192.1, SD 76.2, t(2) 95% CI [-381.3, -2.9] (excludes zero), sign consistency 3/3 negative, one-sided sign p = 0.125.
