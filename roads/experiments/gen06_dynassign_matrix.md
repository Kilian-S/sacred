# gen06: robustness matrix in the competent dynamic-assignment arena
Registered 2026-07-05. Results 2026-07-05 (primary), 2026-07-06 (best-response rows and post-hoc analyses). Code `cfabc90` (build state), `0bc6ec3` (launch).
Artefacts: raw portfolio JSONs; training runs under the `gen06_dynassign_matrix` group; `analysis/gen06_telemetry_probe.py`, `analysis/gen06_snapshot_robustness.py`, `analysis/gen06_snapshot_robustness.json`, `analysis/gen06_matched_temperature.py`, `analysis/gen0506_seedlevel_stats.py`; `scripts/run_generation.py`, `scripts/train_sacred.py`, `scripts/evaluate_portfolio.py`.

## Question
Does adversarial training against a strong scripted attacker buy robustness to a held-out attack, in an arena where policies demonstrably learn to competence?

## Game
- Arena: dynassign, lambda = 0.06, budget 4000, 800 episodes, switch-every 50, UTD 1, batch size 32, hidden dim 64, device cpu, eval-every 50, threads 3.
- Arms: `vanilla` (no training adversary) and `dynassign_scripted` (trained against `pathrand`), seeds {0, 1, 2}. Identical env, reward, nets and hyperparameters; only the training-time adversary differs. Both arms trained fresh on this code state.
- Training attacker: `pathrand`, the first blockable edge on a uniformly random goal-committed truck's path. Route-aimed but stochastic across trucks, which keeps `targeted` fully held out.
- Test instances: demand seeds 10_000_019 to +29, 30 paired instances. Validation instances: 20_000_019 to +7. Protagonists act stochastically.
- Selection rule: per arm, minimum mean attacked wait under `pathrand` on 8 validation instances. Noted asymmetry: `pathrand` is the training attack for one arm. Selection outcomes are not reported.
- Best-response attackers: one per arm against the seed-0 selected checkpoint, 300 episodes.

## Criteria
W = mean total wait over the 30 paired test instances. D(arm, a) = W(a) - W(none), paired per instance.

Primary: pooled dD_targeted = D(vanilla, targeted) - D(scripted, targeted) across the 3 seed pairings. Success requires pooled dD_targeted > 0 with the paired 95% CI excluding 0, and at least 2/3 pairings individually positive.

Competence precondition: each arm's W(none) must land within about 15% of greedy's clean W, else that arm's rows are flagged competence-compromised.

Pre-registered interpretive branches: dD_pathrand > 0 with dD_targeted about 0 gives attack-specific hardening without transfer; both about 0 with competence met gives a null with competence valid. Reporting rule: pooled instance-level CI, per-pairing sign consistency and the 3-pairing t sensitivity, always together.

## Baselines
- `none`: no attacker, the clean baseline for D.
- `random`: scripted, undirected floor.
- `pathrand`: scripted, route-aimed, stochastic across trucks. In-distribution for the scripted arm and the validation attacker for both arms.
- `targeted`: scripted, blocks ahead of the truck nearest its goal. Held out from training and selection, the primary test attack.
- `br_vanilla_s0`, `br_scripted_s0`: learned best-response attackers, seed 0, reference rows.
- `greedy`: untrained reactive dispatcher, reference row.

## Results
Competence gate: met 6/6 arms. W(none) sits +5.5% to +7.0% above greedy (6538 to 6635 against 6200). Attacked W is about 8k to 13k, an unbounded regime with no ceiling compression.

| arm | W(none) | D(random) | D(pathrand) (in-distribution for scripted) | D(targeted) (held out) |
|---|---|---|---|---|
| greedy | 6200 | 1718 | 5035 | 4921 |
| vanilla (s0/s1/s2) | 6618/6635/6590 | 1751/1807/2027 | 5174/5749/5706 | 5196/5627/5882 |
| scripted (s0/s1/s2) | 6538/6609/6600 | 1890/1650/2180 | 6528/6052/6374 | 6575/6413/6361 |

Primary: pooled dD_targeted = -881 +/- 284 (95% CI, n = 90). Criterion met 0/3 pairings (-1379 +/- 519, -785 +/- 510, -479 +/- 400). The pooled CI excludes zero on the negative side.

Secondaries: dD_pathrand = -775 +/- 244, 0/3 pairings positive; dD_random = -45 +/- 221. Clean premium about 0.

Robustness ranking under the held-out attack: greedy 4921, vanilla 5196 to 5882, adversarially trained 6361 to 6575.

Best-response rows, 30 paired instances:

| arm | D(br_vanilla) | D(br_scripted) |
|---|---|---|
| greedy | 577 | 1715 |
| vanilla (s0/s1/s2) | 1086 / 949 / 928 | 1749 / 1908 / 1500 |
| scripted (s0/s1/s2) | 1148 / 1023 / 841 | 1324 / 1864 / 1804 |

Both learned best-response attackers sit at or below the random attacker's 1700 to 2200 and 3 to 4 times below the scripted attacks.

### Training telemetry (`analysis/gen06_telemetry_probe.py`)
Windowed means from the six runs' tfevents, episodes 1 to 100 against 700 to 800:

| quantity | vanilla (3 seeds) | scripted (3 seeds) |
|---|---|---|
| SAC alpha (end) | 0.13 (all seeds) | 0.62 to 0.86 (never anneals) |
| policy entropy (end) | 0.37 to 0.39 | 0.47 to 0.52 |
| Q_Spread | 2.6 to 3.8 | 13.0 to 15.1 |
| critic loss | about 195 to 226 | about 856 to 1131 |
| training Total_Wait | about 7.0k to 7.8k | about 13.6k to 15.6k |
| training delivery rate | about 0.65 to 0.66 | 0.18 to 0.27 |
| final queue | about 17 | about 35 to 40 |

Clean periodic-eval curves are flat from about episode 50 in all six arms.

### Robustness against training time (`analysis/gen06_snapshot_robustness.py`)
Every protagonist snapshot of all six runs evaluated under both aimed attackers on the 8 validation instances. Early (episodes 50 to 200) against late (episodes 650 to 800) window means of attacked W:

| run | pathrand | targeted |
|---|---|---|
| vanilla_seed0 | 14028 to 14892 (+6.2%) | 14939 to 14930 (-0.1%) |
| vanilla_seed1 | 13970 to 15472 (+10.8%) | 14709 to 15702 (+6.8%) |
| vanilla_seed2 | 13686 to 15603 (+14.0%) | 14500 to 15924 (+9.8%) |
| scripted_seed0 | 15466 to 15178 (-1.9%) | 15999 to 15345 (-4.1%) |
| scripted_seed1 | 14388 to 14844 (+3.2%) | 15126 to 14604 (-3.4%) |
| scripted_seed2 | 15019 to 14626 (-2.6%) | 15385 to 14955 (-2.8%) |

Five of six vanilla cells are worse late. Per-point SEM is about +/- 400 to 600 on 8 validation instances; the seed1 and seed2 vanilla trends exceed it, seed0's fall within it. Raw per-snapshot values in `analysis/gen06_snapshot_robustness.json`.

### Matched-temperature diagnostic (`analysis/gen06_matched_temperature.py`)
Both arms of each pairing re-evaluated on the same 30 paired test instances at matched sampling. The tau = 1.0 rows reproduce the recorded portfolio numbers exactly.

| sampling | pooled dD_targeted (n = 90) | per-pair |
|---|---|---|
| tau = 1.0 (as trained) | -881 +/- 284 | -1379 / -785 / -479 |
| tau = 0.5 (sharpened) | -1284 +/- 310 | -2035 / -1234 / -585 |
| argmax | -956 +/- 370 | -1987 / -1211 / +328 |

Argmax evaluation raises attacked D in both arms (for example pair2 vanilla D_targeted 5882 to 8014).

### Seed-level sensitivity (`analysis/gen0506_seedlevel_stats.py`)
Pairing as the unit (n = 3): per-pairing dD_targeted {-1379.0, -785.4, -479.2}, mean -881.2, SD 457.5, t(2) 95% CI [-2017.7, +255.3], which includes zero; sign consistency 3/3 negative, one-sided p = 0.125. dD_pathrand {-1353.3, -303.5, -668.7}, t(2) CI [-2099.1, +548.7].
