# gen34: hidden adversary type
Registered 2026-07-23. Results 2026-07-23 (design probe and reference build), 2026-07-24
(training seeds). Code: launch SHA `0c4ac91`.
Artefacts: `models/runs/gen34_family_probe.json`,
`models/runs/gen34_hidden_adversary/family_refs.json`,
`models/runs/gen34_hidden_adversary/seed$S.json` and `seed${S}_ckpts` per seed,
`models/runs/gen35_mmc_check.json`. Scripts `scripts/train_family_generalist.py`,
`analysis/gen34_family_probe.py`, `analysis/gen34_refs.py`, `analysis/gen34_batch.sh`,
`analysis/dyn_exact.py`.

## Question
When the enemy type is drawn per episode from a hidden doctrine family, can a trained policy
convert placement observations into type inference and beat the exact type-blind cap on a
held-out city?

## Game
- Everything not stated here is the gen27 recipe verbatim (`scripts/train_dyn_generalist.py`
  defaults): N=3, K=1, band (0.15,0.95), k_extra=8, fleet-route menu-select, mission objective,
  interception_loss 10.0, episode = 40 sorties, w=3, gamma 0.95.
- Member family, uniform draw per episode, hidden from the defender, on each instance's stacked
  loss matrix L (normative definition in `analysis/gen34_family_probe.py`, `member_fns`):
  M1 `reactive`, softmax-BR tau=0.15 to the trailing-window route counts; M2 `sharp`,
  softmax-BR tau=0.05; M3 `anticipatory`, softmax-BR tau=0.15 to the anti-repeat prediction
  (uniform over routes with zero window count, fallback uniform); M4 `doctrine`,
  window-independent softmax-BR tau=0.15 to the instance's static equilibrium mixture;
  M5 `scattergun`, uniform over interdiction sets.
- Defender observation: the gen27 route-feature head gains two placement-observation columns
  (the realised interdiction set is revealed after each sortie). Col 4 = minmax over routes of
  L[r, j_{t-1}]; col 5 = minmax of an EWMA (decay 0.8, reset per episode) of L[r, j_s] over the
  episode's observed placements. Window feature (col 3) unchanged. The `--no-intel` control
  zeroes cols 4-5 only.
- Pools, gen27 verbatim: train = kaliningrad, east_london, istanbul, 6 ODs each, pool-seed 0;
  held-out = gdansk, 6 ODs, pool-seed 0, never tuned on.
- Trainer: `scripts/train_family_generalist.py`, additive only. 3 seeds x 12000 sorties,
  eval-every 500, plus the `--no-intel` control at the same budget.
- Every exact reference is computed by `analysis/dyn_exact.py` (Karp minimum-mean-cycle,
  cross-checked by damped RVI).

## Criteria
Deployable value per held-out instance = mean per-sortie expected loss over 40-sortie episodes
with the member resampled uniformly per episode, at the select-on-train best checkpoint
(select-on-test dual-reported). Ratio = value / that instance's blind cap.
- PRIMARY: pooled held-out ratio-to-blind-cap < 1.0, and < 1.0 on >= 4/6 held-out ODs, on
  >= 2/3 seeds.
- STRONG: pooled capture fraction >= 0.5 (value <= blind - 0.5 x (blind - omni), pooled).
- CAUSAL CONTROL: the `--no-intel` arm lands at ratio >= 0.95 on the pooled held-out set.
- Reported, not gated: per-member per-instance values, the playbook row, the worst-case
  committing row, train-pool rows, final-iterate drift.

## Baselines
- Blind cap: the blind-optimal window policy, the exact type-blind optimum, which bounds every
  object that does not use the placement observations.
- Omni cap: play each type's exact specialist with the type known.
- Anti-repeat over disjoint routes; best-of-20-orders rotation; iid_eq static mixture;
  uniform and inverse-vulnerability disjoint static stacks; best fixed route.
- Playbook row, fitted to the known member forms: Bayes-MAP over those forms then that member's
  exact specialist policy (MC, 3000 episodes, MAP threshold 0.6).
- Worst-case committing-adversary row: policy marginal against the oracle one-shot best
  response, beside the one-shot v_eq.

## Results

### Design probe (2026-07-23, oracle-exact)
| instance | omni cap | blind cap | inference gap | fitted Bayes-MAP | anti-repeat rule | iid_eq mixture |
|---|---|---|---|---|---|---|
| kaliningrad 35-159 | 0.0527 | 0.0717 | 1.36x | 0.0564 | 0.0757 | 0.1469 |
| kaliningrad 62-97 | 0.0487 | 0.0756 | 1.55x | 0.0546 | 0.0756 | 0.1552 |
| gdansk 249-95 (held-out class) | 0.0589 | 0.1198 | 2.04x | 0.0690 | 0.1767 | 0.2257 |

Specialist cross-tables: every single-type counter-doctrine rises off-diagonal, up to ~28x its
diagonal. The playbook row captures ~80% of the inference gap.

### Reference build (2026-07-23)
Held-out Gdansk blind cap / omni cap / inference gap per OD: 249-95 0.1198/0.0589/2.04x;
106-173 0.1074/0.0547/1.97x; 351-210 0.1282/0.0674/1.90x; 146-296 0.1101/0.0601/1.83x;
275-72 0.1159/0.0638/1.82x; 193-278 0.1023/0.0735/1.39x. Anti-repeat mixture row 0.1478-0.1780
(all >= 1.39x the blind cap). Train-pool gaps 1.17-2.23x.

### Training (2026-07-24, 3 seeds x 12000 sorties)
| seed | select-on-train held-out ratio-to-blind-cap | beats blind cap | select-on-test | final iterate |
|---|---|---|---|---|
| 0 | 1.406 @ 9520 | 0/6 | 1.403 | 1.429 |
| 1 | 1.359 @ 10520 | 0/6 | 1.359 | 1.406 |
| 2 | 1.355 @ 12000 | 0/6 | 1.355 | 1.355 |
| pooled | 1.373 | 0/18 cells | | |

Criterion outcomes: PRIMARY not met (pooled 1.373 against the bar < 1.0; 0/6 held-out ODs
crossed on every seed). STRONG moot. At 1.373 pooled the policy sits below the composed
anti-repeat rule's mixture row (~1.47 mean on held-out; the stationary-versus-episodic horizon
caveat applies). Select-on-test equals select-on-train to 3 decimals; drift small. Per-member
rows are in the seed JSONs.

Causal control: the `--no-intel` arm was not completed. Its interim value was 2.135 at sortie
5520 of 12000, where the sighted seeds were ~1.5; its pre-registered clause (>= 0.95) held
throughout the recorded window. Train-side intel weights rw[3] and rw[4] are strongly negative.
