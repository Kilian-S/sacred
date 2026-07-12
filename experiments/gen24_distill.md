# Generation: gen24_distill (A1: the LP-distillation control for the ZST act)

- **status: PRE-REGISTERED 2026-07-12 (NEXT_STEPS_MASTER Block A item A1; Kilian's standing
  autonomous launch authority granted 2026-07-12 in-conversation); binding at launch.**
- **git SHA:** the commit landing this ledger + `scripts/train_distill.py`.

## Why (CRITIQUE_EXAMINER §4.2; CRITIQUE_12-07-26 §6 Tier 1 item 1)

gen16 shows the adversarially-trained multi-city generalist transfers zero-shot (held-out Gdansk
1.677 select-on-test / 1.733 select-on-train; random-init ~1.99; vanilla cost-trained control
2.338). The standard ML alternative an examiner will name has never run: SUPERVISED AMORTISATION
of the solver: train the SAME architecture, on the SAME instances, with the SAME map-conditioning
features, to imitate each training instance's oracle equilibrium mixture (labels are milliseconds
at K=1), no adversary, then evaluate zero-shot under the identical metric. Until this control
runs, "adversarial interaction contributes beyond label-fitting" is an assumption.

## Design (decisions recorded before any run)

- **Instances:** the gen16 pools verbatim (`--cities kaliningrad,east_london,istanbul
  --holdout-city gdansk --n-per-city 6 --n-test 6 --pool-seed 0`): 18 train instances, 6 held-out
  Gdansk test instances, identical split across seeds.
- **Target (design decision):** per training instance, the **STACKED-MINIMAX route mixture**: the
  defender LP (`_row_minimiser`) restricted to the stacked occupancies (all N convoys on one
  route). Justification: the generalist's deployable policy class is fleet-route (stacked), so
  the fair distillation target is the optimum OF THAT CLASS, not the full-occupancy equilibrium
  the class cannot express. The per-instance gap stacked-optimum vs full equilibrium is computed
  and REPORTED (if a train instance's stacked optimum is far above its equilibrium, the
  distillation ceiling is disclosed, not hidden). Evaluation stays ratio-to-FULL-equilibrium,
  exactly as gen16 scores its own stacked policy.
- **Architecture/features:** the gen16 actor verbatim (node 14 / edge 5 dims, menu-select head,
  `follow_w` + transferable `route_feat_w` head terms, head-term lr 3e-2, NO `route_bias`).
- **Loss:** full-batch KL(target || policy) averaged over the 18 training instances per step
  (the policy's leader route distribution vs the target mixture), Adam at the gen16 actor lr
  (3e-4; head terms 3e-2). No adversary, no replay buffer, no reward.
- **Budget:** 1,500 full-batch steps, eval every 100 (exact per-instance distributions; TAP over
  the last 3 evals, the gen15/16 estimator), per-eval actor checkpoints. Seeds {0, 1, 2}.
- **Selection:** best checkpoint by the TRAIN-set mean TAP ratio (select-on-train, the standing
  default since gen16); the held-out number is reported AT that checkpoint; select-on-test
  dual-reported as the optimistic bound.

## Decision metric (PRE-REGISTERED)

Primary = held-out Gdansk mean best-checkpoint TAP ratio (select-on-train), mean +/- population
std over 3 seeds. Anchors: gen16 adversarial 1.733 +/- 0.149 (select-on-train) / 1.677 +/- 0.072
(select-on-test); random-init ~1.99; vanilla generalist 2.338; each OD's loss_det ratio.

> **Pre-committed reading (band = +/- 0.15 around the gen16 select-on-train mean, ~1 pooled sd):**
> - **distill mean > 1.88 (or above random-init ~1.99):** adversarial interaction contributes
>   beyond label-fitting; the ZST claim STRENGTHENS ("not reproducible by supervised amortisation
>   at matched architecture/data").
> - **1.58 <= distill mean <= 1.88:** a TIE; the honest claim becomes "adversarial self-play is a
>   label-free equilibrium amortiser: it matches supervised distillation WITHOUT needing labels,
>   and labels stop existing past the enumeration wall (K >= 4, A4's regime) where only self-play
>   can train". Reported plainly; not a loss.
> - **distill mean < 1.58:** distillation WINS; the ZST act is re-scoped (supervised amortisation
>   suffices at K=1 sizes; adversarial training's case rests on the no-label regime, the dynamics
>   account and the boundary results). Reported plainly; this outcome is exactly why the control
>   must run before the thesis is written.
> Secondary rows regardless of branch: per-OD ratios; beats-loss_det cell count /18; the
> per-train-instance stacked-optimum/equilibrium ratios (the distillation ceiling); the training
> KL curve (does it fit the targets at all).

## Command (pinned; 3 seeds, 3-parallel via `scratch/gen24_distill.sh`)

```bash
PYTHONPATH=. .venv/bin/python scripts/train_distill.py \
  --cities kaliningrad,east_london,istanbul --holdout-city gdansk \
  --n-per-city 6 --n-test 6 --pool-seed 0 --steps 1500 --eval-every 100 \
  --head-term-lr 3e-2 --seed $S --threads 3 \
  --json-out models/runs/gen24_distill/seed$S.json \
  --ckpt-dir models/runs/gen24_distill/seed${S}_ckpts
```

## RESULT (2026-07-12, 3 seeds, ~9 min training total): the PRE-REGISTERED branch fires for adversarial training, but the symmetric validation row REVERSES it; the honest claim is sharper than either

**Per-seed (distillation):**

| seed | select-on-TRAIN (primary): held-out @ step (train ratio) | select-on-test (optimistic) | final |
|---|---|---|---|
| 0 | **2.020** @ 1500 (train 1.125) | 1.485 | 2.020 |
| 1 | **2.027** @ 1400 (train 1.143) | 1.472 | 2.040 |
| 2 | **2.157** @ 1200 (train 1.154) | 1.504 | 2.018 |

> **PRIMARY (select-on-train): distill mean 2.068 +/- 0.063 > 1.88** => the pre-registered
> "adversarial interaction contributes beyond label-fitting" branch fires (gen16 adversarial
> select-on-train = 1.733). Mechanism, visible in every seed's curve: distillation fits its 18
> training targets nearly perfectly (train ratio -> 1.12-1.15) while the held-out ratio DEGRADES
> monotonically past ~step 100-300 (textbook supervised overfitting on 18 instances); with no
> external signal, train-side selection picks the overfit endpoint. The adversarial generalist
> never shows this shape (its train and held-out ratios track; select-on-train 1.733 ~ its
> val-stopped 1.761 below): **adversarial self-play is self-stopping; supervised distillation is
> not.**

**The symmetric fairness row (added before looking at its outcome; `scratch/gen24_valstop.py`):
early stopping on a PROPER validation set (9 fresh train-city instances, pool-seed 1), applied to
BOTH arms, held-out Gdansk TAP at the val-selected checkpoint:**

| arm | per-seed | held-out mean |
|---|---|---|
| distillation + val early stop | 1.527 / 1.526 / 1.613 (all select step 100) | **1.555** |
| adversarial (gen16) + val early stop | 1.599 / 1.941 / 1.745 (all select step 500) | **1.761** |

> Under the standard supervised recipe, **distillation transfers BETTER than the adversarial
> generalist (1.555 vs 1.761)** on this pool. (Granularity caveat: checkpoints exist every 100
> steps; the distill val-optimum sits at the first saved checkpoint, so the true optimum may be
> earlier/better. Artefact: `models/runs/gen24_distill/valstop.json`.)

**THE HONEST SYNTHESIS (binding for the storyline; supersedes any "adversarial training is
necessary for transfer" wording):**
1. **Where equilibrium labels exist (K=1, milliseconds per LP), supervised distillation with
   validation early stopping is the strongest transfer recipe measured (1.56x)**: the ZST act
   must not claim adversarial training is necessary for zero-shot transfer at label-available
   instance sizes.
2. **Adversarial self-play reaches comparable transfer WITHOUT labels and WITHOUT any validation
   signal** (1.73 select-on-train, 1.76 val-stopped: the two agree because it does not overfit).
   Its unique value is exactly the regime the thesis's scaling story names: past the enumeration
   wall (K >= 4, A4's regime) labels do not exist and validation-by-oracle is unavailable, so
   self-play is the only trainer on the board; at label-available sizes its advantage is
   label-freeness + self-stopping, not final transfer quality.
3. The regularisation contrast itself (distillation overfits 18 instances catastrophically;
   adversarial FP pressure never lets the policy commit to any instance) is a citable finding
   connecting to the fictitious-self-play literature (the survey's Heinrich & Silver thread).
4. Ceiling disclosure: the stacked-minimax target EQUALS the full equilibrium on all 18 training
   instances (ratio 1.000 everywhere), so the distillation target was not handicapped; fleet-route
   restriction is also fair on the held-out pool (gen16's own policy class).

**Consequence for the thesis's flagship act:** the ZST claim re-scopes to the two-regime form:
*"zero-shot transfer of calibrated hedging is achievable by amortisation in general; below the
enumeration wall, supervised distillation (labels available) is the better recipe; past it, only
adversarial self-play can train, and it carries its transfer quality there label-free and
self-stopped."* Combined with zst_map_robustness (the hedge is geometry-informed and
threat-robust), the act's honest centre of gravity moves from "the map-conditioned adversarial
generalist" to "label-free, self-stopping, threat-robust amortisation of security-game hedging".
