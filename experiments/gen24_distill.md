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

## RESULT (appended after the run; nothing above this line changes after launch)
