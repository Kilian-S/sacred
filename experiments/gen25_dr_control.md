# Generation: gen25_dr_control (A4: the domain-randomisation generalist + gen21 to n=3)

- **status: PRE-REGISTERED 2026-07-12 (NEXT_STEPS_MASTER Block A item A4; autonomous launch
  authority); binding at launch.**
- **git SHA:** the commit landing this ledger + the `--dr` flag in `scripts/train_generalist.py`.

## Why (CRITIQUE_12-07-26 §4.1; CRITIQUE_EXAMINER §4.3)

gen21's "adversarial training is causal for zero-shot transfer" rests on ONE vanilla seed, and the
vanilla control changes BOTH the adversary and the objective at once (travel cost, no adversary),
so it cannot separate "best-response pressure" from "any threat exposure". Two additions close
this: (a) gen21 to n=3 (two more vanilla seeds, config verbatim); (b) ONE domain-randomisation
(DR) generalist: the gen16 config with the MISSION objective kept, but the interdictor sampled
UNIFORMLY at random each sortie instead of smooth-FP best-responding. DR is the classic
robustness-training alternative (the review's Schott taxonomy separates adversarial from random
perturbation training; the campaign's gen07 design carried the same `dr` arm).

## Arms (all: gen16 recipe verbatim; cities kaliningrad+east_london+istanbul, holdout gdansk,
pool-seed 0, 12000 sorties, eval-every 500, head-term-lr 3e-2, per-eval checkpoints)

| arm | flag | seeds |
|---|---|---|
| vanilla generalist (gen21 extension) | `--vanilla` | {1, 2} (seed 0 = the banked gen21 run) |
| **DR generalist** | `--dr` | {0} |

## Decision metric (PRE-REGISTERED)

Held-out Gdansk mean best-checkpoint TAP ratio, SELECT-ON-TRAIN (the standing default), per arm.
Anchors: adversarial gen16 1.733 +/- 0.149 (select-on-train); random-init ~1.99; gen21 vanilla
seed 0 = 2.338.

> **Readings (pre-committed):**
> - **gen21 n=3:** report mean +/- std; the causal sentence upgrades from "a measured control,
>   n=1" to a defensible n=3 comparison, whatever the values.
> - **DR vs adversarial:** DR clearly worse than gen16 (mean > 1.88 = outside ~1 pooled sd) =>
>   "BEST-RESPONSE pressure, not mere threat exposure, is causal for transfer" (the sharp RARL
>   statement). DR within [1.58, 1.88] => "threat exposure suffices; best-response pressure is
>   not the load-bearing ingredient at this instance family" - reported plainly (and consistent
>   with the A2/A3 finding that the hedge is threat-robust rather than threat-reading). DR below
>   1.58 => DR beats adversarial training; a major honest finding, reported plainly.

## Commands (pinned; 3 runs 3-parallel via `scratch/gen25_dr.sh`)

```bash
# vanilla seeds 1,2 (gen21 config verbatim)
PYTHONPATH=. .venv/bin/python scripts/train_generalist.py \
  --cities kaliningrad,east_london,istanbul --holdout-city gdansk \
  --n-per-city 6 --n-test 6 --pool-seed 0 --sorties 12000 --eval-every 500 \
  --head-term-lr 3e-2 --vanilla --seed $S --threads 3 \
  --json-out models/runs/gen25_dr/vanilla_seed$S.json \
  --ckpt-dir models/runs/gen25_dr/vanilla_seed${S}_ckpts
# DR seed 0
PYTHONPATH=. .venv/bin/python scripts/train_generalist.py \
  --cities kaliningrad,east_london,istanbul --holdout-city gdansk \
  --n-per-city 6 --n-test 6 --pool-seed 0 --sorties 12000 --eval-every 500 \
  --head-term-lr 3e-2 --dr --seed 0 --threads 3 \
  --json-out models/runs/gen25_dr/dr_seed0.json \
  --ckpt-dir models/runs/gen25_dr/dr_seed0_ckpts
```

## Launch record (2026-07-12): attempt 1 REAPED at ~sortie 4500; relaunched DETACHED

The first launch ran as a harness-managed background task and was reaped with its children at
~sortie 4500/12000 (~2.8 h in): the exact gen05-era failure mode SYSTEM.md documents ("long jobs:
nohup + disown in their own session"). Truncated artefacts preserved as `*_attempt1.{log,_ckpts}`
under `models/runs/gen25_dr/` (never compared with the fresh run; indicative only: at 4500 the
vanilla seeds sat at TEST 2.56-2.59 and DR at 2.38, all drifting worse with training). Relaunched
2026-07-12 from scratch, same pinned commands, via `nohup bash scratch/gen25_dr.sh & disown`
(detached; three processes verified alive). Lesson re-learned and recorded.

## RESULT (appended after the runs; nothing above this line changes after launch)
