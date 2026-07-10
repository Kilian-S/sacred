# Generation: gen13_lock (the post-fix multi-convoy headline lock on the held-out-screened instance 35-159)

- **status: PRE-REGISTERED 2026-07-10 morning (Kilian: "Do steps 1-3" of the night report =
  explicit go, with the supersession consequence as written there); binding at launch.**
- **git SHA:** the commit landing this ledger.

## Why

gen12's held-out cell ho_N3K1 (35-159 k8, N=3, K=1, plain post-fix fleet-route config) reached
best-checkpoint TAP **0.261 = 1.27x its equilibrium 0.206** on ONE seed, on honest representations,
on an instance screened by the pre-registered criteria (ratio 3.39, leader H/lnR 0.44,
stacked-optimum = equilibrium) BEFORE any training. If that holds across seeds, the multi-convoy
headline moves fully post-fix and the pre-fix/post-fix two-headline asymmetry
(CRITIQUE_PREFREEZE §2) is retired.

## Config (identical to the gen12 ho_N3K1 cell; seeds {0,1,2}; seed 0 = the gen12 run, re-run
fresh here anyway so the lock is one uniform 3-seed batch)

35-159, k_extra 8 menu-select, band 0.15-0.95, N=3, K=1, fleet-route, smooth FP tau 0.05,
switch-every 200, smooth-window 250, leader-ent-frac 0.5, leader-alpha-floor 0.20, 1200 sorties,
eval-every 100, EXACT estimator, per-eval checkpoints, `--threads 3`, 3-parallel. NO gen11 flags.

## Decision metric (PRE-REGISTERED)

Primary = exact best-checkpoint TAP, mean +/- pop std over 3 seeds, under 35-159's oracle BR
interdictor. Anchors (oracle, computed pre-launch): shortest 0.912 > ALNS 0.699 (= loss_det) >>
equilibrium 0.206.

> **LOCK (pre-authorised consequence, night-report step 1 as approved):** all 3 seeds'
> best-checkpoint TAP < ALNS 0.699 AND the mean lands within ~0.05 of the gen12 single-seed
> 0.261 (i.e. <= ~0.31). If met, **this becomes THE multi-convoy headline** (post-fix, honest
> representations, held-out-screened instance); the pre-fix 62-97 exact 0.295 retires to the
> methods narrative as part of the bug-arc story. If the mean is <= ALNS but > 0.31, report as
> measured; the pre-fix number stands and the seed spread is disclosed. Last-iterate drift is
> expected and disclosed as always (best-checkpoint discipline unchanged).

## Command (pinned; via `scratch/gen13_morning.sh` stage 1)

```bash
PYTHONPATH=. .venv/bin/python scripts/train_multiconvoy.py \
  --od 35-159 --N 3 --K 1 --k-extra 8 --menu-select --band 0.15,0.95 \
  --fleet-route --attacker-mode smooth --fp-tau 0.05 --switch-every 200 --smooth-window 250 \
  --leader-ent-frac 0.5 --leader-alpha-floor 0.20 --sorties 1200 --eval-every 100 \
  --seed $S --threads 3 --json-out models/runs/gen13_lock/seed$S.json \
  --ckpt-dir models/runs/gen13_lock/seed${S}_ckpts
```

## RESULT (to be appended)
