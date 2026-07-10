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

## RESULT (2026-07-10 09:13, 3 seeds, ~15 min at 3-parallel): **LOCK PASSED. THE MULTI-CONVOY HEADLINE IS NOW POST-FIX.**

| seed | best-ckpt TAP @ sortie | best single-ckpt | final TAP (drift, disclosed) |
|---|---|---|---|
| 0 | 0.308 @ 900 | 0.301 | 0.563 |
| 1 | 0.254 @ 500 | 0.237 | 0.446 |
| 2 | 0.258 @ 400 | 0.261 | 0.331 |

> **gen13-lock exact best-checkpoint TAP mean 0.274 +/- 0.025 (3 seeds).** Both pre-registered
> clauses met: all seeds < ALNS 0.699; mean 0.274 <= ~0.31 (consistent with the gen12 single-seed
> 0.261). Per the pre-authorised consequence (Kilian, "Do steps 1-3"):

**THE LOCKED MULTI-CONVOY HEADLINE LADDER (35-159 k8, N=3, K=1, fleet-route, post-node-ordering-fix):**

| arm | mission-failure exploitability |
|---|---|
| shortest_path (naive stack) | 0.912 |
| ALNS (= loss_det, the deterministic-class optimum) | 0.699 |
| **SACRED (adversarial, exact best-checkpoint)** | **0.274 +/- 0.025** |
| equilibrium (loss_mixed) | 0.206 |

**What is established:** on an instance SCREENED BY PRE-REGISTERED CRITERIA BEFORE ANY TRAINING,
with honest (post-fix) representations, plain config, the adversarially-trained fleet's randomised
stack is 2.55x less mission-exploitable than the certified deterministic optimum and lands at
1.33x the computable equilibrium, tight across 3 seeds, with the drift saved and disclosed and
every checkpoint re-evaluable. **The pre-fix 62-97 number (exact 0.295 +/- 0.024 at `ad70a9c`)
retires to the methods narrative** as part of the representation-bug arc (where it remains
first-class methods material: a fixed bijective permutation improved measured performance while
destroying the model's claimed semantics). **The pre-fix/post-fix two-headline asymmetry
(CRITIQUE_PREFREEZE §2) is retired: BOTH headlines (single-convoy gen10-SC 0.276; multi-convoy
gen13 0.274) now sit on corrected code.** The 62-97 plateau story (gen10/gen11/gen12) stays in the
thesis as the measured account of instance structure vs head discriminability.
