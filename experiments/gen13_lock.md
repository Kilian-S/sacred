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
| uniform-INDEPENDENT (each convoy uniform over the menu; oracle row) | 0.546 |
| uniform-STACK (all convoys on ONE uniformly-random route; oracle row) | 0.442 |
| **SACRED (adversarial, exact best-checkpoint)** | **0.274 +/- 0.025** |
| equilibrium (loss_mixed) | 0.206 |

**Naive-randomisation rows (added 2026-07-12, oracle-only; `scratch/uniform_stack_probe.py`,
measured in CRITIQUE_EXAMINER.md §5.1):** under a best-response metric any deterministic plan is
maximally exploited, so "beats ALNS" alone is structurally cheap. The rows above close that
attack: SACRED's calibrated mixture also beats the strongest naive-randomisation heuristic
(uniform-stack 0.442) by 0.17 absolute (~40% relative), i.e. the win is CALIBRATION, not mere
randomness. **Menu-sufficiency (same probe):** the equilibrium value is stable from R=8
(0.2061 at k_extra in {4, 8, 12, 16}; only the k_extra=0 / R=4 menu differs at 0.2411), so the
k8 menu-relative equilibrium is not an artefact of menu truncation; meanwhile uniform-stack
DEGRADES with menu size (0.25 at R=4 -> 0.53 at R=20): the value of calibration GROWS with the
route set.

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

### DISJOINT-BASELINE APPENDIX (2026-07-16, Block R0; oracle/eval-only)

> **Context (binding wording rule; CRITIQUE_16-07-26.md §1; probes
> `scratch/disjoint_baseline_probe.py`, `scratch/r0_screen.py`, artefacts
> `models/runs/r0_screen.json`):** the candidate menus' first routes ARE the max-flow
> decomposition, and "uniform-stack over the edge-disjoint routes" (2 lines) is the strongest
> NAIVE baseline. No comparative sentence in this ledger may claim SACRED beats "every
> uncalibrated strategy class" or that its transfer is something "standard algorithms cannot
> achieve" without the rows below beside it.

Rows for THIS ladder (35-159, N=3, K=1; mission exploitability | fleet cost per sortie):
**uniform-disjoint-stack 0.250 | 99.5** and **inverse-vuln-disjoint-stack 0.241 | 99.5**, vs
SACRED 0.256 [0.246, 0.266] | ~90.4 (gen14 ckpts, `models/runs/r0b_structure_discovery.json`),
equilibrium 0.206 | 91.0, det/ALNS 0.699 | 82.7. **Honest reading:** at K=1 the heuristic
MATCHES SACRED's security (inside its CI); SACRED's surviving K=1 edges are (a) fleet cost ~10%
below the heuristic's, tracking the equilibrium's own (90 vs 99.5, eq 91), and (b) the
structure-discovery result (R0b): the policy allocates 0.60-0.65 convoy-mass to the disjoint
core (equilibrium allocates 0.703; uniform 0.333) WITHOUT being given the max-flow structure.
The comparative headline claim moves to K >= m-1 (gen26).
