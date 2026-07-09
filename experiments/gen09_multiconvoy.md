# Generation: gen09_multiconvoy (Act IV, multi-convoy: adversarially-trained randomised fleet routing beats a coordinating metaheuristic under a loss-averse mission-failure objective)

- **status: RESULTS LOCKED 2026-07-09.** This ledger CONSOLIDATES and locks the two banked
  multi-convoy Phase M results so they cannot be lost across sessions. Both were produced under the
  pre-registration already in force (see "Pre-registration provenance" below); this file is their
  dedicated, citable home. The narrative lives in `SACRED_PROGRESS.md` entry 15; the blow-by-blow
  (two "Phase M" sections) lives in `experiments/gen08_interdiction.md`; the design in
  `REDESIGN_INTERDICTION.md` §10; the plan in `ROADMAP.md` Phase M.
- **code state (pinning):** results produced on the working tree on top of **`f801efb`** ("M3:
  multi-convoy trainer built + smoked") plus the uncommitted Phase M machinery diff
  (`scripts/train_multiconvoy.py`, `src/agents/{sac,networks}.py`,
  `src/envs/multiconvoy_interdiction.py`, `src/baselines/interdiction_oracle.py`,
  `scratch/multiconvoy_instance_screen.py`, +test width bumps). This ledger is committed TOGETHER
  with that diff, so the commit that lands it is the reproducing SHA. Suite **146 green**. All Phase
  M machinery is additive / flag-gated: the single-convoy campaign path is byte-identical (14th
  featurise column sliced off by `_clip_x`; `follow_w` and the two role-alphas exist only in the
  menu-select + adversarial multi-convoy mode).

## Pre-registration provenance (metric + exit criterion fixed BEFORE these runs)

The house rule (pin the metric before looking) was satisfied in gen08, not retroactively here:
- **The objective and the yardstick** were fixed at the oracle level (M0, no training,
  `scratch/multiconvoy_{probe,scan,spectrum,cost}.py`): SOFT (probabilistic) interception + a
  LOSS-AVERSE mission-failure objective (P(>=1 convoy lost)), scored as **exploitability =
  mission-failure of the defender's occupancy distribution under the oracle best-response
  interdictor**. The finding that the OBJECTIVE is load-bearing (risk-neutral dilutes the gap to ~0;
  loss-averse holds it) is itself pre-registered in `REDESIGN_INTERDICTION.md` §10.
- **The deployable estimator** is TAP (the trailing-averaged policy occupancy distribution), the
  same estimator pre-registered and validated in the single-convoy B2-P3 arc; for a zero-sum
  fictitious-play learner the trustworthy read is the STATIONARY-TAIL TIME-AVERAGE, not per-eval
  stage play.
- **The exit criterion** (fallback-vs-learned-coordination) was pre-committed by Kilian: if the
  learned-follower coordination does not exceed the structural fleet-route result, the structural
  result is the banked headline and the learned arc is the secondary. That criterion fired; the
  fallback is the headline (see "Decision" below).

## Question (fixed before looking)

**On the multi-convoy contested-resupply mission, does an adversarially-trained SAC dispatcher whose
randomised joint routing (a mixed strategy over routes) beat (a) a coordinating classical
metaheuristic (ALNS) and (b) a non-adversarially-trained SAC, under a soft-interception loss-averse
(mission-failure) objective, approaching the computable minimax equilibrium?** And, secondarily: can
the followers LEARN to coordinate (emergent stack-and-follow) rather than copy the leader by
construction (Obj-3)?

## Arena / instance (Fork A, oracle-screened BEFORE training)

**62 -> 97, k_extra=8 (SHARED-EDGE menu, 12 routes / 364 occupancies), N=3, K=1, soft band
(0.15, 0.95), ABSOLUTE length->prob normalisation, mission-failure objective.** Route action =
route-index MENU-SELECT (scales to shared-edge route sets; NO walk trie). Screened by
`scratch/multiconvoy_instance_screen.py` for:
- **asymmetry** (non-uniform leader equilibrium, leader entropy H/lnR = 0.63 -> fictitious play has a
  gradient, unlike the flat disjoint 33->71),
- **margin** (ALNS / equilibrium ratio **3.23x**, so a leader landing at B2-P3's ~2.2x-equilibrium
  still clears ALNS), and
- **high stack mass** (0.97, near-pure stack-and-randomise coordination).

**Structural finding (screened 72 disjoint OD pairs x 3 bands): DISJOINT route sets are ALWAYS
near-uniform-leader (H/lnR >= 0.97) -> a flat fictitious-play landscape, so the disjoint 33->71 N=3
leader failure (cycling / alpha runaway) is STRUCTURAL, not instance-specific. A non-uniform,
learnable leader REQUIRES shared edges** (Fork A). 33->71 even shared tops out at ratio 2.27x, below
Fork A's 3.23x.

## Arms

| arm | training | role |
|---|---|---|
| `shortest_path` | none (all convoys on the cheapest route) | naive interdiction-unaware planner |
| `ALNS` | classical metaheuristic (destroy/repair, adaptive weights, SA) minimising worst-case mission-failure | the Obj-5 SOTA opponent; ALNS-verified to reach `loss_det` exactly |
| `equilibrium` | none (oracle LP / minimax) | ground-truth minimax value (`loss_mixed`) |
| `vanilla` | SAC, nominal travel objective, no adversary | non-adversarial control |
| `sacred (fleet-route)` | SAC leader vs the oracle best-response interdictor (smooth FP); fleet stacks on the leader (structural) | **the PRIMARY: the banked multi-convoy headline** |
| `sacred (learned-follower)` | as above + followers LEARN to copy via the six-step fix chain | the SECONDARY Obj-3 result |

## Oracle ground truth (Fork A 62->97 k8, N=3, K=1, mission)

**shortest_path 0.973 > ALNS 0.699 (= loss_det, ALNS-verified) >> equilibrium (loss_mixed) 0.216.**

## PRIMARY RESULT (fleet-route: leader-mix + structural fleet stacking)

> **NUMBER STATUS (2026-07-09): the 0.257 below is EARLIER-INDICATIVE, not yet the citable headline.**
> It was produced by an ad-hoc seed-0 command whose exact config was NOT saved (no JSON/log/run dir
> survives). Across seeds the fleet-route leader varied 0.257 / 0.433 / 0.517 / 0.382 (earlier,
> unsaved) from leader-alpha collapsing to different depths. The **citable** fleet-route number is
> being re-established by the PINNED, saved, 3-seed leader-stabilisation re-run pre-registered in
> "gen09-STAB" below. The qualitative result (SACRED << ALNS << vanilla, all five objectives) is
> unchanged and robust: even the worst earlier seed (0.517) beats ALNS (0.699) and vanilla (~0.945).

Seed 0, menu-select route-index action, 400 sorties, smooth FP (earlier-indicative):

> **fleet-route TAP exploitability 0.257 (1.19x the equilibrium 0.216), stable.** No alpha runaway
> (leader-alpha 0.81 -> 0.37, settling); H_lead near its target; TAP sits deep in the ALNS-equilibrium
> gap.

**Headline ladder (mission-failure exploitability, lower better):**

| arm | mission-failure exploitability |
|---|---|
| shortest_path | 0.973 |
| vanilla (non-adversarial SAC) | ~0.945 |
| ALNS (SOTA coordinating metaheuristic, = loss_det) | 0.699 |
| **SACRED fleet-route (adversarial)** | **0.257** |
| equilibrium (loss_mixed, minimax lower bound) | 0.216 |

**What it establishes (Obj-5, met):** on the multi-convoy contested-resupply mission-failure
objective, an adversarially-trained SAC dispatcher that learns a RANDOMISED route mixed strategy (and
fields the fleet as a stack) is far less mission-exploitable than the ALNS coordinating metaheuristic
(0.257 vs 0.699, **+0.442**) AND than non-adversarial vanilla SAC (0.257 vs ~0.945, **+0.688**),
approaching the computable minimax equilibrium (1.19x). Scored against a computable ground truth, on
a shared-edge Kaliningrad instance. Fork A validated: the leader is stable and near-equilibrium here,
unlike the disjoint near-symmetric 33->71 (which cycled / alpha-ran-away / landed ~ALNS).

**Honest caveat (in the ledger, reported as measured):** the fleet stacking is STRUCTURAL (the
followers copy the leader's route by construction), not learned. Making it LEARNED is the secondary
result below.

## gen09-STAB: leader-stabilisation re-run (PRE-REGISTERED 2026-07-09, binding at launch)

**Why (Kilian 2026-07-09):** the fleet-route leader varied across seeds (0.257 / 0.433 / 0.517 /
0.382, earlier-indicative, config unsaved) because leader-alpha collapses to DIFFERENT depths across
seeds; in the bad seeds the leader over-concentrates and becomes exploitable. This re-run kills that
variance and produces the FIRST saved, reproducible, citable fleet-route number. It is tightening,
not rescuing: even the worst earlier seed (0.517) already beats ALNS (0.699) and vanilla (~0.945).

**The three fixes (mirroring the follower late-decay lesson):**
1. **Leader-alpha FLOOR** (new `--leader-alpha-floor`, `src/agents/sac.py`: after each auto-tune step,
   `log_alpha.clamp_(min=log(floor))` on the PRIMARY temperature = the leader's in fleet-route mode).
   The leader temperature can no longer collapse toward a deterministic, exploitable policy.
2. **Higher `--leader-ent-frac`** (0.6, up a notch toward the non-uniform equilibrium leader entropy
   H/lnR = 0.63), so the target keeps the leader spread near the equilibrium instead of over-committing.
3. **Steadier / slightly longer smooth FP** (`--fp-tau 0.15 --switch-every 200`, from attempt 6;
   `--sorties 1200` vs the earlier 400) for reliable convergence.

**Instance and setup UNCHANGED:** 62->97, k_extra=8 (shared-edge menu), band 0.15-0.95, N=3, K=1,
fleet-route (structural stacking), smooth FP, mission-failure objective, absolute vulnerability norm.

**Config (pinned):** leader-ent-frac 0.6, leader-alpha-floor 0.3, fp-tau 0.15, switch-every 200,
sorties 1200, eval-every 200, menu-select route-index action, attacker-mode smooth, seeds {0, 1, 2},
threads 3 (3-parallel = 9 <= 10 cores). TAP_K=5 (TAP = mean occupancy over the last 5 evals);
`follow_w` is present but inert in fleet-route mode (followers hard-copy the leader).

**Command (pinned; the exact per-seed invocation, run for seeds 0/1/2 via
`scratch/gen09_leader_stab.sh`, all outputs saved):**
```bash
PYTHONPATH=. .venv/bin/python scripts/train_multiconvoy.py \
  --od 62-97 --N 3 --K 1 --k-extra 8 --menu-select --band 0.15,0.95 \
  --fleet-route --attacker-mode smooth --fp-tau 0.15 --switch-every 200 \
  --leader-ent-frac 0.6 --leader-alpha-floor 0.3 --sorties 1200 --eval-every 200 \
  --seed $S --threads 3 \
  --json-out models/runs/gen09_multiconvoy/fleetroute_stab_seed$S.json \
  2>&1 | tee models/runs/gen09_multiconvoy/fleetroute_stab_seed$S.log
```

**GATE (pre-committed):** the three seeds {0, 1, 2} land TIGHT near the good end (**leader TAP
~0.25-0.30 with small std**). Clearing that is the bar to lock the config and go to the (separate)
3-seed headline run. **No lock and no 3-seed headline run yet: this is only the stabilised-leader
check.** Report: the three seeds' leader TAP (mean +/- std), the leader-alpha trajectories, and the
pinned command. If the gate is missed (still loose), the floor / leader-ent-frac are raised and it is
re-run (not patched post hoc).

**Reproducibility policy (fixing the gap that lost the earlier config):** every run writes a
`--json-out` (full per-eval history incl. leader-alpha) AND a tee'd `.log` under
`models/runs/gen09_multiconvoy/`; the orchestrator `scratch/gen09_leader_stab.sh` is committed. No
more ad-hoc unsaved commands.

**SHA:** pinned by the commit that lands this pre-registration + the `--leader-alpha-floor` code
(committed BEFORE the run, so the pre-registration is on record regardless of outcome).

### gen09-STAB RESULT (2026-07-09, SHA `eb44350`): GATE FAILED (wrong-regime convergence; diagnosis below)

Three seeds {0,1,2}, saved to `models/runs/gen09_multiconvoy/fleetroute_stab_seed{0,1,2}.{json,log}`.

| seed | leader-alpha trajectory (200->1200) | H_lead/lnR (tail) | per-eval policy expl (200->1200) | **final leader TAP** |
|---|---|---|---|---|
| 0 | 1.20 -> 0.80 -> 0.40 -> 0.30 -> 0.31 -> 0.30 | ~0.79-0.95 | 0.29 -> 0.63 -> 0.61 -> 0.78 -> 0.94 -> 0.88 | **0.818** |
| 1 | 1.01 -> 0.86 -> 0.34 -> 0.30 -> 0.30 -> 0.30 | ~0.57-0.98 | 0.37 -> 0.31 -> 0.75 -> 0.73 -> 0.81 -> 0.78 | **0.767** |
| 2 | 0.47 -> 0.37 -> 0.34 -> 0.30 -> 0.30 -> 0.30 | ~0.82-0.97 | 0.45 -> 0.34 -> 0.57 -> 0.73 -> 0.91 -> 0.69 | **0.715** |

**Leader TAP mean 0.767 +/- 0.042 (pop std).** GATE (~0.25-0.30 tight): **FAILED** - the std is tight
but the value is wrong (near-uniform, worse than ALNS 0.699, ~3.5x the equilibrium 0.216).

**What the run establishes (both halves matter):**
- **The leader-alpha FLOOR works mechanically:** all three seeds drive alpha down and pin at the floor
  0.30 (vs the earlier collapse to different depths). The across-seed alpha variance is killed. That
  half of the fix is sound and retained.
- **But the leader converged to the WRONG regime (near-UNIFORM, not the 0.63 equilibrium), and the
  per-eval exploitability CLIMBS over training** (the leader degrades). Root cause = **`fp-tau 0.15`
  (borrowed from attempt-6, the FOLLOWER run) is too diffuse FOR THE LEADER:** a diffuse smooth
  attacker does not sharply punish the leader's high-vulnerability routes, so the per-route Q-gradient
  is flat and the leader has no incentive to concentrate onto the non-uniform equilibrium - it sits at
  maximum entropy (uniform, H_lead/lnR ~0.9) and stays exploitable. The earlier-indicative good seed
  (0.257) used the sharper **default tau 0.05**. Secondary error: raising `--leader-ent-frac` was the
  wrong DIRECTION here (the leader is over-SPREAD, not over-concentrated; the floor already prevents
  over-concentration, so more entropy target only compounds the uniformity).

**Corrected re-run (gen09-STAB-2, proposed; awaiting Kilian's go before any CPU):** restore the
LEADER's attacker sharpness `--fp-tau 0.05` (sharp attacker -> per-route Q-gradient -> the leader
concentrates onto the non-uniform 0.63 equilibrium), KEEP the leader-alpha floor (it correctly caps
the resulting over-concentration overshoot), and do NOT raise leader-ent-frac above the equilibrium.
Per the pre-registered contingency this is a config tune + re-run, not a post-hoc patch; the FAILED
result above stands on record.

### gen09-STAB-2 (PRE-REGISTERED 2026-07-09, Kilian's go): sharp attacker drives the hedge

**Anti-answer-fitting discipline (Kilian, explicit):** the SHARP ATTACKER must produce the ~1/vuln
hedge; the floor and ent-frac are PERMISSIVE GUARDRAILS, NOT tuned to the oracle's known 0.63. The
leader landing near 0.63 must be VALIDATED as an outcome of the adversarial dynamics, not forced by
knob-sweeping. If it misses, investigate the mechanism (attacker sharpness? floor still clamping?),
do not crank knobs toward 0.63.

**Three changes from STAB-1 (each justified, none tuned to the answer):**
1. **`--fp-tau 0.05`** (was 0.15): the driver. A sharp smooth attacker concentrates coverage on the
   leader's high-vulnerability routes -> per-route Q-gradient -> the leader learns the non-uniform
   hedge. (0.15 was the FOLLOWER's setting from attempt 6; diffuse => flat gradient => uniform leader.)
2. **`--leader-alpha-floor 0.20`** (was 0.30): the one legitimate correction. In STAB-1 the floor
   CLAMPED (all seeds pinned exactly at 0.30, holding the leader uniform); 0.20 sits below the
   good-seed natural alpha ~0.37, so it is a genuine anti-collapse backstop, not a setpoint.
3. **`--leader-ent-frac 0.5`** (was 0.6): deliberately BELOW the 0.63 equilibrium. The entropy
   regulariser would PERMIT concentration past 0.63 (to 0.5*lnR), so if the leader instead settles at
   ~0.63 that is the ATTACKER holding it there, not the target (0.6 ~= 0.63 was a confound).

Everything else UNCHANGED (62->97 k8, band 0.15-0.95, N=3, K=1, fleet-route, smooth FP, switch-every
200, 1200 sorties, eval-every 200, seeds {0,1,2}, saved JSON + logs). Orchestrator
`scratch/gen09_leader_stab2.sh`. **GATE unchanged: 3 seeds land tight ~0.25-0.30 (H_lead/lnR ~0.63)
as an OUTCOME.** SHA pinned by the commit landing this pre-registration.

#### gen09-STAB-2 RESULT (2026-07-09, SHA `97ba7c2`): GATE FAILED, but the MECHANISM is validated; FP cycling isolated

Three seeds, saved to `models/runs/gen09_multiconvoy/fleetroute_stab2_seed{0,1,2}.{json,log}`.

| seed | TAP: 200->400->600->800->1000->1200 | per-eval expl tail | H_lead/lnR (early->late) | final TAP |
|---|---|---|---|---|
| 0 | 0.39 -> 0.39 -> **0.29** -> **0.27** -> 0.34 -> 0.45 | spikes to 0.96 | 0.55 -> 0.79 | 0.609 |
| 1 | 0.37 -> **0.28** -> 0.37 -> 0.49 -> 0.58 -> 0.71 | spikes to 0.94 | 0.75 -> 0.90 | 0.830 |
| 2 | 0.40 -> **0.28** -> 0.33 -> 0.39 -> 0.47 -> 0.65 | spikes to 0.92 | 0.65 -> 0.88 | 0.806 |

**Leader TAP mean 0.748 +/- 0.099.** GATE (~0.25-0.30 tight): **FAILED** on the endpoint. But the run
is the most informative yet:
- **The adversarial mechanism is VALIDATED (as an outcome, not forced):** the SHARP attacker drives
  all three leaders to the equilibrium hedge EARLY - **TAP 0.27-0.29 at H_lead/lnR 0.55-0.75** around
  sortie 400-600, with alpha settling naturally ABOVE the 0.20 floor (floor not binding early). The
  ~1/vuln hedge is produced by adversarial pressure, exactly as intended.
- **The TAIL diverges = last-iterate FICTITIOUS-PLAY CYCLING** (the single-convoy B2-P failure mode):
  per-eval exploitability spikes to ~0.9 (the leader jumps onto a near-pure route the attacker then
  covers) and the policy re-spreads; TAP climbs back to 0.6-0.8.

**Root cause (mechanism, investigated per the anti-knob-cranking discipline):** the multi-convoy
"smooth" attacker is NOT actually smooth. In `train_multiconvoy.py` it samples ONE interdiction set
and HOLDS it for the entire `switch_every`=200 block, then resamples - block-held single-iset play,
which is the CYCLING regime. The stable single-convoy **B2-P3** did two things this code does not:
(a) it sampled a **FRESH iset EVERY sortie** from the softmax; (b) it used a **TRAILING-WINDOW** of
recent play, not the all-history occupancy (which goes stale and lets the leader drift to exploit the
attacker's lag). So the multi-convoy trainer never had the FP discipline that stabilised single-convoy.
This also retroactively explains the earlier seed-to-seed variance.

### gen09-STAB-3 (PRE-REGISTERED 2026-07-09, Kilian's go): the ported B2-P3 smooth-FP discipline

**Mechanism port, NOT a knob tune (Kilian, explicit):** the exact single-convoy B2-P3 smooth-FP
attacker discipline is factored into a SHARED helper `src/baselines/fp_dynamics.py` and used by BOTH
`scripts/train_interdiction.py` (routes) and `scripts/train_multiconvoy.py` (occupancies) - reusing
the proven-stable code, not a second re-derivation. The discipline: softmax BR (temperature `tau`)
to the defender's TRAILING-WINDOW recent play, recomputed per block, with a committed iset SAMPLED
FRESH EVERY sortie (block-holding one iset was the cycling regime; all-history averaging went stale).
Verified: the single-convoy B2-P3 smooth path is preserved after the refactor (400-sortie smoke:
sacred TAP 0.259 < vanilla 0.389, H_pol stable ~2.1); suite 146 green.

**ONLY change vs STAB-2 = this attacker mechanism.** tau 0.05, leader-alpha-floor 0.20,
leader-ent-frac 0.5, switch-every 200 ALL unchanged (switch-every kept at 200 not B2-P3's 50: with
per-sortie sampling the switch interval only sets the softmax-weight refresh cadence, the iset is
already fresh each sortie). smooth-window 250. Instance 62->97 k8 band 0.15-0.95 N=3 K=1 fleet-route.

**Command (pinned; per-seed, via `scratch/gen09_leader_stab3.sh`, all saved):**
```bash
PYTHONPATH=. .venv/bin/python scripts/train_multiconvoy.py \
  --od 62-97 --N 3 --K 1 --k-extra 8 --menu-select --band 0.15,0.95 \
  --fleet-route --attacker-mode smooth --fp-tau 0.05 --switch-every 200 --smooth-window 250 \
  --leader-ent-frac 0.5 --leader-alpha-floor 0.20 --sorties 1200 --eval-every 200 \
  --seed $S --threads 3 --json-out models/runs/gen09_multiconvoy/fleetroute_stab3_seed$S.json \
  > models/runs/gen09_multiconvoy/fleetroute_stab3_seed$S.log 2>&1
```

**SUCCESS CRITERION (pre-committed, Kilian):** (a) the leader REACHES AND HOLDS ~0.27 across the tail
(not diverging to 0.6-0.8); (b) per-eval spikes damp and the trailing-window TAP - the FP result, since
per-eval cycling is expected - sits STABLE near 0.27; (c) seed-to-seed variance gone (this bug also
explains the earlier 0.257/0.433/0.517/0.382 variance). Report all three seeds' trailing-window TAP
(mean +/- std) and the per-eval tail behaviour. NO lock, NO 3-seed headline lock-in yet. SHA pinned by
the commit landing this pre-registration + the fp_dynamics port. The two FAILED runs stand on record.

#### gen09-STAB-3 RESULT (2026-07-09, SHA `b7b9a98`): FAILED the "hold" criterion; but the 3 runs reveal a reproducible TRANSIENT

Saved to `models/runs/gen09_multiconvoy/fleetroute_stab3_seed{0,1,2}.{json,log}`.

| seed | TAP: 200->400->600->800->1000->1200 | **best-ckpt TAP (min, ~sortie 400)** | final TAP | tail-avg |
|---|---|---|---|---|
| 0 | 0.34 -> **0.33** -> 0.38 -> 0.45 -> 0.54 -> 0.69 | 0.327 | 0.806 | 0.604 |
| 1 | 0.49 -> **0.30** -> 0.37 -> 0.52 -> 0.59 -> 0.77 | 0.295 | 0.942 | 0.691 |
| 2 | 0.36 -> **0.26** -> 0.32 -> 0.45 -> 0.53 -> 0.70 | 0.257 | 0.843 | 0.648 |

**Final TAP mean 0.864 +/- 0.057; tail-average 0.648 +/- 0.036 (~ALNS 0.699).** GATE ("hold ~0.27
across the tail"): **FAILED** - the ported smooth-FP discipline did NOT stop the drift (same signature
as STAB-2: concentrate early, drift to uniform). So the block-held/all-history diagnosis was wrong;
per-sortie sampling + trailing window changed nothing.

**The reproducible finding across STAB-1/2/3 (the real result):** with the SHARP attacker (tau 0.05,
STAB-2 and STAB-3) all three seeds concentrate to the equilibrium hedge EARLY and TIGHT
(**best-checkpoint TAP: STAB-2 0.277 +/- 0.007, STAB-3 0.293 +/- 0.029, ~sortie 400**), all beating
ALNS 0.699 / vanilla 0.945 and approaching the equilibrium 0.216 - THEN over-train / drift toward
uniform (TAP -> 0.7-0.9, H_lead -> uniform, alpha -> floor). So:
- the adversarial mechanism reproducibly PRODUCES the equilibrium hedge (validated, tight across seeds);
- but the equilibrium is a TRANSIENT, not a stable fixed point: **uniform is a competing FP attractor**
  and the leader over-trains into it. Kilian's "hold across the tail" criterion is NOT met.
- **The banked 0.257 was a 400-sortie result = this transient best-checkpoint** (not a converged value).
- The seed-to-seed variance Kilian cited (0.257/0.433/0.517/0.382) is the drift caught at different
  training lengths; the BEST-CHECKPOINT (the project's standing discipline: final-checkpoint is
  misleading under co-evolution, use best-checkpoint) is tight (~0.28) and reproducible.

**Two honest paths (Kilian's call):** (A) one more MECHANISM attempt - `switch_every 50` (the exact
B2-P3 weight-refresh cadence; I kept 200, leaving the attacker stale for 200-sortie stretches so the
leader drifts to exploit it) to try to HOLD the equilibrium; (B) bank the BEST-CHECKPOINT framing
(fleet-route best-ckpt TAP ~0.28 +/- 0.01-0.03, tight, beats ALNS, approaches equilibrium; the late
drift to uniform is a documented over-training instability handled by best-checkpoint selection). All
three STAB runs stand on record.

## THE LOCKED MULTI-CONVOY HEADLINE (best-checkpoint; Kilian's decision 2026-07-09)

**Decision (Kilian):** STOP the STAB stabilisation chase. The leader's late drift to uniform is
INHERENT last-iterate fictitious-play cycling, resolved the standard way the single-convoy programme
already used: **best-checkpoint selection** (the final iterate over-trains toward uniform; select the
lowest-exploitability training point), NOT more knob-tuning toward 0.63. Stay on current code (the
true-smooth fp_dynamics port), no revert. The old unsaved single 0.257 is confirmed to have been a
transient best-checkpoint (both the old config-recovery and STAB-2/3 agree); it is not reproduced.

**gen09-HEADLINE (PRE-REGISTERED 2026-07-09, Kilian's go): the definitive, saved, re-evaluable run.**
Sharp-attacker config (the one that reproducibly reaches the equilibrium hedge), CURRENT code, FULL
saving (JSON per-eval history + per-eval actor checkpoints, so the best-checkpoint is a re-evaluable
ARTEFACT, not just a log number), ~1200 sorties so the plot shows BOTH the best-checkpoint and the
subsequent drift. Config: tau 0.05, leader-alpha-floor 0.20, leader-ent-frac 0.5, switch-every 200,
smooth-window 250, k8 menu-select, band 0.15-0.95, N=3, K=1, fleet-route, eval-every 100, seeds {0,1,2}.

**Command (pinned; per-seed via `scratch/gen09_headline.sh`, all saved):**
```bash
PYTHONPATH=. .venv/bin/python scripts/train_multiconvoy.py \
  --od 62-97 --N 3 --K 1 --k-extra 8 --menu-select --band 0.15,0.95 \
  --fleet-route --attacker-mode smooth --fp-tau 0.05 --switch-every 200 --smooth-window 250 \
  --leader-ent-frac 0.5 --leader-alpha-floor 0.20 --sorties 1200 --eval-every 100 \
  --seed $S --threads 3 --json-out models/runs/gen09_multiconvoy/headline_seed$S.json \
  --ckpt-dir models/runs/gen09_multiconvoy/headline_seed${S}_ckpts \
  > models/runs/gen09_multiconvoy/headline_seed$S.log 2>&1
```

**SELECTION RULE (pre-registered + disclosed, this is what makes it rigorous not a cherry-pick):**
best-checkpoint per seed = the training point (per-eval snapshot) of LOWEST exploitability, measured as
the deployable trailing-averaged-policy TAP under the oracle best-response interdictor. **Best-checkpoint
by exploitability is standard for adversarial / minimax training (the last iterate over-trains toward
uniform); the drift is disclosed plainly, not hidden.** The per-eval actor checkpoints + `pol_hist`
occupancy distributions are saved so any window is re-evaluable.

**The locked ladder (fleet-route best-checkpoint, mission-failure exploitability, lower better):**

| arm | mission-failure exploitability | note |
|---|---|---|
| shortest_path (naive, all on cheapest) | 0.973 | interdiction-unaware |
| vanilla (non-adversarial SAC) | ~0.945 | control |
| ALNS forced to STACK | **0.912** | fairness: ALNS *free* to stack does far worse stacking |
| ALNS (SOTA metaheuristic, spreads = loss_det) | 0.699 | spreads BY CHOICE (0.699 < 0.912) |
| **SACRED (adversarial, best-checkpoint)** | **~0.28 (to lock from the run)** | randomised stack |
| equilibrium (loss_mixed, minimax bound) | 0.216 | computable ground truth |

**Fairness rows (Kilian):** (1) the equilibrium is shown alongside ALNS (the computable optimum);
(2) "ALNS forced to stack" = 0.912 >> ALNS spread 0.699 - ALNS is FREE to stack (every stacked
occupancy is in its search space) but SPREADS because that is its optimal deterministic plan, so
SACRED's win is NOT that we denied ALNS stacking; SACRED beats even the spread plan by RANDOMISING
(a mixed strategy ALNS cannot play). SHA pinned by the commit landing this pre-registration + the
per-eval-checkpoint code.

### gen09-HEADLINE RESULT

_(appended after the three seeds complete: best-checkpoint TAP mean +/- std + the locked ladder)_

## SECONDARY RESULT (Obj-3): the LEARNED-FOLLOWER bootstrap arc

**Question:** can the followers LEARN to copy the mixing leader (emergent coordination) rather than
copy structurally? Instance 62->97 k8 (menu-select; NO walk trie), N=3, K=1, smooth FP.
**Outcome: partially, and it loses to the structural fallback (0.257); the fallback stays the
headline.** Six attempts, each pre-diagnosed, each isolating the next blocker.

**The root blocker (chicken-and-egg):** under independent exploration the convoys share a route only
at the ~2% random-coincidence rate, so the CRITIC never experiences the low-failure STACK reward,
never learns Q(follow) high, and the followers (once their temperature anneals) collapse onto FIXED
wrong routes.

**The fix chain + the diagnostic that carried it (`follow_w` = the learned critic-side correlation
weight):**

| attempt | change | `follow_w` | tail stack | read |
|---|---|---|---|---|
| 1-2 | two-alpha (leader high / follower ~0) + role target-entropy | n/a | 0 | followers collapse to fixed routes |
| 3 | forced-copy warmup vs FROZEN mixing leader (ERB / demonstration bootstrap) + route-correlation feature | flat ~1.0 | ~0.22 modal | actor found NO Q-advantage -> critic does not value following |
| 4 | LEVER 2: learned undiluted `taken` term at the policy head | 1.0 -> 1.21 (climbs a little) | 0.15 | actor CAN represent following, critic still weak |
| 5 | + critic-side lever 2 (`taken` term on the Q head) + prioritised replay of stacks (x4) | **1.0 -> 1.30 climbs** | 0.34 (still rising, cut off) | **the critic now VALUES following (follow_w climbs) = the milestone; per-eval expl cycles** |
| 6 | + steadier attacker (switch_every 200), softer fp-tau 0.15, longer horizon (3200) | 1.0 -> 1.25 then PLATEAUS | 0.18 (plateaued) | tau damped the mid-phase cycle; coordination SATURATED weak |

**THE MILESTONE (attempts 4-6): `follow_w` climbs monotonically** = direct evidence the CRITIC can be
made to value emergent coordination (the four-attempt blocker, fixed by the critic-side lever 2). It
required the Bellman-consistent, undiluted, LEARNED `taken` term on BOTH the actor logits and the
critic Q (a Q input, not a hard bonus), plus prioritised replay so the critic keeps seeing the rare
stack.

**Attempt-6 tail-average (the trustworthy zero-sum-FP metric = exploitability of the mean occupancy
over the converged tail):** **0.482, beats ALNS 0.699 by +0.217 and vanilla 0.945 by +0.463**
(per-eval cycle amplitude 0.116). **BUT `follow_w` plateaued at 1.25 -> coordination saturated at
tail stack ~0.18 -> 0.482 is WORSE than the STRUCTURAL fallback 0.257** (full stacking beats partial
learned following).

## Decision (pre-committed exit criterion fired)

**BANK THE FALLBACK (fleet-route 0.257) as the multi-convoy headline.** The learned-follower
bootstrap is a genuine-but-weaker SECONDARY / Obj-3 result: it PROVES the mechanism (learned emergent
coordination that beats ALNS and vanilla on the time-average, with `follow_w` climbing as the direct
diagnostic) but does not exceed the structural version. **Coordination-dynamics work is CLOSED**
(diminishing returns; fp-tau was the last reserved lever).

## Objectives met (all five)

- **Obj-1** multi-convoy asymmetric zero-sum game with a computable minimax equilibrium (+ the
  zero-sum-FP time-average framing).
- **Obj-2** multi-convoy interdiction env layer + the route-index menu-select action head.
- **Obj-3** SAC + ATLA-as-fictitious-play + ERB / demonstration bootstrapping, load-bearing in the
  learned-follower arc (forced-copy vs a frozen mixing leader; prioritised replay of the rare stack).
- **Obj-4** fleet composition (N is a design lever).
- **Obj-5** vs a SOTA adaptive metaheuristic (ALNS, non-degenerate) AND a non-adversarial SAC control,
  under varied disruption (K / N / connectivity sweeps available), scored against a computable optimum:
  the fleet-route ladder above.

Single-convoy **B2-P3 (0.362)** remains the banked single-convoy headline (see
`experiments/gen08_interdiction.md`); this multi-convoy result is the extension that wins bigger and
supplies the metaheuristic opponent single-convoy could not.

## Reproduction / machinery

Fork-A instance screen (oracle only, no training):
```bash
PYTHONPATH=. python scratch/multiconvoy_instance_screen.py
```
Fleet-route PRIMARY (structural stacking, leader learns the mixed strategy):
```bash
PYTHONPATH=. python scripts/train_multiconvoy.py --od 62-97 --N 3 --K 1 --k-extra 8 \
  --menu-select --fleet-route --attacker-mode smooth --sorties 400 --seed 0 \
  --save-leader models/runs/gen09_multiconvoy/leader_62_97.pt
```
Learned-follower SECONDARY (bootstrap vs the FROZEN mixing leader; lever-2 follow_w on actor+critic;
prioritised stack replay; steadier/softer smooth FP):
```bash
PYTHONPATH=. python scripts/train_multiconvoy.py --od 62-97 --N 3 --K 1 --k-extra 8 \
  --menu-select --attacker-mode smooth --leader-ckpt models/runs/gen09_multiconvoy/leader_62_97.pt \
  --forced-copy-warmup 600 --stack-dup 4 --fp-tau 0.15 --switch-every 200 --sorties 3200 --seed 0
```
Machinery (all additive / flag-gated; suite 146 green): menu-select route-index head (`menu_routes`,
mean-pooled per-route embeddings, `src/agents/networks.py`); two role-alphas
(`role_alpha`/`log_alpha_foll`, per-sample `target_entropy` / `alpha_group`, `src/agents/sac.py`);
lever-2 `follow_w` on actor + critic (undiluted per-route `taken` input on both the policy logits and
the Q head); forced-copy / frozen-leader bootstrap (`--leader-ckpt`, `--forced-copy-warmup`,
`--save-leader`); prioritised replay (`--stack-dup`); `--fp-tau`; route-correlation featurise col 14
(`taken_node_frac`); `absolute_vuln_norm` (cross-instance-comparable arc vulnerability).

## New dogmas earned this generation

1. A JOINT / correlated objective needs the coordination signal EXPLICIT and UNDILUTED at the scoring
   head, AND the CRITIC must value it (the actor cannot follow what the critic will not rank;
   `follow_w` climbing is the diagnostic).
2. DISJOINT route sets give structurally uniform leader equilibria -> asymmetry needs SHARED edges.
3. To learn a RARE joint behaviour, the critic must EXPERIENCE it -> demonstration bootstrapping
   (forced-copy vs a frozen mixing leader) + prioritised replay of the rare transitions.
4. Zero-sum fictitious play CYCLES by construction -> judge on the stationary-tail TIME-AVERAGE, not
   per-eval stage play.

## Honest limitations (reported, not patched)

- The PRIMARY's fleet stacking is STRUCTURAL, not learned (the SECONDARY is the honest attempt to make
  it learned; it saturates below the structural version).
- Single seed (seed 0) at present. 3-seeding the fleet-route headline (and tightening
  `--leader-ent-frac`, which varied 0.26-0.52 across runs) is recorded optional future work (M5, each
  launch needs Kilian's explicit go).
- No separate "STRONG form" distance threshold was pre-registered for the multi-convoy generation
  (unlike single-convoy B2-P3's <= 0.05 bar). Reported as measured: the fleet-route TAP 0.257 sits at
  1.19x the equilibrium 0.216 (absolute distance 0.041) and is the closest trained quantity to the
  minimax value; the learned-follower tail-average 0.482 is well above it.
