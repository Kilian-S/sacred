# Generation: gen05_hybrid_matrix (Phase 3 — the headline robustness matrix)

- **git SHA:** `cd11f14` (Phase-3 build commit)
- **date opened:** 2026-07-04
- **status:** LEDGER OPEN — awaiting Kilian's launch

## Question (fixed before looking)

**Does adversarial training against a strong adversary buy robustness to held-out attacks?**
Arms: `hybrid_vanilla` (no adversary during training) vs `hybrid_scripted` (trained against the
scripted `targeted` attacker — promoted per the gen04 gate FAIL; the learned/co-evolved adversary
is parked in the back pocket along with the gen04b entropy re-gate). Arena: the FIXED hybrid rung
(assignment + next-hop routing, chokepoint geometry, route-reach, full-block, budget 1500,
max_ticks 800, `--update-every 8` — identical for both arms).

## Design

| | value | why |
|---|---|---|
| arms | `hybrid_vanilla`, `hybrid_scripted` × seeds {0,1,2} | 2×3 runs; identical env/reward/nets/hparams, only training-time adversary differs |
| episodes | 400, switch-every 25 (= snapshot cadence, 16 snapshots) | hybrid episodes carry ~5–10× more decisions than dynassign; 400 ep ≈ gen03's gradient budget at `--update-every 8` |
| budget / horizon | 1500 / 800 ticks | budget sweep (+84% on greedy, episodes end ~tick 416 attacked); 800 halves untrained-wandering cost with full headroom |
| smoke timings | vanilla ~28 s/ep, scripted ~18 s/ep (untrained; shortens as policies learn) | 6 runs ≈ 2–3 h each, 3 parallel → **~5–6 h wall** |

## Attack portfolio (eval)

| attacker | role |
|---|---|
| `none` | clean baseline for D |
| `random` | undirected floor |
| `targeted` | **in-distribution** for the scripted arm (its training attacker) — reported, NOT the primary; also the VALIDATION attacker for checkpoint selection (same selector for both arms; noted asymmetry: it is train-attack for one arm) |
| `gateway` (first-maskable-edge, route-reach) | **HELD OUT — the PRIMARY test attack.** Never used in training or selection; did +40…+184% on greedy in the budget sweep |
| `br_vanilla_s0`, `br_scripted_s0` | learned best-response rows (one per arm, seed 0 only — gen04 showed BRs ≈ random, kept as the learned-attack reference, not the primary) |

## Decision metric (PRE-REGISTERED)

W = mean total_wait over 24 paired rollout instances (static demand → instance = episode seed);
D(arm, a) = W(a) − W(none) paired per instance.

> **Primary:** `dD_gateway = D(vanilla, gateway) − D(scripted, gateway)` per seed pairing
> (v_k vs s_k), pooled across the 3 pairings. **Success = pooled dD_gateway > 0 with the paired
> 95% CI excluding 0, and ≥ 2/3 pairings individually positive.**

Secondary (reported, not gating): dD under `random` and the br rows; the `targeted` row
(in-distribution — expect the largest gap; explicitly not claimable as held-out robustness);
clean premium W(scripted, none) − W(vanilla, none) (want ≲ +15%); greedy reference rows;
Eval/* training curves. Checkpoint selection per arm: `evaluate_portfolio.py --select-best
--problem hybrid` (targeted attacker, validation rollout seeds 20_000_019+, 8 instances).

## Commands

```bash
# 1. the matrix (6 runs, 3 parallel, ~5-6 h)
PYTHONPATH=. python scripts/run_generation.py --group gen05_hybrid_matrix --configs hybrid_vanilla,hybrid_scripted --seeds 0,1,2 --episodes 400 --switch-every 25 --update-every 8 --eval-every 50 --threads 3 --max-concurrent 3

# 2. selection per run (validation attacker, 8 val instances)
PYTHONPATH=. python scripts/evaluate_portfolio.py --problem hybrid --select-best models/runs/gen05_hybrid_matrix/<run> --instances 8

# 3. two BR attackers (seed-0 selected checkpoints, ~300 ep each)
PYTHONPATH=. python scripts/train_sacred.py --problem hybrid --train-antagonist-only --protagonist-snapshot <sel> --episodes 300 --switch-every 50 --eval-every 0 --seed 0 --group gen05_hybrid_matrix --tag br_<arm>_s0

# 4. portfolio (7 arms x 6 attacks x 24 instances ~ 1000 episodes, ~15-20 min)
PYTHONPATH=. python scripts/evaluate_portfolio.py --problem hybrid \
  --policy vanilla_s0=<sel> --policy vanilla_s1=<sel> --policy vanilla_s2=<sel> \
  --policy scripted_s0=<sel> --policy scripted_s1=<sel> --policy scripted_s2=<sel> \
  --br vanilla_s0=<br actor> --br scripted_s0=<br actor> \
  --attackers none,random,targeted,gateway,br_vanilla_s0,br_scripted_s0 \
  --instances 24 --out experiments/gen05_portfolio.json
```

## Result (2026-07-04, primary pass; BR reference rows pending) — **PRIMARY NOT MET, sign REVERSED; both arms failed to learn the task, voiding the robustness interpretation**

Per-pairing D (paired over 24 rollout instances; greedy is deterministic → single trajectory):

| arm | W(none) | D(random) | D(targeted) | D(gateway) |
|---|---|---|---|---|
| greedy | **847** | 1036 | 1154 | 714 |
| vanilla (s0/s1/s2) | 4739 / 4769 / 4845 | 593 / 585 / 407 | 978 / 841 / 725 | 476 / 366 / 320 |
| scripted (s0/s1/s2) | 4716 / 4605 / 4726 | 559 / 604 / 566 | 849 / 910 / 844 | 582 / 584 / 571 |

**Primary:** pooled `dD_gateway = −192 ± 181` (95% CI, n=72), pairings positive **0/3**
(−106±313, −219±258, −251±368) → **NOT MET**, and the pooled CI excludes zero on the *negative*
side: the scripted-adversarially-trained arm degrades slightly MORE under the held-out attack.

**The finding that dominates the table:** the learned arms' clean performance is
**W(none) ≈ 4.6–4.8k vs greedy's 847 — ~5.6× worse** — and the in-distribution secondary
`dD_targeted = −20 ± 135`: 400 episodes of training *against* the targeted attacker taught **no
measurable coping even with the attack it saw every episode**. Neither arm learned the hybrid
task to competence (delivery ~0.5 clean within the horizon; Q_Spread ~0.1; deterministic-argmax
eval delivers zero). With W(none) that close to the 6.4k saturation ceiling (8 requests × 800
ticks), degradation D is **ceiling-compressed** — weak policies have little left to lose (note
both learned arms show *smaller* D than greedy, which starts at 847 and has everything to lose).
**The robustness comparison between two incompetent policies is therefore not interpretable as a
robustness result**, in either direction; the pre-hoc "muted learning" risk materialized fully.

**Diagnosis shift:** gen03/gen04 located the framework's binding constraint in the *adversary*;
gen05 relocates it for this arena to the **protagonist**: hybrid-rung learning (hundreds of
edge-level decisions/episode, thin credit, γ=0.99/tick, UTD 8, 400 ep) did not produce a
competent policy on either arm. The robustness question requires competent baselines first.

**Options recorded for Kilian:** (a) extend training (lossless resume, 400→1000+, ~a day;
uncertain — Q_Spread ≈ 0.1 suggests structural credit-assignment weakness, not just
under-training); (b) make the hybrid rung learnable first (tighter corridor slack → fewer
decisions + denser credit, γ↑, headroom gate: W(none) ≤ ~1.5× greedy before any re-matrix);
(c) **move the headline matrix to the dynassign arena where policies demonstrably learn to
within ~7% of greedy** — train a scripted-adversarial arm there (vanilla arms already exist from
gen03), with a second scripted attacker variant for training so `targeted` stays held out;
(d) freeze and write the arc as-is. BR reference rows will be appended when their training
completes (they cannot change the primary).

## Launch record (2026-07-04 09:13)

- **git SHA:** `324a644`
- **configs:** hybrid_vanilla, hybrid_scripted  **seeds:** [0, 1, 2]
- **common args:** `--episodes 400 --switch-every 25 --batch-size 32 --hidden-dim 64 --device cpu --eval-every 50 --group gen05_hybrid_matrix --threads 3 --update-every 8`

## Recovery note (2026-07-04 10:28)

The launch-session wrapper was reaped at ~10:25 (harness task cleanup; no reboot — both the
launcher and an inert watcher died simultaneously), taking the three vanilla trainings with it at
~ep 78. Resumed losslessly from the ep-75 phase checkpoints (`--resume-checkpoint`, replay buffer
included; verified the new event stream continues at ep 76). RNG streams are not part of the
checkpoint, so the resumed trajectories differ from an uninterrupted run — same config, same
seeds, noted for the record. The relaunch runs under a detached nohup orchestrator
(scratchpad/gen05_orchestrator.sh → gen05_status.txt) immune to session cleanup; the scripted
batch auto-starts when the vanilla batch completes.

## BR reference rows (appended 2026-07-04 ~22:15; primary unchanged)

D under the learned best-response attackers (24 paired instances; greedy deterministic):

| arm | D(br_vanilla) | D(br_scripted) |
|---|---|---|
| greedy | **1667 ± 0** | **1667 ± 0** |
| vanilla (s0/s1/s2) | 596 / 561 / 532 | 597 / 570 / 436 |
| scripted (s0/s1/s2) | 721 / 754 / 685 | 582 / 764 / 602 |

Two observations. (1) Against the weak learned arms the BR rows sit in the same
ceiling-compressed 430–760 band as every other attack — no new signal about the arms, primary
unchanged. (2) **Against GREEDY, the learned attackers finally work**: +1667, i.e. more damage
than the scripted `targeted` (+1154) and `gateway` (+714) attacks — and both BR nets collapse to
the identical deterministic attack trajectory. In the hybrid arena the route-reach mask does the
aiming and the motion features (gen04's N1 fix) are present; against a *competent, predictable*
victim the learned adversary is now the strongest attacker in the portfolio. This partially
rehabilitates learned attackers **for this arena** (gen03/04's "weaker than random" was the
leashed dynassign setting) and is relevant to any future competent-protagonist matrix and to the
back-pocket ATLA option.

## Post-hoc analysis (2026-07-06, ROADMAP A3.4; primary unchanged): seed-level sensitivity

Dual-reporting note (generation remains closed; this adds an inference-level sensitivity to the
recorded primary). The pre-registered primary pooled paired instances across the 3 seed pairings
(n=72). Treating the PAIRING as the unit instead (n=3; conservative, seeds as random effects):
per-pairing dD_gateway = {−106.2, −219.0, −251.2} (recomputed from the raw portfolio JSONs,
matching the recorded values), seed-level mean −192.1, SD 76.2, **t(2) 95% CI [−381.3, −2.9]
(excludes zero)**, sign consistency 3/3 negative (one-sided sign p = 0.125). Here the
conservative level agrees in sign and significance with the pooled level (the gen05 reversal is
small but consistent). Interpretation unchanged: the matrix is competence-void, so no robustness
reading either way. Script: `scratch/gen0506_seedlevel_stats.py`.
