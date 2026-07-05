# Generation: gen06_dynassign_matrix (Phase 3 retake — the robustness matrix in the competent arena)

- **git SHA:** `cfabc90`
- **date opened:** 2026-07-05
- **status:** LAUNCHING (option (c) chosen by Kilian 2026-07-05 after the gen05 competence-void)

## Question (fixed before looking)

**Does adversarial training against a strong scripted attacker buy robustness to a HELD-OUT
attack, in an arena where policies demonstrably learn to competence?** gen05's hybrid matrix was
uninterpretable because neither arm learned the task (ceiling compression). dynassign is the
arena where this stack reaches within ~7% of greedy clean (gen03, replicated ×5). Both arms are
trained FRESH on this SHA (gen03's vanilla arms predate the motion-feature bump — never compare
across git states / information sets).

## Design

| | value | why |
|---|---|---|
| arms | `vanilla`, `dynassign_scripted` (trains vs **pathrand**) × seeds {0,1,2} | identical env/reward/nets/hparams; only training-time adversary differs |
| training attacker | `pathrand` = first blockable edge on a uniformly RANDOM goal-committed truck's path | route-aimed but stochastic across trucks (less determinism to overfit); keeps `targeted` fully held out |
| config | dynassign λ=0.06, budget 4000, 800 ep, switch-every 50, UTD 1 | the gen02/gen03 lineage config; ~29 s/ep at 3-parallel → ~13 h for 6 runs |

## Attack portfolio (eval)

| attacker | role |
|---|---|
| `none` | clean baseline |
| `random` | undirected floor |
| `pathrand` | **in-distribution** for the scripted arm; also the VALIDATION (selection) attacker for BOTH arms (asymmetry noted: it is train-attack for one arm; selection is not reported) |
| `targeted` | **HELD OUT — PRIMARY test attack** (never in training or selection; D ≈ +5.9k on a competent frozen defender, gen04) |
| `br_vanilla_s0`, `br_scripted_s0` | learned reference rows (seed 0; leashed-mask arena → expect ≈ random per gen03/04) |

## Decision metric (PRE-REGISTERED)

W = mean total_wait over 30 paired test instances (demand seeds 10_000_019…+29; validation
20_000_019…+7); D(arm, a) = W(a) − W(none) paired per instance; protagonists stochastic.

> **Primary:** pooled `dD_targeted = D(vanilla, targeted) − D(scripted, targeted)` across the 3
> seed pairings. **Success = pooled dD_targeted > 0 with the paired 95% CI excluding 0, and
> ≥ 2/3 pairings individually positive.**

**Competence precondition (pre-registered sanity gate on interpretation):** each arm's W(none)
must land in the competent band (≈ within ~15% of greedy's clean W, per gen03's ~+7%); if an arm
fails this, the matrix is reported but flagged as competence-compromised (the gen05 lesson).

Pre-registered interpretive branches: `dD_pathrand > 0` with `dD_targeted ≈ 0` → attack-specific
hardening without transfer (reportable finding, not headline success); both ≈ 0 with competence
met → adversarial training confers nothing here (honest null, competence-valid this time).

## Commands

```bash
# 1. the matrix (6 runs, 3 concurrent, ~13 h)
PYTHONPATH=. python scripts/run_generation.py --group gen06_dynassign_matrix --configs vanilla,dynassign_scripted --seeds 0,1,2 --episodes 800 --switch-every 50 --eval-every 50 --threads 3 --max-concurrent 3
# 2. selection per run (pathrand validation attacker, 8 val instances)
# 3. two BR attackers (seed-0 selected checkpoints, 300 ep)
# 4. portfolio: arms paired per invocation + greedy; attackers none,random,pathrand,targeted,br_*; 30 instances
```

## Result

_(to be filled)_

## Launch record (2026-07-05 01:42)

- **git SHA:** `0bc6ec3`
- **configs:** vanilla, dynassign_scripted  **seeds:** [0, 1, 2]
- **common args:** `--episodes 800 --switch-every 50 --batch-size 32 --hidden-dim 64 --device cpu --eval-every 50 --group gen06_dynassign_matrix --threads 3 --update-every 1`
