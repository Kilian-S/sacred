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

## Result (2026-07-05 ~20:40, primary pass; BR reference rows pending) — **COMPETENCE GATE PASSED; PRIMARY NOT MET — SIGNIFICANTLY REVERSED**

**Competence gate: PASS, all six arms** — W(none) within **+5.5…+7.0%** of greedy (6538–6635 vs
6200), exactly gen03's band, replicated. No ceiling compression (attacked W ≈ 8–13k, unbounded
regime). This matrix is fully interpretable — the gen05 confound is absent.

| arm | W(none) | D(random) | D(pathrand) *(in-dist. for scripted)* | D(targeted) **(held out)** |
|---|---|---|---|---|
| greedy | 6200 | 1718 | 5035 | **4921** |
| vanilla (s0/s1/s2) | 6618/6635/6590 | 1751/1807/2027 | 5174/5749/5706 | 5196/5627/5882 |
| scripted (s0/s1/s2) | 6538/6609/6600 | 1890/1650/2180 | 6528/6052/6374 | **6575/6413/6361** |

**Primary:** pooled `dD_targeted = −881 ± 284` (95% CI, n=90), pairings positive **0/3**
(−1379±519 / −785±510 / −479±400) → **NOT MET, significantly reversed**: the adversarially-
trained arm degrades ~900 MORE under the held-out attack. Secondaries: `dD_pathrand = −775 ±
244` (0/3) — the scripted arm is worse even under **its own training attacker**; `dD_random =
−45 ± 221` (dead even). Clean premium ≈ 0 (scripted ≈ vanilla unattacked).

**Reading.** With competence established, the result is unambiguous and consistent across seeds
and both aimed attacks: **training under constant strong attack made the policy measurably LESS
robust to route-aimed attacks, in- and out-of-distribution, at no clean-performance difference.**
The robustness ranking is `greedy (4921) > vanilla (5196–5882) > adversarially-trained
(6361–6575)` — the more adversarial exposure, the worse; the reactive classical dispatcher is
the most robust policy in the matrix (consistent with Ritzinger et al.'s reactive-dominance).
Leading mechanism (fits the campaign-wide SNR theme): under constant attack the latency reward is
dominated by unavoidable attack damage, so the *learnable* signal (assignment quality under
pressure) is diluted — adversarial exposure degraded learning rather than conferring robustness;
the deficit surfaces exactly where queue compounding amplifies policy quality (aimed attacks) and
not where damage is undirected (random) or absent (clean).

**Campaign conclusion (gen03→gen06):** the SACRED zero-sum co-training premise fails on both
sides for this problem class, with a common root cause — (i) the learned adversary cannot learn
to attack (gen03/04: below-random, entropy pinning, SNR); (ii) the protagonist cannot learn
decision-dense arenas (gen05); (iii) even with a strong scripted adversary and a competent
protagonist, adversarial training *worsens* held-out robustness (gen06). This is the thesis's
definitive experimental finding — pre-registered, competence-gated, paired, seeded.
BR reference rows to be appended (cannot change the primary).

## Launch record (2026-07-05 01:42)

- **git SHA:** `0bc6ec3`
- **configs:** vanilla, dynassign_scripted  **seeds:** [0, 1, 2]
- **common args:** `--episodes 800 --switch-every 50 --batch-size 32 --hidden-dim 64 --device cpu --eval-every 50 --group gen06_dynassign_matrix --threads 3 --update-every 1`
