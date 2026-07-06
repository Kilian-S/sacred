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

## BR reference rows (appended 2026-07-06; primary unchanged — pipeline complete)

| arm | D(br_vanilla) | D(br_scripted) |
|---|---|---|
| greedy | 577 | 1715 |
| vanilla (s0/s1/s2) | 1086 / 949 / 928 | 1749 / 1908 / 1500 |
| scripted (s0/s1/s2) | 1148 / 1023 / 841 | 1324 / 1864 / 1804 |

Both learned best-response attackers remain weak in this leashed-reach arena (≈ at or below the
random attacker's ~1700–2200, and 3–4× below the scripted attacks) — consistent with gen03/04;
the gen05 finding that learned attackers become strong under ROUTE reach is arena-specific.
br_scripted transfers somewhat better (~1500–1900 vs all victims) than br_vanilla (577–1148) but
neither approaches the scripted heuristics. No change to the primary. **gen06 closed.**

## Post-hoc analyses (2026-07-06, ROADMAP A3; generation stays closed, primary unchanged)

Approved by Kilian 2026-07-06 as evidence-hardening for the thesis's mechanism chapter. All
read-only over the artifacts above; scripts committed under `scratch/`.

### A3.1 — training-telemetry comparison between the arms (`scratch/gen06_telemetry_probe.py`)

Windowed means from the six runs' tfevents (ep 1–100 → 700–800). The arms differ systematically
in ways the mechanism paragraph above did not capture:

| quantity | vanilla (3 seeds) | scripted (3 seeds) |
|---|---|---|
| SAC alpha (end) | **0.13** (all seeds) | **0.62–0.86** (never anneals) |
| policy entropy (end) | 0.37–0.39 | 0.47–0.52 |
| Q_Spread | 2.6–3.8 | **13.0–15.1 (HIGHER)** |
| critic loss | ~195–226 | ~856–1131 (4–5×) |
| training Total_Wait | ~7.0–7.8k | ~13.6–15.6k (~2×) |
| training delivery rate | ~0.65–0.66 | **0.18–0.27** |
| final queue | ~17 | ~35–40 |

Reading: (a) the scripted arms trained in a near-collapse regime (a quarter of requests
delivered, double the backlog) — the training distribution is far from the evaluation regime;
(b) their temperature never annealed, so the extra stochasticity is baked into the actors;
(c) the protagonist's critic under attack discriminates MORE, not less (Q_Spread 4×) — the
"critic can't resolve" wording from the gen03/04 *antagonist* diagnosis does **not** transfer to
the gen06 protagonist; the SNR mechanism must be stated as noisier targets (critic loss 4–5×)
plus mis-scaled entropy pressure, not as a flat "no signal". Clean periodic-eval curves are flat
from ~ep 50 in all six arms (learning plateaus almost immediately at the competence band).
Mechanism candidates recorded in `DIRECTION.md` §4: M1 reward SNR, M2 entropy-target mis-scaling
(the 0.45·ln N target grows with the attack-inflated backlog), M3 collapse-regime state
distribution.

### A3.2 — robustness vs training time (`scratch/gen06_snapshot_robustness.py`)

Question: why did selection pick ep100 for two of three vanilla arms? Every protagonist snapshot
of all six runs evaluated under both aimed attackers on the 8 validation instances (the selection
machinery, swept over training time; W_attacked ≈ D trend since clean W is flat from ~ep 50 per
the Eval curves). Early (ep50–200) vs late (ep650–800) window means:

| run | pathrand | targeted |
|---|---|---|
| vanilla_seed0 | 14028 → 14892 (+6.2%) | 14939 → 14930 (−0.1%) |
| vanilla_seed1 | 13970 → 15472 (+10.8%) | 14709 → 15702 (+6.8%) |
| vanilla_seed2 | 13686 → 15603 (+14.0%) | 14500 → 15924 (+9.8%) |
| scripted_seed0 | 15466 → 15178 (−1.9%) | 15999 → 15345 (−4.1%) |
| scripted_seed1 | 14388 → 14844 (+3.2%) | 15126 → 14604 (−3.4%) |
| scripted_seed2 | 15019 → 14626 (−2.6%) | 15385 → 14955 (−2.8%) |

**Reading: in the vanilla arms, aimed-attack robustness DECLINES with clean training** (5/6
cells worse late, up to +14%; the ep100 selections and seed0's ep750-dip selection are exactly
what a min over this drift picks). The scripted arms start substantially worse and improve only
slowly, converging toward the same ~14.5–16k band. So neither arm's trajectory is "learning
robustness": vanilla's early advantage is the generality of an under-specialised policy, and
clean-task specialisation (entropy 0.56 → 0.38, alpha → 0.13 per A3.1) progressively erodes it —
growing commitment = growing predictability = growing vulnerability to aimed attacks. This is
the exploitability axis (DIRECTION.md §2) surfacing in the closed campaign's own data: training
on this reward buys clean competence at the price of attackability, in BOTH arms' end states.
Caveat: 8 validation instances (per-point SEM ~±400–600); the seed1/seed2 vanilla trends exceed
it, seed0's are within noise. Raw per-snapshot values: `scratch/gen06_snapshot_robustness.json`.

### A3.3 — matched-temperature diagnostic (`scratch/gen06_matched_temperature.py`)

Question: the arms converged at different temperatures (A3.1), and eval is stochastic — is the
robustness gap just sampling temperature, or is it in what the policies learned? Both arms of
each pairing re-evaluated on the same 30 paired test instances at matched determinism
(tau = 1.0 as trained; tau = 0.5 sharpened; argmax — symmetric labelled diagnostic, not a
headline protocol). Sanity: the tau = 1.0 rows reproduce the recorded portfolio numbers
**exactly** (pooled dD_targeted = −881 ± 284, per-pair −1379/−785/−479 — same episode seeds,
same sampling path).

| sampling | pooled dD_targeted (n=90) | per-pair |
|---|---|---|
| tau = 1.0 (as trained) | **−881 ± 284** | −1379 / −785 / −479 |
| tau = 0.5 (sharpened) | **−1284 ± 310** | −2035 / −1234 / −585 |
| argmax | **−956 ± 370** | −1987 / −1211 / +328 (n.s.) |

**Reading: the deficit is NOT temperature.** At matched (and fully deterministic) sampling the
adversarially-trained arm remains significantly less robust; sharpening actually *widens* the
gap (the scripted arms' mode policies are worse than their sampled behaviour — their extra
entropy was partially masking the deficit, not causing it). So the gen06 reversal reflects a
genuinely worse learned policy under aimed attack, strengthening the training-time mechanisms
(M1 noisy critic targets, M3 collapse-regime training distribution) over an evaluation-time
temperature artifact; M2 (entropy-target mis-scaling) remains relevant as a *training-time*
channel (it shapes what was learned), not as an eval-time one. Side observation, consistent with
house dogma: argmax-ing hurts BOTH arms' attacked performance (e.g. pair2 vanilla D_targeted
5882 → 8014), so mixing has real value for every max-entropy policy here — stochastic evaluation
remains the only honest headline protocol.

### A3.4 — seed-level sensitivity of the primary (`scratch/gen0506_seedlevel_stats.py`)

Dual-reporting note. The pre-registered primary pooled paired instances (n=90): dD_targeted
= −881 ± 284, CI excluding zero — that criterion stands as recorded. Treating the seed PAIRING
as the unit instead (n=3, conservative): per-pairing means {−1379.0, −785.4, −479.2} (recomputed
from the raw JSONs, matching the recorded values), seed-level mean −881.2, SD 457.5, **t(2) 95%
CI [−2017.7, +255.3] — includes zero**; sign consistency 3/3 negative (one-sided p = 0.125).
Same picture for dD_pathrand ({−1353.3, −303.5, −668.7}, t(2) CI [−2099.1, +548.7]). The thesis
must therefore say: *reversed under the pre-registered pooled criterion, with full sign
consistency across pairings; the 3-seed random-effects view is directionally consistent but not
individually significant.* (For contrast, gen05's smaller reversal IS seed-level significant —
see that ledger's A3.4 note.)
