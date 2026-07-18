# Generation: gen29_multiod (the closing experiment: three-stream coordination; boundary-map final cell)

- **status: PRE-REGISTERED 2026-07-18 (GEN29_MULTIOD_HANDOFF.md; Critic Aerial's brief, Kilian's
  framing). Screen DONE + bars PINNED below BEFORE any trainer result. PAUSE gate: the batch
  launches only on Kilian's explicit in-conversation go.**
- **branch:** `gen29-multiod` off `e6c29e2` (worktree `../sacred-gen29`); all code additive, new
  files (`src/envs/multiod_interdiction.py`, `scripts/train_multiod_generalist.py`,
  `scratch/gen29_screen.py`, `tests/test_multiod_interdiction.py`). Nothing existing changed.

## Why (the register where the ceiling provably lifts)

Every prior register died the same death: a short rule matched or beat the trained policy once the
baseline family was complete (roads' disjoint-route finding; the aerial branch's tabular-FP and
best-5-stack ties). Root cause: in every single-destination game the optimal defence is a small
object, so a napkin rule sits ON the optimum. The multi-OD game is the one register where no small
hand-built object can express the optimal defence, MEASURED against the most hostile baseline family
in the project (probes `738ddd1`, `e6c29e2`, reproduced this session): median coordination gap
survives coordinated napkin rules (14.4%) and even the in-sample m-pairing cap (11.7%); a third
stream widens it to 25-55% above the cap. Framing (binding, Kilian): the CLOSING cell of the
boundary map ("where does learning pay in contested routing?"), moving the game toward the
multi-destination VRP of the title. No wording may say "SACRED superior" unconditionally: a solver
still computes any single instance (tabular FP ties eq, reported ungated with the
"best-response-oracle methods" wording); the unique claim is the zero-shot generalist + the
coordination-mechanism causal row.

## The game (pinned)

Three supply streams s -> t1,t2,t3 (Kaliningrad, `build_route_set` k_extra 8, band (0.15,0.95));
one convoy/stream/sortie; K=1 interdictor commits one edge from the UNION candidate list (hidden);
soft interception; **mission objective P(>=1 of 3 lost)** (additive objective is provably
correlation-gap-free, so the loss-averse coupling is load-bearing: B3 extended). Defender = ONE
policy routing streams SEQUENTIALLY (stream 1, then 2 observing 1's committed route, then 3
observing both): coordination inside one policy's sequential joint action (trains; avoids the gen18
independent-learner boundary). Adversary in training = per-instance smooth FP (`fp_dynamics`
verbatim, tau 0.05, window 250, over joint plays). Estimator = EXACT joint distribution by
conditional enumeration (~1 + R1 + R1R2 forwards/instance; no Monte Carlo), verified vs
Monte-Carlo (test suite). Head features: [worst-vulnerability, **OVERLAP-WITH-COMMITTED**] at a
dedicated lr; NO cost channel (the railroading lesson); all three streams mix equally (ent-frac
0.5, alpha floor 0.20; no leader/follower split).

## Baseline family (PRE-REGISTERED; scored under the same oracle BR)

det (loss_det) · best INDEPENDENT product (alternating LPs, restarts; upper bound on the
independent class) · per-stream disjoint-stack product (R0a heuristic composed) · deconflict-uniform
(payoff-blind) · **the in-sample m-pairing cap m<=4 (the hardest, oracle-fitted row; the screen's
aiming metric)** · tabular smooth FP (ties eq; ungated) · equilibrium (exact joint LP; per-seed
refs, LP-degeneracy dogma). No comparative sentence survives unless it clears whichever rows the
results clear.

## Screen RESULT (2026-07-18, oracle-only; `scratch/gen29_screen.py` -> `models/runs/gen29_screen.json`;
## prevalence figure `assets/gen29_prevalence.png`)

55 valid non-degenerate cells; **median gap-vs-cap 31%, >20% on 44/55, >35% on 23/55** (the
coordination moat is PREVALENT, not cherry-picked). Probes reproduced exactly (napkin 14.4%, cap
11.7%; headline F=3 147->212,188,195 at 55%). **Headline (pre-registered cell, cap strictly < det
= a genuine mixture cap): 147->212,188,195 eq 0.205, cap 0.317 (55% gap), indep 0.321, det 0.555,
tabular-FP 0.207 (ties eq).** Split (disjoint, from the shortlist): 16 train (headline + 15 pool),
6 gated held-out, 4 validation. Held-out cap/eq ratios 1.36-1.73 (mean 1.461).

## Bars (PINNED before the trainer batch)

> **TIER 1 (headline cell 147->212,188,195, in-pool):** best-checkpoint exact joint TAP **< 0.317**
> (the in-sample cap, NOT the weaker independent 0.321) on >= 2/3 seeds AND pooled. **STRONG:
> <= 0.261** (halfway cap->eq 0.205). Tabular-FP 0.207 reported beside it, ungated.
> **UNTRAINED-CONTEXT ROW (measured, mandatory):** sighted random-init lands train ratio 2.24,
> held-out 1.90, beats-cap 2/6 — clearing Tier 1 requires calibration, not init luck.
>
> **TIER 2 (THE ACT'S PRIMARY; supremacy form): zero-shot.** On the 6 gated held-out cells, the
> VALIDATION-selected checkpoint beats **each cell's in-sample cap** on **>= 4/6 AND pooled, on
> >= 2/3 seeds**. **STRONG: pooled ratio-to-eq <= 1.231** (halfway from the held-out mean cap/eq
> 1.461 to 1.0). Beating an oracle-fitted rule zero-shot, on instances never seen, is the sentence
> no prior register could attempt; if only the independent/napkin rows are cleared, the act
> re-scopes to that honestly.
>
> **CAUSAL CONTROL (mandatory; the gen27/no-window pattern):** a BLINDED arm (`--blind`:
> overlap feature + taken_node_frac zeroed, streams route independently within one net) must land
> ~ the best independent product (held-out mean indep/eq ~1.9-2.0), NOT near the sighted policy.
> The sighted-minus-blinded gain is CAUSALLY the coordination channel — the row that makes this
> science, not a score.
>
> **Reported rows (ungated):** worst-case premium; fleet-cost; per-seed refs; final-iterate drift;
> per-cell results (no averaging-away the hard cell).
> **Fail branches (all writable):** Tier-2 partial = the transfer boundary of coordination,
> measured; total fail = "the gap exists (37-55% oracle-proven) but model-free self-play at thesis
> scale cannot capture it" — the final boundary-map cell, still a result under the framing.

## Compute envelope (from the 300-sortie smoke: plumbing PASSED, mechanism signature present:
## rw[overlap] trains to -2.7 = the policy avoids overlapping earlier streams; alpha anneals)

~0.5 s/sortie at `--threads 2`. Plan: 14,000 sorties/seed, eval every 1000; **3 sighted seeds +
1 blinded control**, 4 processes at threads 2 (8 compute threads <= 10 cores), all thread pools
capped (`OMP_NUM_THREADS=1` etc.) + `nice`: **~2-2.5 h wall** for the full batch + control.
Suite 173 green (6 new gen29 tests: joint-payoff cross-check, eq well-posed, node-ordering
contract, overlap/blind channel, deterministic pool, exact-vs-MC estimator).

## RESULTS (appended after Kilian's go; nothing above changes)

## RESULTS (2026-07-18; 3 sighted seeds + 1 blinded control, STOPPED at sortie 6000/14000,
## Kilian's call: the curve was flat from sortie 3000 so the full budget could not change the
## verdict; validation-selected best checkpoints re-evaluated per-cell offline from the saved
## per-1000-sortie checkpoints. DISCLOSED early stop.)

**BOTH TIERS FAIL. The pre-registered boundary outcome.**

| tier | measured | bar | verdict |
|---|---|---|---|
| Tier 1 (headline 147->212,188,195 < 0.317) | 0.330 / 0.399 / 0.364 (pooled 0.364) | < 0.317 on 2/3 + pooled | **FAIL** (0/3) |
| Tier 2 (held-out beat cap >=4/6 + pooled, 2/3 seeds) | beats 0 / 0 / 2 of 6; pooled ratio-to-eq 2.16 / 2.00 / 1.91 | 4/6 + pooled<1, 2/3 | **FAIL** |
| Causal control (blinded) | held-out pooled 2.07 ~ sighted 2.02 | blinded ~ indep, sighted below | channel carried ~nothing |

**Reading (binding for the writeup):**
1. The trained generalist plateaued near its UNTRAINED level (sighted pooled 2.02 vs untrained
   1.90) and did NOT reach even the best INDEPENDENT product (mean indep/eq 1.44), let alone the
   in-sample cap. It did not clear ANY baseline row.
2. **sighted (2.02) ~ blinded (2.07):** the overlap coordination channel did not convert into a
   gain: the gen18 independent-learner boundary REPLICATES in the closing register, now with an
   architectural-conditioning + undiluted-head-term + causal-control design that was built
   specifically to avoid it. That the channel still did not carry is the measured finding.
3. Best checkpoints are EARLY (ep1000-3000); last-iterate FP drift (the project's standing
   dynamics finding) confirmed, so the full 14000 budget could not change the verdict.
4. **What SURVIVES as the contribution (the oracle half):** the screen proves the multi-OD
   coordination moat is real and PREVALENT (median gap-vs-cap 31% over 55 cells, 37-55% on the
   screened best; no hand-built lottery, even oracle-fitted, closes it) — the FIRST and ONLY
   register in the whole project where a complete hostile baseline family leaves a gap. The
   learning half does not reach it at thesis scale.

**The boundary-map sentence this act earns (Kilian's framing, honoured):** *contested routing
needs no learning below a measurable boundary (a two-line rule is near-optimal, proven across
registers); in the one register where no simple rule, however oracle-assisted, can express the
optimal defence (coordination gap 37-55%, measured), model-free adversarial self-play at thesis
scale also fails to capture it, with a blinded control confirming the coordination channel does
not carry.* The map closes with a measured edge, not a trophy.

**Pre-committed re-aim (one attempt, per the brief) — AVAILABLE, Kilian's call, NOT taken
autonomously:** the flat, below-independent curve points at a CREDIT/CAPACITY problem
(terminal-only reward over the 3-stream chain at ~375 sorties/instance) more than a pure
coordination-difficulty wall. A single re-aim (dense per-stream marginal-interception reward +
full 14000 budget, or per-stream immediate credit) would test whether the gap is reachable with
better credit assignment; if it also ties the blinded arm, the boundary is final. Per the brief's
"one re-aim maximum then close", this is the last permissible attempt.

### SINGLE-INSTANCE TRAINABILITY DIAGNOSTIC (2026-07-18, headline cell alone, 8000 sorties,
### same recipe; `models/runs/gen29_single_diag.log`)

Ratio-to-eq curve (cap ratio 1.547): 1.76/1.99/1.78/1.48/1.66/1.78/1.81/1.69/1.52/1.60/1.55/
1.49/1.44/**1.41**/1.47/1.50. **Best-checkpoint 1.41x eq (exploit ~0.29 < cap 0.317); beats the
cap at most back-half checkpoints but NEVER approaches eq (floors ~1.4x) and OSCILLATES**
(overlap head-weight swings -1.5..+4 = FP cycling on the coordination landscape).

**Diagnosis (binding):** (1) self-play is not fundamentally incapable - it beats the oracle-fitted
cap on ONE instance (1.41 < 1.547); (2) but it cannot converge to eq (floors ~1.4x, unstable);
(3) the generalist was DENSITY-STARVED - one instance needed ~7000 sorties to floor at 1.41, the
generalist gave each of 16 instances ~375, so it landed worse (1.61) than single-instance and
below independent (1.44). The generalist failure is dominated by per-instance training density +
FP instability, NOT a pure coordination wall. Note: single-instance self-play beating the cap is
NOT a differentiator (tabular FP ties eq < cap on any single instance, the standing wording rule);
the unique claim must be ZERO-SHOT. **Lever verdict: self-play tuning -> likely Tier-1, unlikely
clean Tier-2 (transfer loss on a 1.4x floor). DISTILLATION from the joint LP is the better shot
(targets eq directly, avoids the measured instability, the proven ZST mechanism), realistic
outcome Tier-1 + PARTIAL Tier-2 (~2-3/6), not a clean 4/6.** Decision (build distillation vs close
as the boundary cell) = Kilian's, pending.
