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
