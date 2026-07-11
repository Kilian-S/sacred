# Generation: gen17_lastiterate (C4: ONE bounded attempt to hold the equilibrium in the last iterate)

- **status: PRE-REGISTERED 2026-07-10 (Kilian: "chain C2 and C4"); auto-launches after gen16;
  binding now. HARD GATE: one attempt, no chase (the standing no-chasing discipline; the gen09
  STAB arc burned three attempts on the pre-fix instance and is not re-opened).**
- **git SHA:** the commit landing this ledger + the `--fp-tau-final` anneal.

## Why

Every SACRED result carries the "equilibrium is a reproducible transient; best-checkpoint-selected,
drift disclosed" caveat (last-iterate fictitious-play cycling). This is the ONE item that can
upgrade BOTH headlines from "transient" to "converged". The old "no more leader experimentation"
rule was scoped to the pre-fix 62-97 stabilisation chase; this is a new question on the post-fix
favourable instance (35-159), pre-authorised in the amended DIRECTION_EXPANSION ordering.

## Mechanism (ONE change; principled, not knob-tuned to the answer)

**Annealed smoothing:** the smooth-FP attacker's softmax temperature anneals linearly
`--fp-tau 0.05 -> --fp-tau-final 0.02` across training. Theory: smoothed-game equilibria converge
to Nash as the smoothing vanishes; a SHARPENING attacker raises the penalty for post-hedge drift
exactly where the observed failure occurs (the leader over-trains off the hedge once the smoothed
attacker's pressure is too diffuse to punish it). This is the "annealed smoothing" future-work item
recorded at B2-P3 and in the gen09 ledger, now spent. Guardrails stay PERMISSIVE (ent-frac 0.5,
floor 0.20, unchanged - the STAB-2 anti-answer-fitting discipline; nothing is set to the oracle's
known equilibrium entropy).

## Config

35-159 k8, N=3, K=1, fleet-route, smooth FP window 250, switch 200, ent-frac 0.5, floor 0.20,
**2400 sorties** (the anneal needs runway), eval-every 100, exact estimator, per-eval ckpts, seeds
{0,1,2}, `--threads 3` 3-parallel. ONLY change vs the gen13/gen14 headline config: the tau anneal.

## Decision metric (PRE-REGISTERED; the hold-the-tail bar, LAST-iterate not best-checkpoint)

Anchors: eq 0.206, ALNS 0.699; the best-checkpoint reference band from gen14 n=10: 0.256 [0.246,
0.266].

> **PASS (converged):** the FINAL-THIRD (sorties 1601-2400) per-eval TAP mean <= 0.27 on all 3
> seeds AND the final TAP <= 0.31 - i.e. the tail HOLDS near the best-checkpoint band instead of
> drifting to 0.4-0.8. Consequence: both headlines' "transient" caveat is upgraded to "converged
> under annealed smoothing" (the best-checkpoint numbers stand; the caveat text changes).
> **FAIL:** report as measured; the transient/best-checkpoint discipline stands as the honest
> resolution; NO further attempts (the gate).

Secondaries: the full TAP trajectory (does drift onset move later as tau tightens?); alpha/H_lead
tails; best-checkpoint TAP (should match the gen14 band regardless).

## RESULT (2026-07-11, 3 seeds, 2400 sorties): **FAIL on the hold-the-tail bar; the gate closes the question**

| seed | best TAP @ sortie | final-third per-eval TAP mean (bar <= 0.27) | final TAP (bar <= 0.31) |
|---|---|---|---|
| 0 | 0.288 @ 800 | 0.422 | 0.660 |
| 1 | 0.300 @ 400 | 0.522 | 0.833 |
| 2 | 0.264 @ 400 | 0.592 | 0.659 |

- **FAIL:** no seed holds the tail (final-third means 0.42-0.59 vs the 0.27 bar; final TAPs
  0.66-0.83). The annealed smoothing DELAYED the drift in places (seed 0 held ~0.30-0.40 through
  mid-training, longer than the constant-tau runs typically manage) but did not prevent it: as tau
  tightened, the sharper attacker eventually re-created the pure-BR cycling pressure and the tail
  left the band. Best-checkpoint values (0.264-0.300) sit in the gen14 band as pre-registered -
  the headline number is unaffected.
- **Consequence (per the hard gate): the question is CLOSED.** Four independent hold-the-tail
  attempts have now failed across two instances and two eras (STAB-1 diffuse-tau, STAB-2 sharp-tau,
  STAB-3 ported-discipline, gen17 annealed-tau post-fix on the favourable instance). The
  equilibrium-as-reproducible-transient finding is therefore NOT an artefact of any single
  configuration: it is inherent to last-iterate fictitious-play dynamics in this game class, and
  **best-checkpoint selection with disclosed drift stands as the honest, final resolution** in
  every headline. The thesis gains a stronger sentence, not a weaker one: the caveat is now backed
  by a systematic, pre-registered attempt ladder, and true last-iterate convergence (optimistic /
  extragradient / magnetic dynamics) is precisely-scoped future work.
