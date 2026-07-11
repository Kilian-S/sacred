# Generation: gen18_learnedfollower (C2: the learned-coordination redo, post-fix, on the favourable instance)

- **status: PRE-REGISTERED 2026-07-10 (Kilian: "chain C2 and C4"); auto-launches after gen17;
  binding now.**
- **git SHA:** the commit landing this ledger (+ `--head-term-lr` now also scaling `follow_w`).

## Why

The multi-convoy act's biggest remaining caveat: the fleet stacking is STRUCTURAL (followers copy
the leader by construction), not learned. The original learned-follower arc (six attempts,
gen09-era) saturated at tail stack ~0.18, but it was run (a) PRE-node-ordering-fix, (b) on 62-97
(the instance now known to be unfavourable), and (c) with `follow_w` learning at the base lr - the
same silent-under-training gen11 diagnosed for the other head terms. All three handicaps are now
removable. If learned coordination reaches the structural level, the caveat disappears from the
headline; if it improves but falls short, the decomposition sharpens the Obj-3 secondary.

## Config (the banked follower-bootstrap recipe, updated for the three fixes)

35-159 k8, N=3, K=1, menu-select, band 0.15-0.95, smooth FP (tau 0.15, switch 200 - the attempt-6
follower recipe), **frozen mixing leader = the gen14 best headline actor**
(`models/runs/gen14_evidence/mc_seed1_ckpts/actor_ep500.pt`, best-ckpt TAP 0.238), forced-copy
warmup 600, stack-dup 4, **`--head-term-lr 3e-2`** (follow_w now actually trains), 3200 sorties
(the attempt-6 horizon), eval-every 200, seeds {0,1,2}, `--skip-vanilla`, `--threads 3` 3-parallel.

## Decision metric (PRE-REGISTERED)

Anchors: structural best-checkpoint band (gen14 n=10) 0.256 [0.246, 0.266]; ALNS 0.699; pre-fix
learned-follower reference: tail stack ~0.18, tail-avg 0.482.

> **PASS (the caveat falls):** tail STACK-RATE >= 0.8 AND the learned-coordination tail-average
> exploitability <= 0.266 (the structural band's upper edge) on >= 2/3 seeds - learned coordination
> matches structural stacking, so the headline no longer needs the "structural" caveat.
> **PARTIAL (reported, sharpens the secondary):** tail stack materially above the pre-fix 0.18
> AND tail-average < ALNS 0.699. **FAIL:** at or below the pre-fix levels; the structural caveat
> stands as the honest boundary, now measured post-fix on the favourable instance.

Secondaries: `follow_w` trajectory (the diagnostic; pre-fix it plateaued at 1.25 - with lr 3e-2 it
can actually move); stack/follow rates over training; H_lead/H_foll.

## RESULT (2026-07-11, 3 seeds, 3200 sorties): **FAIL - the structural caveat stands, now post-fix on the favourable instance, and the diagnosis sharpens**

| seed | tail-avg expl | tail stack (pre-fix ref 0.18) | follow rate | final TAP |
|---|---|---|---|---|
| 0 | 0.709 | 0.08 | 0.09 | 0.925 |
| 1 | 0.732 | 0.08 | 0.07 | 0.926 |
| 2 | 0.468 | 0.08 | 0.12 | 0.643 |

- **FAIL on every clause** (PASS needed stack >= 0.8; PARTIAL needed stack materially above 0.18 -
  it is BELOW, at 0.08). Seed 2's tail-average (0.468 < ALNS 0.699) reproduces the pre-fix partial
  exploitability result, but coordination itself did not emerge.
- **The informative part: all three original handicaps were removed and it STILL fails.**
  `follow_w` trained hard (1.0 -> 2.93, vs the pre-fix plateau at 1.25 - the lr fix worked); the
  embeddings are honest; the instance is the favourable one. Telemetry shows the sharper failure
  mode: the followers commit DETERMINISTICALLY to fixed wrong routes (H_foll ~ 0 with follow ~0.1),
  and the follower temperature controller fights the collapse late (alpha_foll 0.27 -> 0.92). A
  +2.93 logit shift on the leader's taken route is not enough to overcome a policy that has already
  collapsed elsewhere before the correlation signal became valuable.
- **Consequence:** the "fleet stacking is structural, not learned" caveat is CONFIRMED as a real
  boundary, not an artefact of the bug/instance/lr - measured cleanly post-fix. The honest thesis
  statement strengthens: emergent leader-follower coordination under independent exploration
  remains an open problem in this game class even when the critic demonstrably values coordination
  (follow_w 2.93); candidate future work is EXPLORATION-side (e.g. correlated exploration noise /
  follower policies initialised from the leader), not more signal-side levers (three signal-side
  fix generations are now on record). The structural fleet-route headline and its disclosed caveat
  stand as-is.
