# ZST step 0 (F4): zero-shot transfer of the post-fix single-convoy policy to a held-out OD

- **status: PRE-REGISTERED 2026-07-10 ~02:30 (night programme item 5; auto-launches after the
  gen12 sweeps via `scratch/zst_stage.sh`); binding now, results appended below.**
- **git SHA:** the commit landing this ledger.

## Question (fixed before looking)

Does the mixed-strategy routing behaviour learned adversarially on ONE OD pair transfer ZERO-SHOT
to a held-out OD pair's security game, i.e. is the policy's route mixture on the unseen game less
exploitable than (a) an untrained (random-init) network and (b) the deterministic shortest-path
default, measured against the held-out game's own oracle? (The aim-level ZST promise, first
trained test; meaningful only post-node-ordering-fix, since the pre-fix policy was an
instance-memorised lookup by construction.)

## Setup

- **Source:** gen10-SC config retrained seed 0 with actor saving (`--skip-vanilla --save-actor`;
  33->71 k8 hard interception, walk mode, smooth FP, 3000 sorties): the post-fix single-convoy
  policy. (The gen10-SC runs saved no actors; this retrain is the same pre-registered config.)
- **Held-out target: 110->135 k8 hard = B2-S**, the secondary instance PRE-REGISTERED in the gen08
  ledger (2026-07-06) and never run; anchors were pinned there before any training: equilibrium
  0.333, uniform 0.818, best cost-mixture >= 0.862, shortest 1.000. Choosing the already-registered
  instance avoids any suspicion of target-shopping. Note the target game is HARDER to mix on
  (3 distinct route groups; uniform is nearly as bad as deterministic).
- **Eval:** exact deployable route mixture (trie branch product) under the held-out game's oracle
  BR interdictor: `scratch/zst_transfer.py`. Rows: transferred policy; random-init reference
  (same architecture, untrained); home-OD sanity (should reproduce a gen10-SC-like reading);
  anchors from the held-out oracle.

## Decision reading (PRE-REGISTERED)

- **Minimal transfer claim:** Expl_holdout(transferred) < Expl_holdout(random-init) AND
  < Expl_holdout(shortest-path). Both must hold for ANY transfer claim.
- **Strong (not expected):** within 1.5x of the held-out equilibrium 0.333.
- **Recorded expectation (honest, before looking):** PARTIAL transfer at best. The policy has
  never seen the target geometry, hard interception carries no observable threat map to condition
  on, and the walk policy's branch decisions at unfamiliar nodes may be near its random init. A
  negative here is a REPORTABLE scoping result for the aim's ZST promise (it motivates ZST step 1:
  vulnerability-observable multi-instance training, the gen11 feature mechanism), not a failure of
  the fix. Either way B2-S finally gets a measured row.

## RESULT (2026-07-10 04:00): NO TRANSFER (fails the random-init clause); the pre-registered scoping negative

| row | OD | expl (exact route mixture) | cost |
|---|---|---|---|
| home sanity | 33-71 | 0.362 (final-iterate policy; the retrain's deployable TAP was 0.277) | 11.1 |
| **transfer** | 110-135 | **0.699** | 51.4 |
| random-init reference | 110-135 | 0.584 | 52.5 |
| anchors (110-135) | | shortest 1.000, uniform 0.818, equilibrium 0.333 | |

- **Minimal transfer claim: FAILED.** The transferred policy beats shortest-path (0.699 < 1.000)
  and uniform (0.699 < 0.818) on the held-out game, but is WORSE than an untrained network
  (0.699 > 0.584): the training specialised the walk policy to the home OD's branch geometry, and
  that specialisation is negative transfer elsewhere. (The untrained reference's 0.584 < uniform
  0.818 is itself interesting: near-uniform PER-BRANCH mixing on the trie is better-calibrated on
  this instance than uniform-over-routes.)
- **Read (exactly the pre-registered expectation):** with hard interception there is NO observable
  threat/geometry signal for the policy to condition on, so zero-shot generalisation had no
  mechanism to work through; the result cleanly scopes the aim's ZST promise: ZST requires
  map-conditioned, multi-instance training (ZST step 1 = per-route/edge vulnerability features +
  training across sampled ODs, the gen11-B mechanism at proper learning-rate scale). Reported as
  the honest boundary, not a failure of the node-ordering fix (the fix made this test MEANINGFUL;
  pre-fix it would have been vacuously negative).
- **B2-S closure (bonus):** the never-run gen08 secondary instance now has measured rows: the
  home-trained policy 0.699, random-init 0.584, vs its pre-registered anchors.
- Minor disclosed delta: the home row (0.362) is the final-iterate reading recomputed from the
  saved actor; the retrain's in-run final-policy reading was 0.415 and its TAP 0.277 (FP cycling
  makes per-iterate readings wobble; the transfer conclusion is unaffected: 0.699 vs 0.584).
