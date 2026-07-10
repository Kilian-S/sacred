# D3: the composite exhibit (surrogate over the TRAINED generalist, priced by ZST)

- **status: PRE-REGISTERED 2026-07-10 (expansion programme; the culminating computational exhibit);
  chains after A1 (gen15) + reuses the D1 loop; EVAL-ONLY. Binding now.**

## Question (fixed before looking)

Can the three pillars compose into ONE claim: fit a surrogate to (upstream design -> the TRAINED
generalist policy's OPERATIONAL exploitability), run SBO acquisition on that target, and select the
design that minimises the deployed policy's actual vulnerability - a loop in which NO LP can
participate (it re-solves per design AND scores only the equilibrium abstraction, not a policy)?

## Why this is the composite

- **ZST (A1)** makes each design's operational evaluation ONE forward pass + one BR (no retraining
  per design; the objection to holistic supply-chain optimisation is that the operational tier is
  too expensive to sit inside a design loop - ZST removes it).
- **SBO (D1)** makes the design search sample-efficient (proven: median 33 evals to the optimum).
- **The interdiction game** supplies the operational objective the design is optimised against.
- The LP CANNOT enter: it cannot evaluate a policy (only loss_mixed), so this target is
  RL-specific by construction - the honest successor to the retired wall-clock scaling claim.

## Design

- Space: the D1 placement x fleet space (300 ODs x N in {2,3,4}), same features.
- Target per design = the frozen A1 generalist's exact best-checkpoint exploitability on that
  design (one forward pass -> fleet occupancy dist -> oracle BR). Computed for the full space once
  (for regret reference only); the optimiser never sees it.
- SBO arm (ensemble LCB, D1 recipe) vs random vs one-shot, 20 repeats, budget 60, n0 15.
- Cross-check row: how the design minimising the TRAINED-policy exploitability compares to the
  design minimising the ORACLE equilibrium (D1's target) - do the strategic optima agree, or does
  designing against the real policy differ from designing against the abstraction? (Either answer
  is a finding.)

## Decision reading (PRE-REGISTERED)

> **PASS:** SBO median evals-to-(regret<=0.01) <= half of random (the D1 bar, on the policy
> target). **Exhibit claim (reported regardless):** the (design -> trained-policy exploitability)
> surrogate is trainable (held-out Spearman > 0.5) and the acquisition loop selects a near-optimal
> design at one-forward-pass-per-evaluation cost. Cross-check: report the rank correlation between
> the policy-exploitability target and the oracle-equilibrium target across designs.

## RESULT (to be appended by the post-A1/D3 chain)
