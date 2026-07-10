# D2: defender-side hardening (the tactical tier of the holistic supply-chain stack)

- **status: DONE 2026-07-10 (expansion programme; ORACLE-ONLY). `scratch/d2_hardening.py`;
  artefact `models/runs/d2_hardening.json`.**

## Question

Given a pre-game budget to reduce edge vulnerabilities (escorts / route clearance), where should
the defender invest, how much does optimal hardening buy against the equilibrium, and does it
change WHERE operational randomisation pays? (Completes the strategic/tactical/operational stack:
harden -> place/size [D1] -> route [SACRED], on one game.)

## Design

35-159 k8, N=3, K=1, mission. Each hardening unit multiplies a chosen edge's interception prob by
(1 - eta), eta=0.5; budget 4 units; greedy allocation (each unit to the edge whose hardening most
reduces the equilibrium, full oracle resolve per candidate), vs a random-allocation baseline.

## RESULT

- **Unhardened equilibrium 0.206; greedy-hardened 0.169 (18% reduction) at budget 4;** loss_det
  (the deterministic/ALNS optimum) falls further, 0.699 -> 0.497.
- **Greedy hardening beats random allocation by +0.034** (random 0.203 +/- 0.006): WHERE the
  budget goes matters, and the surrogate/greedy tier earns its keep (a random escort plan wastes
  most of the budget).
- **The tier INTERACTION (the interesting output): hardening RELOCATES where randomisation pays**
  (equilibrium leader-route mass shift L1 = 0.29): investing in the network changes the operational
  policy's optimal mixed strategy, so the tiers are coupled, not separable - exactly the holistic
  supply-chain claim (strategic investment and operational routing must be co-designed, the review's
  Blanning/Sacks metamodel-coupling motivation).

**What it establishes:** the tactical hardening tier is a well-posed, oracle-evaluable design
problem whose optimum couples to the operational game; with D1 (placement/fleet) and D3 (the
composite over the trained policy), the project demonstrates a three-tier computational stack on a
single interdiction game. Future work (recorded): surrogate-guided hardening (the D1 loop on this
target) and joint place+harden co-optimisation.
