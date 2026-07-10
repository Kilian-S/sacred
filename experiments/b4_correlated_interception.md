# B4: correlated interception (the independence caveat, converted to a robustness curve)

- **status: DONE 2026-07-10 (expansion programme rider; ORACLE-ONLY, no training). Script
  `scratch/correlated_interception_probe.py`; artefact `models/runs/correlated_interception.json`.**

## Question

The headline models interception draws as INDEPENDENT across convoys; a real ambush team catching
a stacked column is positively CORRELATED (`CRITIQUE_INTERDICTION.md` §3.3). Is independence
optimistic or conservative for SACRED's randomised stack, and by how much?

## Model

Within-route common-shock mix parameter rho in [0,1]: rho=0 independent (the default everywhere
else); rho=1 comonotone (all convoys on a route caught all-or-nothing by one shock, prob p_r).
Routes stay mutually independent (distinct edges/teams). Added additively to
`multiconvoy_oracle.objective_value(..., rho=)`; E[fraction lost] is rho-invariant by linearity, so
only the loss-averse mission objective feels it.

## RESULT (oracle equilibrium ladder vs rho, N=3, K=1, mission)

| rho | 35-159 ALNS | 35-159 eq | 62-97 ALNS | 62-97 eq |
|---|---|---|---|---|
| 0.00 (independent, headline) | 0.699 | 0.206 | 0.699 | 0.216 |
| 0.25 | 0.645 | 0.188 | 0.699 | 0.198 |
| 0.50 | 0.583 | 0.169 | 0.690 | 0.178 |
| 0.75 | 0.554 | 0.148 | 0.644 | 0.156 |
| 1.00 (comonotone) | 0.458 | 0.127 | 0.555 | 0.141 |

**Reading:** as correlation rises, BOTH the deterministic optimum and the equilibrium fall (a
stacked column shares one shock, so mission-failure drops), and the SACRED-over-ALNS gap HOLDS
(62-97: 0.483 -> widens to 0.512 mid-range -> 0.414; 35-159: 0.493 -> 0.331, still large).
**Therefore independence (rho=0) is the CONSERVATIVE assumption for the multi-convoy headline:**
positive correlation only makes the randomised stack MORE effective and mission-failure LOWER, so
the reported ladder is a worst case over rho, not a favourable modelling choice. The disclosed
caveat is now a measured Obj-5 robustness curve. (Training a policy under rho>0 is a recorded
option; the oracle result already settles the direction, so it was not spent tonight.)
