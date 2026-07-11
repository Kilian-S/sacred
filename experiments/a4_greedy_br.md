# A4: matrix-free submodular greedy best-response interdictor (the regime the LP cannot follow)

- **status: CORE DONE 2026-07-10 (verified, oracle-only); large-K TRAINING integration is the
  recorded remaining step (position 9 in the ordering). `src/baselines/multiconvoy_oracle.py::
  greedy_br_attacker`; verified `tests/test_greedy_br.py`.**

## What

The exact best-response interdictor enumerates C(E, K) interdiction sets and the [occ x iset]
objective matrix; at K=4 that is 1.5M sets, at K=5 22.5M (RAM-infeasible; the measured naive-oracle
wall). For the MISSION objective, a defender occupancy distribution's expected mission-failure is a
monotone SUBMODULAR function of the interdicted edge set (a weighted "at-least-one" coverage over
the convoys' edge-crossing events), so the greedy K-edge best response carries the classic
**(1 - 1/e) approximation guarantee** and never builds the matrix.

## Verification (tests/test_greedy_br.py, real Kaliningrad 62-97 k8)

- **K=1: greedy == exact** best response to random defender supports (abs diff < 1e-9, 4 seeds).
- **K=2: greedy within [(1-1/e)*exact, exact]** (in practice exact here), 4 seeds.

## Reach (timing, uniform occupancy support, 62-97 k8, N=3)

| K | greedy value | greedy time | exact would enumerate |
|---|---|---|---|
| 1 | 0.834 | 1.6 s | 79 isets |
| 2 | 0.898 | 3.2 s | 3,081 |
| 3 | 0.941 | 4.7 s | 79,079 |
| 4 | 0.966 | 6.5 s | 1,502,501 |
| 5 | 0.981 | 8.2 s | 22,537,515 (RAM-infeasible for the exact matrix) |

(Times are the full 364-occupancy support; a trained defender's trailing-window support is a
handful of occupancies -> far faster. rho correlation supported.)

## What it establishes + remaining step

This is the first regime where "the oracle cannot follow" is TRUE rather than rhetorical: at K>=4
the exact matrix is infeasible and the greedy BR is the only available strong attacker, with a
proven bound. **Remaining step (recorded, pre-registered when launched):** wire `greedy_br_attacker`
into `train_multiconvoy.py`'s attacker refresh + the exploitability eval in place of the eager
`env.obj_matrix` (CRITIQUE_INTERDICTION §5.4), then train/evaluate a K=4 and K=5 cell on 35-159.
The BR function + its guarantee are the load-bearing, verified piece; the training cell is an
ordinary run once wired. Not launched tonight (position 9; the keystone arc has the machine).

## DECISION (2026-07-11): the K=5 training cell is DEFERRED as scoped future work

After the expansion programme completed (F2 positive, ZST at city scale + rotation + K/N shift, the
SBO stack, D3-on-Gdansk), the A4 K=5 training cell is deliberately NOT built, for four reasons: (1)
it is the pre-committed drop-first item (CRITIQUE_EXPANSION §9 drop order); (2) its scaling claim is
HEDGED regardless - against column generation / double oracle the exact frontier extends far past
naive enumeration, so the thesis concedes wall-clock scaling in one sentence either way (the honest
scaling story now runs through ZST + D3, both strongly evidenced); (3) wiring the greedy BR into
the trainer + gating the eager objective matrix is a non-trivial refactor of the code every
generation depends on, and the regression risk outweighs one hedged datapoint given how complete
the rest is; (4) the A4-CORE (the matrix-free submodular greedy BR, VERIFIED against exact at K<=2,
reaching K=5 matrix-free in 8 s) already exists and is citable: "the mechanism that scales past the
naive oracle wall exists and is verified; the trained K=5 cell is scoped future work." This is the
defensible, honest position and it matches the critique's own drop-order. Recorded, not launched.
