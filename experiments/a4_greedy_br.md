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
