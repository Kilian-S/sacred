# A3: the amortisation benchmark (the honest successor to the retired wall-clock claim)

- **status: DONE 2026-07-10 (expansion programme; EVAL-ONLY). `scratch/amortisation_benchmark.py`;
  artefact `models/runs/a3_amortisation.json`.**

## RESULT (40 fresh held-out instances, pool-seed 7, N=3, K=1)

| | per instance | cumulative (40) | quality |
|---|---|---|---|
| oracle LP re-solve | 5.1 ms | 0.20 s | exact (ratio 1.00) |
| generalist forward pass | 3.4 ms | 0.13 s | ratio 1.90 +/- 0.34 |

**The honest frame (as pre-registered, no overclaim):** at this instance size the LP is FASTER AND
EXACT per instance (5.1 ms, ratio 1.0) than the trained policy (3.4 ms, ratio 1.9); the naive
amortisation crossover is astronomically far (~5.5M instances) because the per-instance LP is so
cheap here. So **wall-clock amortisation does NOT favour the policy at K=1 small instances** - the
2026-07-10 retirement of the wall-clock scaling claim (gen09 ledger) stands, confirmed by direct
measurement. The policy's defensible case is exactly the two things the LP CANNOT do: (i) it never
re-solves - it transfers across instances by a forward pass (ZST, A1); (ii) it can be PRICED inside
a design loop over the trained policy's operational exploitability, where the LP cannot participate
at all (D3). The amortisation story runs through those, not the clock. Reported as measured; this
is the honest closure of the scaling narrative.
