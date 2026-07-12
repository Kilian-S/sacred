# ZST map robustness (A2 + A3): does the generalist READ the threat map, and does it survive intel error?

- **status: PRE-REGISTERED 2026-07-12 (NEXT_STEPS_MASTER Block A items A2/A3; EVAL-ONLY, no
  training; autonomous authority). Binding now; results appended below.**
- **git SHA:** the commit landing this ledger + `scratch/map_robustness_eval.py`.

## Why (CRITIQUE_12-07-26.md §3.1, §3.3)

Every threat map any policy has ever trained on is an affine transform of edge length
(`length_band_vulnerability`), so gen16's "map-conditioned transfer" is currently observationally
equivalent to "geometry-conditioned transfer" (probe: route cost vs route worst-vulnerability
|corr| 0.60-0.99 on 8/8 pool instances; geometry-decorrelated maps move the equilibrium strategy
by L1 0.44-1.03). Separately, every trained defender has always OBSERVED the true map; intel
error is untested. Both are eval-only questions on the frozen gen16 actor.

## Design (fixed before looking)

**Frozen policy:** the gen16 seed-0 actor, TAP over the three checkpoints centred on its selected
best (ep 500/1000/1500), i.e. exactly the estimator of the banked zero-shot K/N rows
(`scratch/zst_kn_rows.py`). Instances: the 6 held-out Gdansk ODs (pool-seed 0 = the gen16 test
set). Random-init reference net on identical footing in every condition.

**A2 (shuffled reality: does it read the map?):** per instance, 3 seeded permutations of the
vulnerability values ACROSS ITS CANDIDATE EDGES (the minimal decisive intervention; non-candidate
edges keep true values as geometry-consistent background). Each shuffled map defines a NEW game
(same routes, `survival_intercept_fn(shuffled)`): recompute its equilibrium, loss_det and BR
matrix; the policy OBSERVES the shuffled map (edge-vulnerability column + recomputed per-route
[cost, worst-vuln] features); score its exact occupancy distribution under the SHUFFLED game's
oracle BR; report ratio to the SHUFFLED game's equilibrium.

> **Pre-committed reading (A2):** TRACKS THE MAP = shuffled-map mean ratio <= 2.0 AND below the
> random-init reference on the same shuffled games (the gen16-style bar; the map-conditioning
> claim is then EARNED and extends to threat fields decorrelated from geometry). READS GEOMETRY =
> shuffled mean at or above random-init (no edge on decorrelated maps): the thesis wording changes
> to "conditioned on road geometry under a geometry-consistent threat model", and the recorded fix
> is a randomised-map training pool. Between = partial map-reading, reported as measured.
> Sanity row required either way: the TRUE-map ratio through this harness must reproduce ~1.7x
> (the banked N3K1 sanity value); if it does not, the harness is wrong, not the policy.

**A3 (intel error: reality fixed, observation corrupted):** the TRUE game scores everything
(oracle BR of the true map; ratio to the true equilibrium); only the OBSERVED map is corrupted:
(i) shuffle-fraction f in {0.25, 0.5, 1.0} (a seeded random f-subset of candidate edges gets its
values permuted); (ii) multiplicative noise sigma in {0.1, 0.25, 0.5}
(p' = clip(p*(1+eps), 0.05, 0.99), eps ~ N(0, sigma)); 3 draws each. Route features recomputed
from the corrupted map. Anchors: the true-map policy row (no corruption) and random-init.

> **Pre-committed reading (A3):** GRACEFUL = mean ratio degrades monotonically but stays below
> random-init up to f = 0.5 / sigma = 0.25 (a robustness exhibit: the hedge survives moderate
> intel error). CLIFF = the policy falls to or past random-init at the smallest corruption
> (a scope sentence: claims are conditional on accurate threat intelligence). Either is one
> figure/table; nothing is tuned post hoc.

## Command (pinned)

```bash
PYTHONPATH=. .venv/bin/python scratch/map_robustness_eval.py \
  models/runs/gen16_multicity/seed0_ckpts/actor_ep1000.pt \
  --json-out models/runs/zst_map_robustness.json
```

## RESULT (2026-07-12, eval-only, ~3 min): A2 bar PASSED; A3 maximally graceful; the constant-map diagnostic REATTRIBUTES the mechanism

One reproducibility fix during the run (disclosed): candidate-edge order was originally sorted by
`repr(frozenset)`, which depends on the per-process string-hash seed, so the seeded shuffles were
not reproducible across invocations (three early runs = three valid draws, gen mean 1.83-1.86,
rand 2.09-2.17, beats 12-16/18: the conclusion was draw-stable). Fixed to a canonical node-id sort
key; the pinned command now reproduces exactly (verified twice). Numbers below are the canonical
run. Artefact: `models/runs/zst_map_robustness.json`.

| condition | gen (TAP, seed-0 window 500/1000/1500) | random-init | note |
|---|---|---|---|
| SANITY: true map, true game | **1.71x** [2.13, 2.33, 1.20, 1.69, 1.52, 1.41] | 1.99x | reproduces the banked N3K1 sanity row exactly; harness valid |
| **A2: SHUFFLED reality (18 cells)** | **1.80x** | 2.19x | beats loss_det 13/18; **pre-registered "tracks" bar PASSED** (<= 2.0 and < random) |
| A3 shuffle-fraction 0.25 / 0.5 / 1.0 | 1.70 / 1.72 / 1.74 | 1.99 | reality true, observation corrupted |
| A3 multiplicative sigma 0.1 / 0.25 / 0.5 | 1.72 / 1.78 / 1.78 | 1.99 | " |
| DIAGNOSTIC (post-hoc, labelled): information-free constant map | **1.80x** | 1.99 | all candidate vulns observed as 0.55 |

**What is established (three findings, all load-bearing for the ZST wording):**
1. **The transfer edge survives threat fields decorrelated from geometry (A2 bar passed):** on 18
   shuffled-reality cells the frozen generalist scores 1.80x those games' own equilibria vs
   random-init 2.19x, beating each game's deterministic optimum on 13/18. The zero-shot claim is
   therefore NOT an artefact of the length-derived threat-map family (the CRITIQUE_12-07-26 §3.1
   exposure is closed on the favourable side).
2. **Intel-error robustness is essentially total (A3):** fully shuffling the OBSERVED map while
   reality stays fixed costs +0.03 (1.71 -> 1.74); the strongest multiplicative noise costs +0.07.
   Against the pre-registered reading this is GRACEFUL at every level: the deployment does not
   depend on accurate threat intelligence.
3. **BUT the mechanism is NOT per-edge map-reading (constant-map diagnostic):** an
   information-free observed map costs only +0.09 (1.71 -> 1.80, still far below random 1.99). The
   per-edge vulnerability observation contributes little; the hedge is carried by the
   geometry/cost pathway plus multi-instance adversarial training, and it happens to be robust
   ACROSS threat fields rather than adaptive TO them.

**Consequence for the thesis (wording rule, binding for the storyline rewrite):** replace every
"the policy conditions on the threat map" sentence with the measured statement: *the generalist
learns a geometry-informed, threat-robust calibrated hedge: its zero-shot advantage persists on
threat fields decorrelated from geometry (1.80x vs random 2.19x) and is insensitive to
threat-map observation error (a fully wrong map costs +0.03), but per-edge map-reading is not the
mechanism (an information-free map costs +0.09).* The gen15 mechanism sentence ("give the policy
the map and transfer works") is superseded: what ZST-0 lacked was multi-instance training +
transferable route features, not the map observation per se. Note this STRENGTHENS the ZST-vs-LP
framing: an exact solver requires the true instance model at decision time; the trained hedge
keeps its edge with a wrong or absent map.
