# gen22: hold-out rotation, Istanbul held out

Registered 2026-07-11. Results 2026-07-11 (three seeds), 2026-07-16 (disjoint-baseline rows).

Artefacts: `models/runs/r0_screen.json`. Scripts: `scripts/train_generalist.py`, `analysis/disjoint_baseline_probe.py`, `analysis/r0_screen.py`.

## Question

Does cross-city zero-shot transfer hold when the held-out city is Istanbul, the structurally most distant of the four?

## Game

- The gen16 recipe exactly, with the hold-out rotated: train on Kaliningrad, East London and Gdansk; hold out Istanbul (1266 nodes, mega-city arterial grid) entirely.
- 3 seeds, pool-seed 0, 12000 sorties, eval-every 500, 6 held-out ODs.
- Selection: best checkpoint, select-on-train.
- Single rotation cell, not the full leave-one-city-out rotation.

## Criteria

Held-out-Istanbul best-checkpoint mean ratio (select-on-train) <= 2.0, below the random-init reference, and beating loss_det on >= 4/6 ODs. Anchor: gen16 Gdansk 1.677 select-on-test, 1.733 select-on-train.

## Baselines

- random-init: an untrained network on the same held-out ODs, 2.30.
- loss_det: the OD's deterministic-class optimum.
- uniform-disjoint stack: uniform stack over the edge-disjoint routes, 1.145x eq, beating loss_det 6/6.
- inverse-vuln variant: 1.048x eq.

## Results

| seed | best held-out-Istanbul ratio @ sortie | beats loss_det |
|---|---|---|
| 0 | 1.781 @ 6000 | 4/6 |
| 1 | 2.042 @ 1000 | 2/6 |
| 2 | 1.815 @ 1000 | 3/6 |

Held-out-Istanbul best-checkpoint mean 1.880 +/- 0.116 over 3 seeds, against a random-init reference of 2.30. Mean clause met (1.880 <= 2.0) and random-init clause met. The loss_det clause (>= 4/6 ODs) is met on 1/3 seeds; Istanbul's grid carries several ODs where loss_det/eq is only ~1.3-2.0.

Two rotation points now exist for the cross-city claim, Gdansk at 1.677 / 1.733 and Istanbul at 1.880. On the held-out Istanbul ODs the generalist does not beat either disjoint-stack heuristic.
