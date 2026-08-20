# gen21: travel-objective generalist, the non-adversarial transfer control

Registered 2026-07-11. Results 2026-07-11 (seed 0), 2026-07-13 (seeds 1 and 2, run under gen25).

Artefacts: the vanilla arms in the run directory `models/runs/gen25_dr/`. Script: `scripts/train_generalist.py`.

## Question

Does a generalist trained on travel cost, with no adversary, transfer zero-shot as well as the adversarially trained generalist?

## Game

- Identical to gen16: `--cities kaliningrad,east_london,istanbul --holdout-city gdansk --n-per-city 6 --n-test 6 --pool-seed 0`, 12000 sorties, eval-every 500.
- Changed: `--vanilla`, so the reward is negative normalised fleet travel cost and there is no adversary.
- Unchanged: map conditioning (edge-vulnerability observation plus per-route features), and zero-shot evaluation on the same held-out Gdansk ODs under the oracle best response.
- Seeds: 1 at registration, extended to 3 with the same configuration.

## Criteria

Held-out Gdansk mean best-checkpoint ratio, compared against the adversarial gen16 anchors 1.677 (select-on-test) and 1.733 (select-on-train), and against the random-init reference ~1.99. A ratio materially above the adversarial anchors makes the adversarial ingredient a measurement rather than an inference; a comparable ratio is reported as measured.

## Baselines

- Adversarial generalist (gen16): 1.677 select-on-test, 1.733 select-on-train.
- random-init: an untrained network on the same ODs, ~1.99.
- Domain-randomisation arm (gen25): mission objective with a uniformly random interdictor, 2.056.

## Results

| arm | held-out Gdansk best-checkpoint ratio | final iterate |
|---|---|---|
| vanilla seed 0 | 2.338 | 2.575 |
| vanilla seed 1 | 2.351 | - |
| vanilla seed 2 | 2.372 | - |
| vanilla, n=3 | 2.354 +/- 0.014 | - |

The travel-objective generalist transfers at 2.354 +/- 0.014, above the adversarial generalist (1.68) and above the random-init reference (1.99). Full record of the n=3 extension and the domain-randomisation companion in `gen25_dr_control.md`.
