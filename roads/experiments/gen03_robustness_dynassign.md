# gen03: adversarial training against a non-adversarial control on dynamic assignment
Registered 2026-07-02. Results 2026-07-03. Code `c361b4c` (infrastructure state), `de5ff7d` (launch).
Artefacts: `experiments/gen03_portfolio_pair0.json`, `experiments/gen03_portfolio_pair1.json`, `experiments/gen03_portfolio_v2.json`, `analysis/gen03_aggregate.py`, best-response runs under `models/runs/gen03_robustness_dynassign/br_*`, `scripts/run_generation.py`, `scripts/train_sacred.py`, `scripts/evaluate_portfolio.py`.

## Question
Does adversarial (ATLA) training buy measurable robustness to held-out attacks, relative to a non-adversarially trained but otherwise identical SAC dispatcher?

## Game
- Arena: dynassign, lambda = 0.06.
- Arms: `sacred` (ATLA co-evolution, existing gen02 runs, seeds 0 and 1), `vanilla` (`--vanilla`, protagonist trains every episode with the antagonist inert, seeds 0, 1, 2), `greedy` (`greedy_insertion_policy`, untrained reference).
- Training: 800 episodes, switch-every 50, batch size 32, hidden dim 64, device cpu, eval-every 50, threads 3.
- Asymmetries recorded at registration, both conservative against the robustness claim: vanilla receives roughly twice the protagonist gradient updates per episode, and 3 seeds against sacred's 2.
- Test instances: demand seeds 10_000_019 to +29 (N=30), paired across all cells, held out from every training stream (training uses seed*100003+episode).
- Validation instances: 20_000_019 to +7 (N=8), selection only.
- Protagonists act stochastically (per-episode crc32 seeding); learned attackers act deterministically.
- Selection rule: per arm, minimum mean attacked wait under `targeted` on the validation instances (`evaluate_portfolio.py --select-best`), never on the test attackers or test instances.
- Evaluation portfolio: 30 paired instances per seed pairing.

## Criteria
W = mean total wait over paired instances. D(arm, a) = W(a) - W(none), paired per instance.

Primary: dD = D(vanilla, br_vanilla) - D(sacred, br_sacred), each policy against its own best-response attacker. Success requires dD > 0 with the paired 95% CI excluding 0, in the same direction for both sacred seeds.

Secondary (reported, not gating): dD under `targeted` and `random`; the cross best-response 2x2; clean premium W(sacred, none) - W(vanilla, none), target no more than about +15%; greedy's row.

## Baselines
- `none`: no attacker, the clean baseline for D.
- `random`: scripted, uniform over maskable edges, seeded per instance.
- `targeted`: scripted, blocks the first blockable edge ahead of the truck nearest its goal. Validation (selection) attacker.
- `br_sacred`, `br_vanilla`: learned attackers, `--train-antagonist-only` for about 300 episodes against the selected checkpoint of that arm. Held-out test attacks.
- `greedy`: untrained reactive dispatcher, reference row only.

## Results
Primary dD (95% CI, n = 30 paired instances each): pair0 -291 +/- 500, pair1 -255 +/- 295. Criterion met 0/2 pairings.

Secondary dD: `targeted` -96 and +388; `random` +247 and +54.

Pooled per-seed D against each arm's own best response: sacred {1374, 884}, vanilla {1083, 629, 1865}.

Attacker hierarchy on the same paired instances, D = degradation against clean:

| attacker | D on greedy | D on learned arms |
|---|---|---|
| scripted `targeted` | +4921 +/- 393 (about +79%) | +5434 to +5822 |
| `random` | +1718 +/- 385 | +1770 to +2108 |
| learned best response (300 dedicated episodes each) | +927 to +1276 | +607 to +1865 |

Clean performance: W(none) for sacred and vanilla both about 6.62k to 6.68k, greedy 6.20k (about +7%).

Antagonist training telemetry, all five best-response trainings: true episode reward fell from about 9.0k to 8.4k while Q tripled from 35 to 115.

Paired 95% CIs came in at +/- 300 to 500, against about +/- 1000 for the single-instance protocol.
