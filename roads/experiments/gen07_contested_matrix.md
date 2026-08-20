# gen07: exploitability matrix on the contested arena, and its pre-launch gates
Registered 2026-07-06. Results 2026-07-06. Code: branch `gen07-contested`, no SHA pinned.
Artefacts: `analysis/capacity_probe.py`, `analysis/stress_sweep.py`, `analysis/hybrid_lever_probe.py`, `scripts/br_gate.py`, `scripts/train_sacred.py`, `scripts/evaluate_portfolio.py`.

## Question
Does adversarial co-training against an adversary population produce a dispatcher that is harder to exploit by attacks tailored to it than an identically trained non-adversarial dispatcher, at bounded clean cost?

## Game
- Arena: contested resupply, dynassign dynamics with `antag_reach="route"` and a full-block antagonist, so both scripted and learned attacks aim along committed routes. Truck capacity 1. Load lambda and budget were left to be pinned by the stress and recoverability probes.
- Arms, identical env, reward, nets and hyperparameters, only training-time exposure differing: `greedy` (untrained deterministic reactive reference), `vanilla` (no adversary, 3 seeds), `dr` (random-attack exposure under the same curriculum schedule as sacred, 3 seeds), `sacred` (curriculum ATLA against the adversary population of scripted seeds plus successively trained best responses, mixture weights logged, 3 seeds), `vanilla@tau` (evaluation-time row: selected vanilla checkpoints sampled at a temperature matched to sacred's realised policy entropy), optional `erb_*` seeded variants.
- Test instances: 30 paired, seed base 10_000_019. Validation instances: seed base 20_000_019. Protagonists stochastic with the standard per-episode crc32 seeding; best-response attackers deterministic.
- Selection rule: per arm, minimum mean attacked W on validation instances under a third scripted attacker variant reserved for selection only, never on test attacks or test instances.
- Best-response gate (B9.iv) as run: victim = frozen greedy on the contested arena, antagonist trained 300 episodes with `--arena contested --reward-baseline twin --antag-target-entropy 0.5 --gamma 0.997`, evaluated with `evaluate_portfolio.py --problem contested --br gate=<actor> --attackers none,random,pathrand,targeted,br_gate --instances 24 --seed-base 20000019`.

## Criteria
W = mean total wait; D(a, atk) = W(atk) - W(none), paired per instance. Exploitability Expl(a) = max over atk in the tailored portfolio P(a) of mean D(a, atk).

Primary: dExpl = Expl(vanilla) - Expl(sacred), per seed pairing and pooled. Success requires pooled dExpl > 0 with a paired-bootstrap 95% CI excluding 0 (resampling instances and recomputing the per-arm max within each resample), at least 2/3 seed pairings individually positive, and both bounds below holding.

Secondary (reported, not gating): Expl(sacred) < Expl(greedy) with the portfolio fitted to greedy; the held-out D and dD rows under random, pathrand and targeted; the `dr` and `vanilla@tau` rows under the full portfolio; per-arm realised policy entropy at the selected checkpoints; budget-axis sweep curves.

Bounds: competence gate, every learned arm's W(none) within +15% of greedy's, else that arm's rows are flagged competence-compromised; clean-premium bound, W(sacred, none) - W(vanilla, none) no more than +10%. Reporting rule: pooled instance-level CI, per-pairing sign consistency and the 3-pairing t sensitivity together.

Pre-launch gates: B9.i suite green on the branch; B9.ii timing probe and published compute envelope; B9.iii greedy band reachable with a fitted scripted attack in the +30 to +60% band on greedy and attacked delivery in the 0.4 to 0.8 trainable band; B9.iv best-response gate, PASS = D(br against greedy) >= 1.25 x D(random against greedy) on held-out validation instances, STRONG = additionally D(br) >= D(targeted); B9.v coping-channel probe, an epsilon-randomised greedy must reduce D under the fitted `targeted` attack relative to deterministic greedy with a CI excluding 0.

## Baselines
- `none`: no attacker, the clean baseline for D.
- `random`: undirected floor, kept inside the exploitability max as a sanity lower bound.
- `pathrand`, `targeted`: scripted families, with per-victim parameter fitting on validation instances only.
- `br_a`: one learned best-response antagonist trained against frozen arm `a` for an equal 300-episode budget.
- `greedy`: untrained deterministic reactive dispatcher.

## Results
Arena-scoping probes (greedy rollouts only, no training):

| capacity | clean W | attacker bite D | exploitability lever |
|---|---|---|---|
| 1 | 5908 | 4768 | 217 |
| 3 | 1133 | 1783 | -39 |
| 5 | 1036 | 697 | -88 |

Load probe over 12 instances. At lambda = 0.06 the lever is 217, ratio 1.36, delivery 0.75; at lambda = 0.08 the lever is 491, ratio 2.5, delivery 0.56; collapse follows at 0.10 to 0.12. The powered sweep uses 40 instances with a per-instance lever 95% CI. At the sweet spot the greedy-measured lever is about 10% of D and is a lower bound (crude epsilon-random assignment).

Best-response gate B9.iv, 24 held-out validation instances, greedy W(none) = 6729:

| attacker | D against greedy |
|---|---|
| random | +4733 +/- 213 |
| pathrand | +4618 +/- 176 |
| targeted (scripted) | +4920 +/- 218 |
| br_fixed (trained best response) | +1666 +/- 185 |

br/random = 0.35 against the 1.25 PASS bar. PASS criterion met 0/1. STRONG criterion met 0/1.

Across snapshots: D(br) = 457 at episode 100, 1890 at episode 200, 1666 at episode 300.

Antagonist telemetry: entropy stayed pinned at about 2.2 throughout despite the lowered 0.5 target, alpha collapsed from 0.80 to 0.08, antagonist reward stable at about 4000, and Q-spread across block actions was near zero. Attack magnitudes saturate at about 4600 to 4920, with random at 4733, about 96% of scripted `targeted` at 4920.

The training matrix was not run, so none of the matrix decision metrics (dExpl, the competence gate, the clean-premium bound) were evaluated. Gate B9.v was not run.
