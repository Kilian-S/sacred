# gen27: zero-shot dynamic hedging on a held-out city

Registered 2026-07-16. Results 2026-07-16 (trained arms, static and naive-dynamic rows, worst-case row), 2026-07-17 (no-window control), 2026-07-23 (exact dynamic-optimum yardsticks). Code 2f4ffd5 for the trained runs.

Artefacts: `models/runs/gen27_dyn_generalist/seed{0,1,2}.json`, `models/runs/gen27_dyn_generalist/seed0_nowin.json`, `static_rows.json`, `worstcase.json`, `models/runs/critique_followup_probes.json`, `models/runs/dyn_yardstick_repair.json`. Scripts: `scripts/train_dyn_generalist.py`, `analysis/gen27_static_rows.py`, `analysis/critique_followup_probes.py`, `analysis/dyn_exact.py`.

## Question

Does one history-aware policy, trained on three cities, beat the static equilibrium cap zero-shot on a fourth, never-seen city's dynamic games?

## Game

- Instance: fleet-route stacked routing; adversary = softmax best response (tau=0.15) to the trailing w=3 realised-route window.
- Reward: analytic expected mission failure; episodes = S=40 sorties chained with gamma=0.95; the window is cleared per episode.
- Pools: kaliningrad + east_london + istanbul x 6 ODs for training (18 instances), Gdansk x 6 held out entirely; pool-seed 0; N=3, K=1, vulnerability band 0.15-0.95, k8 menus.
- Conditioning: per-instance menus plus [cost, worst-vuln, window-frequency] route features on every observation; head-term lr 3e-2; no route_bias term.
- Yardsticks per instance, computed exactly at pool build: static_det (deterministic static reference), iid_eq (the static equilibrium mixture's value against this adversary, the static cap), and the exact dynamic optimum over the window MDP (Karp minimum-mean-cycle, equivalently damped RVI, `analysis/dyn_exact.py`).
- Arms: history-aware policy on seeds {0,1,2}; no-window control on 1 seed, with the window feature zeroed.
- Budget: 12,000 sorties, eval-every 500, train evaluation n=400 per instance, held-out n=1000 per instance, per-eval checkpoints.
- Selection: select-on-train (mean train-instance ratio to iid_eq); select-on-test and the final iterate reported alongside.
- Scoring: each seed is scored against its own stored references. The equilibrium LP has degenerate optima, so iid_eq varies by roughly 1-2% across processes on identical games; static_det and the dynamic optimum do not depend on the LP vertex.

```bash
PYTHONPATH=. .venv/bin/python scripts/train_dyn_generalist.py \
  --sorties 12000 --eval-every 500 --seed $S --threads 3 \
  --json-out models/runs/gen27_dyn_generalist/seed$S.json \
  --ckpt-dir models/runs/gen27_dyn_generalist/seed${S}_ckpts
# control: --no-window --seed 0, json/ckpt paths suffixed _nowin
```

## Criteria

- PRIMARY: at the select-on-train checkpoint, held-out Gdansk mean ratio to iid_eq < 1.0 and < 1.0 on >= 4/6 ODs, on >= 2/3 seeds.
- STRONG: pooled held-out mean per-sortie loss <= 2x the per-instance dynamic optimum.
- CAUSAL CONTROL: the no-window arm lands at ratio ~1.0.
- REPORTED, not gated: worst-case exploitability of the marginal route mixture under each held-out OD's oracle best response, as a premium over its single-shot stacked V_eq.

## Baselines

- iid_eq: the static equilibrium mixture played against the pattern-of-life adversary; the pre-registered bar.
- Uniform-disjoint heuristic: uniform stack over the edge-disjoint routes, static value.
- Inverse-vulnerability heuristic: the inverse-vulnerability weighted stack, static value.
- Local-search static optimum: multi-start local search over static mixtures, a local optimum.
- Rotation: deterministic round-robin over the disjoint routes.
- Anti-repeat, full menu: uniform over menu routes not in the last-3 window.
- Anti-repeat, disjoint: uniform over the disjoint routes not in the last-w window.
- Exact dynamic optimum: the optimal history-conditioned value on the window MDP.
- No-window arm: the identical policy and architecture with the window feature zeroed.

## Results

Naive rules on the 6 held-out Gdansk instances, against the same adversary (w=3, tau=0.15), oracle-exact.

| OD | iid_eq | rotation | anti-repeat, disjoint | anti-repeat / iid_eq |
|---|---|---|---|---|
| 249-95 | 0.223 | 0.207 | 0.118 | 0.53 |
| 106-173 | 0.213 | 0.203 | 0.119 | 0.56 |
| 351-210 | 0.232 | 0.206 | 0.117 | 0.50 |
| 146-296 | 0.189 | 0.196 | 0.115 | 0.61 |
| 275-72 | 0.218 | 0.203 | 0.115 | 0.53 |
| 193-278 | 0.187 | 0.186 | 0.112 | 0.60 |

All six held-out ODs have m=3 disjoint routes, and the w=3 window covers the whole disjoint set.

Trained history-aware arms, 12,000 sorties each, select-on-train selection.

| seed | selected sortie | held-out mean ratio to cap | beats cap | ratio to exact optimum | select-on-test | final iterate |
|---|---|---|---|---|---|---|
| 0 | 11,000 | 0.605 | 6/6 | 1.868 | 0.602 | 0.647 |
| 1 | 9,520 | 0.644 | 5/6 | 1.991 | 0.615 | 0.816 |
| 2 | 11,000 | 0.666 | 5/6 | 2.060 | 0.630 | 0.631 |

Pooled held-out ratio to iid_eq 0.639 +/- 0.025 over 3 seeds. Pooled ratio to the exact dynamic optimum 1.973x. PRIMARY criterion met on every clause, 3/3 seeds. STRONG criterion met. Held-out OD index 1 (106-173) sits at 0.90-1.07 across seeds.

Static baseline rows (oracle-exact, `static_rows.json`): on every held-out OD the local-search static optimum improves on the equilibrium-mixture cap by only 2-5% (0.179-0.227 against iid_eq 0.187-0.236), and both max-flow heuristics' static values sit within +/-5% of the cap.

Naive-dynamic baseline rows: anti-repeat over the full menu is 1.368x the cap in the mean and worse than static play on 5/6 ODs; anti-repeat over the disjoint routes is 0.50-0.61x the cap.

Rows restated against the exact dynamic optimum (`models/runs/dyn_yardstick_repair.json`).

| row | value |
|---|---|
| held-out ratio to optimum, per seed (select-on-train) | 1.868 / 1.991 / 2.060 |
| pooled | 1.973x |
| anti-repeat over disjoint routes, per held-out OD | 1.63-1.85x |
| rotation, per held-out OD | 2.75-3.16x |
| exact optimum, per held-out OD | ~0.065-0.072 |

Worst-case row (seed-0 select-on-train checkpoint, marginal route mixture under each OD's oracle best response, versus its single-shot stacked V_eq): premiums 1.43 / 1.91 / 1.56 / 1.60 / 1.49 / 1.44x, mean ~1.57x.

No-window causal control (seed 0, 12,000 sorties): held-out ratio to cap 1.434 at select-on-train, beats the cap on 0/6 ODs, final iterate 2.206. The window-feature weight stayed at 0.00 throughout. Control criterion outcome recorded against the sighted arms' 0.639 and beats-cap 6/6, 5/6, 5/6.
