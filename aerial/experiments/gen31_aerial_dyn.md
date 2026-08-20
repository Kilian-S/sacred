# gen31_aerial_dyn: dynamic aerial routing against an anticipatory pattern-of-life enemy

Registered 2026-07-19. Results 2026-07-19 (corridor hunt), 2026-07-20 (confirmation). No code SHA is pinned in the source ledger.
Artefacts: `analysis/gen31_corridor_hunt.py`, `models/runs/gen31_corridor_hunt.json`, trainer `scripts/train_aerial_dyn31.py`, run directory `models/runs/gen31_dyn/` (per-seed `.json` and `.log`).

## Question

Does one history-aware fleet policy, trained across aerial threat layouts and evaluated zero-shot on never-touched layouts, beat every static object and the payoff-blind dynamic rule family?

## Game

- Layout family: structured double-pinch aerial threat layouts, with open-banded layouts as context only.
- Enemy: anticipatory mixed doctrine, aim distribution softmax(tau) over per-position expected damage against a mixture of defender models (repeat-the-pattern weight, flee-anticipation weight, committing-equilibrium weight).
- Pinned operating point: q = (0.7 repeat, 0.3 flee), tau = 0.10, window w = 2, K=1.
- Defender: fleet N=3 menu-select, per-instance smooth fictitious play, validation checkpoint selection.
- Head columns per route: exposure (static), window recency, and the doctrine column (this sortie's expected damage per route given the window). Information parity is binding, every rule in the family receives the same columns.
- Layout pools: train 18 (12 double-pinch, 6 base), validation 4 (D3000-1, B3100-1), dev-test D2100-2102 (burned by the hunt, iteration diagnostics only), gated D4100-4105 (pristine, confirmation only), context B4000-4001.
- Runs: attempt seeds 0/1/2, confirmation seeds 10/11/12 plus a blinded control on seed 10, 16,000 sorties each.
- Selection rule: validation mean ratio at per-eval checkpoints; the gated set is never used for selection and is evaluated once.
- Scoring: exact stationary damage of the policy-induced window chain; exact dynamic optimum (history_opt) by RVI with the lazy-chain aperiodicity transform; static cap = min(iid_eq, static_opt) per layout.

## Criteria

Phase 0 gates (oracle only, on at least 3 probe layouts, before any training):

- G1: iid_eq / history_opt >= ~1.4, with the multi-start local static optimum within a few % of iid_eq.
- G2: min over the payoff-blind anti-repeat/rotation family >= ~1.25x history_opt.
- G3: fitted doctrine-informed rules computed and recorded, never gating.
- G4: a small trainable function of the pinned feature columns reaches materially below the payoff-blind family towards history_opt.
- G5: trainable asymmetry, values inside (0.02, 0.9).

Confirmation bars, fixed at registration:

- PRIMARY: zero-shot per-layout stationary damage below that layout's static cap on >= 4/6 gated layouts, on >= 2/3 seeds, and pooled, at the validation-selected checkpoint.
- STRONG: pooled <= 2.5x history_opt.
- CAUSAL: the blinded arm (window-frequency and doctrine columns zeroed) lands at about the cap.
- REPORTED, never gating: the full rule-family ladder, the worst-case-versus-committing row, final-iterate drift, per-layout values.

## Baselines

- Static cap: min(iid_eq, static_opt) per layout, the best static object.
- Payoff-blind dynamic family: anti-repeat and rotation rules over every lane spacing, full menu, and eq-support variants, with no payoff information.
- Fitted rules: the doctrine-informed myopic dodge, a temperature-fitted softened dodge, and hedge-composed variants, all oracle-fitted per instance.
- Exact dynamic optimum: history_opt by RVI.
- Blinded control: the identical trainer with the window-frequency and doctrine head columns zeroed, so only a static mixture is expressible.

## Results

Phase 0 (48 cells, exact throughout). Anchors at the pinned operating point, probe layouts s2100/s2101/s2102:

| probe layout | static cap | best payoff-blind rule | myopic dodge | best fitted rule | history_opt |
|---|---|---|---|---|---|
| s2100 | 0.429 | 0.305 | 0.173 | 0.166 | 0.113 |
| s2101 | 0.462 | 0.274 | 0.156 | 0.136 | 0.094 |
| s2102 | 0.464 | 0.294 | 0.186 | 0.173 | 0.096 |

Gate outcomes: G1 3.8-4.9, G2 2.7-3.1, G3 1.45-1.80, G5 values inside the band. G4: the fitted rows reach 0.136-0.173, inside the policy head's function class once the doctrine column is a head feature. Operating point pinned at q = (0.7 repeat, 0.3 flee), tau = 0.10, w = 2, structured double-pinch family.

Confirmation, gated layouts D4100-4105 evaluated once:

| arm | val-selected @ | beats cap | beats payoff-blind family | pooled ratio-to-cap | x history_opt | vs fitted | drift |
|---|---|---|---|---|---|---|---|
| seed 10 | 8000 | 6/6 | 5/6 | 0.523 | 2.09x | 1.48x | none |
| seed 11 | 15000 | 6/6 | 6/6 | 0.517 | 2.07x | 1.46x | none |
| seed 12 | 5000 | 6/6 | 6/6 | 0.505 | 2.02x | 1.43x | none |
| blinded control (seed 10) | 15000 | 0/6 | 0/6 | 1.206 | 4.83x | 3.42x | none |

Criterion outcomes:

- PRIMARY met on 3/3 seeds, 18/18 seed-layout cells, pooled 0.515x the static cap.
- STRONG met, pooled 2.06x the exact dynamic optimum.
- CAUSAL met, blinded control 1.21x the cap at 0/6, with its window and doctrine head weights at 0.00.
- REPORTED: the payoff-blind dynamic family is beaten on 17/18 cells; the oracle-fitted doctrine rules stay ahead of the policy by 1.4-1.5x (per seed 1.48x, 1.46x, 1.43x); worst-case-versus-committing premium 1.22x mean; no last-iterate drift on any arm.
- On open sectors the payoff-blind family sits at 0.09-0.13, below the cap.
