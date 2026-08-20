# gen45_unified_corridor: real-corridor dynamic routing on the unified substrate

Registered 2026-08-09. Results 2026-08-09 (corridor hunt), 2026-08-10 (attempt wave, confirmation, verification, worst-case probe). Scorer `analysis/gen45_score.py` committed at `ea5e293` before the confirmation artefacts existed.
Artefacts: `analysis/gen45_corridor_hunt.py`, `models/runs/gen45_hunt.json`, `models/runs/gen45_hunt.log`, trainer `scripts/train_gen45_unified.py`, wave script `analysis/gen45_batch.sh`, run directory `models/runs/gen45_unified/` (`attempt_seed*.json`, confirmation artefacts, `worstcase.json`), gate `analysis/gen45_gate.py`, verifier `analysis/gen45_verify.py`, worst-case probe `analysis/gen45_worstcase.py`, environment `src/envs/aerial_conceal.py`.

## Question

Does the real-corridor dynamic positive reproduce when its game is rebuilt on the gen39 substrate, so that the aerial acts share one game?

## Game

- Theatre: kgd_gvardeysk, `data/maps/theatre_kgd_gvardeysk_vec.json`.
- Terrain table: v2 via `terrain_v2(hidden_leth=1.0, conceal_reach=0.85)`. Terrain sets reach and lethality (open 3.5/0.90, field 2.5/0.85, forest concealed-reach 0.85 x open at 0.55, urban emplaceable 0.45); forest hides without blinding; urban blocks line of sight with the self-polygon exemption.
- Range scale: 0.7.
- Sites: quota sampler, `n_sites=200`, spacing 2.0, standoff 4.0, non-grid points whose class shares match the terrain composition.
- Hidden field: `resample_field` multiplier, band (0.55, 1.0). Lethality = terrain class x field draw. The field is what varies per instance.
- Menu: as `ConcealBase` builds it on this substrate, 14 geometric lanes plus terrain-aware cover routes screened against the v2 field, pinned by construction.
- Enemy: DOC32 doctrine components q = (0.6 punish-the-window, 0.2 pre-aim-the-escape, 0.3 anti-repeat-anticipation), softmax tau 0.10, aiming over all candidate sites with a uniform prior (full-map relocation each serial). No reveal channel in this act.
- Defender: fleet N=3, 40-serial episodes, mission damage, head columns per route = [exposure, recency (window frequency), doctrine column].
- Window w: 2 (pinned by the hunt; w in {2,3} was the pre-declared freedom, alongside q and tau; the substrate was frozen).
- Scoring: exact stationary damage of the policy-induced window chain; exact optimum by damped RVI; static CAP = min(iid_eq, static_opt) per field.
- Field seeds: hunt 45001-45012, train 45300-45317, validation 45400-45403, dev-test 45101-45102, gated 45200-45205 (pristine, confirmation only).
- Runs: attempt seeds 0/1/2 and confirmation seeds 10/11/12 plus a blinded control on seed 10, 16,000 sorties each.
- Selection rule: validation-selected checkpoints on 45400-45403; the gated set is evaluated once, via `--eval-gated`, and never used for selection.

## Criteria

Phase 0 gates over the 12 hunt fields:

- G1: static CAP / exact optimum >= 2.0 minimum across fields.
- G2: best payoff-blind dynamic rule / optimum >= 1.25 on >= 10/12 fields.
- G3: fitted doctrine-informed rules reported, never gating.

Attempt gate before the confirmation wave (`analysis/gen45_gate.py`): each seed's validation-selected checkpoint must beat both the static cap and the whole payoff-blind family on both dev fields, 3/3 seeds.

Confirmation bars, fixed at registration:

- PRIMARY: zero-shot per-field stationary damage below that field's static CAP on >= 4/6 gated fields, on >= 2/3 seeds, and pooled below the pooled cap.
- STRONG: pooled <= 2.5x the exact dynamic optimum.
- CAUSAL: the blinded control beats the cap on 0/6 fields, with its recency and doctrine weights pinned at zero.
- REPORTED, never gating: the beats-payoff-blind-family count over the 18 seed-field cells, the fitted-rule ladder, the worst-case-versus-committing premium, checkpoint drift.

## Baselines

- Static CAP: min(iid_eq, static_opt) per field.
- Payoff-blind dynamic family: rotation and anti-repeat rules with no payoff information.
- Fitted rules: doctrine-informed rules fitted per instance against the hidden field and the enemy's response model (oracle caps).
- Exact optimum: history-optimal value by damped RVI.
- Blinded control: the identical trainer with the recency and doctrine head columns zeroed.
- Static equilibrium: the stacked equilibrium mixture, whose own worst case equals the equilibrium value on every field (reference row 1.00 exactly), used as the worst-case denominator.

## Results

Substrate as frozen: R=26 routes (14 geometric lanes, 12 terrain-aware), H=200 quota sites (open 81 / field 66 / forest 34 / urban 19).

Flat-limit regression, run before any gate number: max |stepdmg difference| 3.9e-12 and |history_opt difference| 1.4e-13 between `DynTheatre` and `ConcealDyn` with one team, huge sigma and no class mask, on the same base and field.

Hunt gates over fields 45001-45012 at DOC32, tau 0.10, w=2. G1 reaches a minimum of 3.71 against a bar of 2.0 (range 3.71-4.31), G2 clears 1.25 on 12/12 fields against a bar of 10/12 (range 2.01-2.82), and G3 has a median of 1.11 with a floor of 1.00-1.01 on two fields. Pinned operating point, w=2, DOC32 q=(0.6, 0.2, 0.3), tau 0.10.

Attempt wave, development fields:

| seed | val-selected @ | VAL | dev ratio | beats CAP | beats blind family | alpha | rw [exposure, recency, doctrine] |
|---|---|---|---|---|---|---|---|
| 0 | 12,000 | 0.314 | 0.339 | 2/2 | 2/2 | 0.21 | [-0.56, -10.33, -22.24] |
| 1 | 5,000 | 0.324 | 0.348 | 2/2 | 2/2 | 0.20 | [+0.51, -5.80, -19.71] |
| 2 | 9,000 | 0.324 | 0.347 | 2/2 | 2/2 | 0.23 | [+0.72, +3.06, -20.19] |

Attempt gate: pass 3/3.

Confirmation, gated fields 45200-45205 evaluated once:

| seed | val-selected @ | beats CAP | beats payoff-blind family | mean ratio-to-cap | mean x optimum | drift (sel -> final) |
|---|---|---|---|---|---|---|
| 10 | 14,000 | 6/6 | 6/6 | 0.347 | 1.44 | 0.312 -> 0.328 |
| 11 | 16,000 | 6/6 | 6/6 | 0.352 | 1.47 | 0.324 -> 0.324 |
| 12 | 6,000 | 6/6 | 6/6 | 0.355 | 1.48 | 0.323 -> 0.331 |
| blinded control (seed 10) | - | 0/6 | 0/6 | 1.242 | - | - |

Per gated field, with references identical across seeds:

| field | CAP | best payoff-blind rule | fitted (oracle caps) | exact optimum | seed10 | seed11 | seed12 |
|---|---|---|---|---|---|---|---|
| 45200 | 0.1775 | 0.1119 | 0.0516 | 0.0414 | 0.0601 | 0.0603 | 0.0613 |
| 45201 | 0.2106 | 0.1107 | 0.0593 | 0.0522 | 0.0756 | 0.0774 | 0.0779 |
| 45202 | 0.2008 | 0.1180 | 0.0581 | 0.0491 | 0.0689 | 0.0716 | 0.0713 |
| 45203 | 0.1950 | 0.1173 | 0.0499 | 0.0439 | 0.0644 | 0.0654 | 0.0664 |
| 45204 | 0.2005 | 0.1255 | 0.0586 | 0.0467 | 0.0699 | 0.0694 | 0.0709 |
| 45205 | 0.1959 | 0.1245 | 0.0620 | 0.0505 | 0.0708 | 0.0717 | 0.0714 |

Criterion outcomes:

- PRIMARY met on 6/6 gated fields for 3/3 seeds, all 18 cells, pooled ratio-to-cap 0.351.
- STRONG met at 1.46x the exact dynamic optimum.
- CAUSAL met, blinded control 0/6 at 1.242x the cap, with recency and doctrine weights at exactly 0.00 and the exposure weight trained freely.
- REPORTED: the payoff-blind dynamic family is beaten on 18/18 seed-field cells.
- The fitted rules remain ahead, at 0.0499-0.0620 against the policy's 0.060-0.078, a factor of 1.22.
- Drift is 0.312 -> 0.328, 0.324 -> 0.324, 0.323 -> 0.331.
- Head weights: the doctrine column trains to -19.7 to -21.6 on every seed and every wave; seed 11's selected checkpoint carries a positive recency weight (+6.78) and scores 6/6, as does attempt seed 2 (+3.06).

Independent re-scoring (`analysis/gen45_verify.py`, selection logic written separately from the scorer) reproduces every value above from the raw artefacts, including all 42 per-field values to four decimals, the pooled 0.351x cap and 1.46x optimum, the blinded control at 1.242x with 0/6 and its recency and doctrine weights at 0.000000 across all 16 of its evaluations, and the fitted-rules factor 1.22. Gated references are byte-identical across the three confirmation artefacts.

Worst-case versus a committing enemy (eval-only, `analysis/gen45_worstcase.py`, `models/runs/gen45_unified/worstcase.json`). Each confirmation seed's validation-selected checkpoint yields its stationary marginal route mixture per gated field, the enemy abandons the doctrine and commits to the single site maximising the stacked fleet's damage against that mixture, and the ratio is taken to the field's static equilibrium value. The premium is pooled 1.52x over the 18 seed-field cells, with seed means 1.53 / 1.54 / 1.48 and a per-cell range of 1.42-1.63.
