# gen39: concealment game and LLM force composition

Registered 2026-07-25. Results 2026-07-26 to 2026-08-13. Code: each step pins its own commit in the repository; no SHA is printed in this record.

Artefacts: `models/runs/gen39_screen2.json`, `results/gen39_conceal_cost_v2.txt`, `models/runs/gen39_compose/`, `models/runs/gen39_step3/*.json`, `models/runs/gen39_phase1_confound.json`, `models/runs/gen39_phase1b_scores.json`, `models/runs/gen39_phase1c.json`, `models/runs/gen39_phase1d.json`, `models/runs/gen39_phase1e.json`, `models/runs/gen39_phase1e_thinking*.json`, `models/runs/gen39_phase1f.json`, `models/runs/gen39_step5/*.json` (including `curricula_qwenthink.json`, `curricula_qwenthink2.json`, `step5c_verdict.txt`, `step5e_zeroshot.json`), `models/runs/gen39_zeroshot.json`, `zeroshot2_build.json`, `zeroshot2.json`, `models/runs/gen42_ladder/step2_rescore.json`.

Code: `src/envs/aerial_theatre_vec.py`, `src/envs/aerial_conceal.py`, `src/redforce.py`, `scripts/train_gen39_conceal.py`; screen and operating point `analysis/gen39_screen2.py`, `analysis/gen39_read_screen.py`, `analysis/gen39_read_scoping.py`, `analysis/gen39_read_operating_point.py`, `analysis/gen39_pick_operating_point.py`, `analysis/gen39_pinned_cell.py`, `analysis/gen39_conceal_cost.py`, `analysis/gen39_site_map.py`; composition and diagnostics `analysis/gen39_compose.py`, `analysis/gen39_step3_score.py`, `analysis/gen39_phase1_confound.py`, `analysis/gen39_phase1b_score.py`, `analysis/gen39_phase1c.py`, `analysis/gen39_phase1d.py`, `analysis/gen39_phase1d_b3.py`, `analysis/gen39_phase1e.py`, `analysis/gen39_phase1e_thinking.py`, `analysis/gen39_phase1f.py`, `analysis/gen39_repair_rerun.sh`; step 5 and transfer `analysis/gen39_step5_score.py`, `analysis/gen39_zeroshot.py`, `analysis/gen39_zeroshot2.py`, `analysis/gen39_step5c_prep.py`, `analysis/gen39_step5e_llama_prep.py`, `analysis/gen39_step5e_qwen3_prep.py`, `analysis/gen39_step5e_zeroshot.py`, `analysis/gen39_step2_think_rescore.py`.

## Question

Does concealment, cover that denies the defender's observation channel at the price of reach and lethality, make the attacker's force design a real decision? And can a language model contribute to that design, either by composing forces or by authoring a training curriculum for SACRED?

## Game

Terrain table v2 with the asymmetric forest rule (forest conceals the team without blinding it; urban keeps both properties, since buildings are vertical obstacles). Ranges are relative to the standing anchor (v1 open = 2.5 km at kgd scale), with a global multiplier on top.

| class | emplaceable | r (rel) | r @ kgd | p_max | reveals on engagement | blocks LOS |
|---|---|---|---|---|---|---|
| open | yes | 1.4 | 3.5 km | 0.90 | yes | no |
| field | yes | 1.0 | 2.5 km | 0.85 | yes | no |
| forest | yes | 0.6 | 1.5 km | 0.55 | no | no |
| urban | yes | 0.4 | 1.0 km | 0.45 | no | yes |

- Water and sea are not emplaceable and do not block line of sight; alpine is not emplaceable and blocks it.
- A site on revealing ground (open, field) that engages the flight is visible to the defender from the next serial; a site on concealed ground (forest, urban) never becomes visible.
- Reveal follows exposure, not a kill (the flight knows it was engaged whether or not the interception succeeded).
- Spot-where-it-fires: a team is revealed when the flight comes within range of any position carrying at least 5% of its peak concentration weight, spotting reveals that team's operating zone, and only that team is spotted.
- A team's engagement concentration stays on its own terrain class.
- Defender memory is a decaying average over the episode (decay 0.8), reset per episode. The trained defender's state is a w=2 track window plus a whole-mission seen-mask, with head columns for public terrain exposure, recency and spotted-team threat.
- Sites: 200 candidates per theatre, quota-sampled so class shares match the theatre, with farthest-point selection inside each class.

Pinned operating point for steps 2 to 5: narva, K=3 teams, concealed reach 0.85, range multiplier 0.7, hidden lethality 1.0 times the pinned table (no weapons knob turned). Absolute characteristics on narva (range_scale = 2.27 x 0.7) are open 5.6 km / 0.90, field 4.0 km / 0.85, forest 4.7 km / 0.55 (concealed), urban 3.3 km / 0.45 (concealed). Held out: kgd_gvardeysk, ukraine and fulda.

Models: llama-3.3-70b, and Qwen3.6-27B served under the alias `qwen3-27b`. All qwen calls are thinking-off unless a row is labelled thinking. Reporting is per model, never pooled.

## Criteria

- Screen: operating point = the cell with the largest G2 (best simple rule / exact optimum) subject to G2 >= 1.25, G1 (static cap / optimum) >= 2.0 and non-degenerate values. Operating-point filter: at least 90% of laydowns a real game, G1 >= 2, G2 >= 2, sight >= 1.4x, ranked by closeness of the omniscient hidden/open ratio to 1.00 against the size of the observing ratio.
- Step 2: per model, the model's forces induce more damage against a best-responding defender than the heuristic on at least 2/3 fields and pooled, and above the random arm's mean. Binding control: relabelling terrain in the brief must materially change the composition.
- Step 3: the llm-trained defender is below both control arms on at least 4/6 held-out cells and pooled, on at least 2/3 seeds, at the validation-selected checkpoint.
- Diagnostics: irreducible-threat bar 0.0215 (the tuned-doctrine curriculum's median damage against a defender that knows the laydown). Phase 1d, B1 free lanes fall, B2 reach 0.0215, B3 beat the heuristic force against trained defenders on all six held-out fields. Phase 1e, G grounding >= 80%, C1 >= 60% of the restricted ceiling, C2 beat a random slot choice. Phase 1f, S1 beat random search, S2 beat local search, S3 reach 0.0215, at a matched 96 exact evaluations. Thinking rider, T1 median >= 0.0107, T2 >= 60% of ceiling, T3 grounding >= 80%.
- Step 5: the llm16-trained defender is below the tuned control on at least 4/6 held-out cells and pooled, on at least 2/2 seeds, with a third seed added only if the arm ordering is ambiguous. Secondary and non-gating, the same clauses against local16 and random16.
- Step 5c: the act's primary clause applied to the qwenthink16 arm, plus a paired readout against llm16 and local16 with no superiority bar.
- Step 5e: all three qwen rolls transfer better than all three llama rolls (exact permutation p = 1/20 = 0.05).

## Baselines

- avoid-revealed rules: fly the menu route minimising damage from the currently known sites. The best member of this family under persistent memory is the best observing rule.
- Blind rules: payoff-blind anti-repeat and rotation over lanes and the full menu.
- Static caps: the static equilibrium mixture and the multi-start static local optimum.
- Exact optimum: best possible adaptive play against a fixed enemy rule (Karp minimum-mean-cycle over the window MDP, episodic solve at T=40).
- Oracle-searched ceiling: `choose_force` over the three archetypes with the single gen32 doctrine, scored exactly.
- Tuned-doctrine curriculum (`tuned`): positions from `choose_force` with the gen32 doctrine (0.6/0.2/0.3, tau 0.10, w 2).
- Search controls at a matched budget of 16 exact evaluations per field: `random16` (uniform triples) and `local16` (steepest-descent swaps).
- Every arm, trained or rule, receives the same observations, including the revealed-site channel.

## Results

### Screen

Third run, 2026-07-26, 12,960 cells, 36/36 blocks, both defender memories per cell. Medians over real cells, persistent arm.

| map | a real game | G1 cap/opt | G2 rules/opt |
|---|---|---|---|
| kgd_gvardeysk | 83% | 3.17 | 3.21 |
| ukraine | 67% | 4.28 | 3.25 |
| narva | 79% | 3.95 | 3.58 |
| fulda | 91% | 2.91 | 2.55 |

Gates (G1 >= 2, G2 >= 1.25) pass on 86% of real cells, that is 69% of the grid. Sight is worth 1.29x to the defender against open laydowns, 1.37x against mixed, 1.26x against random, and exactly 1.00x against concealed ones (same map, same rules, only the ground changes). Whole-mission memory improves the observing defender 0.89x against the forgetful window, with optimum and blind rules unmoved (0.99-1.00x).

At the pinned cell (narva, K=3, concealed reach 0.85, range multiplier 0.7, hidden lethality 1.0 times the pinned table), 92% of laydowns are a real game, G1 is 3.65, G2 is 4.36, sight is worth 2.08x to the defender, and hidden vs open is 1.07 against an omniscient defender (firepower matched by construction) and 2.66 against one that must observe.

Cost of concealment, as the share of an open force's damage achieved by a concealed force against an observing defender at the pinned table:

| map | cover share | K=3 | K=6 | options control |
|---|---|---|---|---|
| kgd_gvardeysk | 26% | 57% | 52% | 77% / 87% |
| ukraine | 12% | 82% | 69% | 113% / 112% |
| narva | 64% | 80% | 91% | 100% / 100% |
| fulda | 66% | 121% | 91% | 100% / 100% |

### Step 2 (composition)

32/32 live calls valid, fields 5100-5102, one placer for every arm, doctrine and roles from the model only.

| arm | n | vs observing defender (primary) | vs omniscient | coverage |
|---|---|---|---|---|
| oracle-searched ceiling | 3 | 0.0964 | 0.0217 | 1.00 |
| llm: llama-3.3-70b | 8 | 0.0747 | 0.0018 | 0.96 |
| llm: qwen3-27b | 8 | 0.0613 | 0.0006 | 0.96 |
| heuristic (gen32 doctrine) | 1 | 0.0603 | 0.0191 | 1.00 |
| random | 20 | 0.0123 | 0.0001 | 0.77 |
| relabel control: llama | 8 | 0.0059 | 0.0005 | 0.85 |
| relabel control: qwen | 8 | 0.0057 | 0.0004 | 0.85 |

llama-3.3-70b passes every clause (2/3 fields, pooled, above the random mean). qwen3-27b passes pooled (0.0613 against 0.0603, thin) and above random, and fails the per-field clause at 1/3 fields, so it is partial. The relabel control passes for both models: under the swapped brief the forest share moves from 54% to 33% for llama (open 13% to 33%) and from 71% to 42% for qwen, and the relabelled forces, resolved on the true table, score 0.0059 and 0.0057.

### Step 3 (curriculum)

12 runs, 3 curricula x 3 seeds plus a blinded llm arm, 5000 sorties, validation-selected checkpoints, held-out fields 6100-6105.

| arm (curriculum) | seed 0 | seed 1 | seed 2 | pooled |
|---|---|---|---|---|
| heuristic (single gen32 doctrine) | 0.0872 | 0.0985 | 0.0888 | 0.0915 |
| llm-composed population | 0.0878 | 0.1166 | 0.1194 | 0.1079 |
| llm-composed, blinded | 0.1232 | 0.1135 | 0.1034 | 0.1134 |
| random compositions | 0.1619 | 0.1271 | 0.1672 | 0.1520 |

Criterion outcome 0/3 seeds. The llm arm beats the random-composition control on 6/6, 4/6 and 5/6 cells (pooled 0.108 against 0.152) and is beaten by the heuristic control on 5/6, 6/6 and 6/6. Blinded arm by enemy type (pooled 0.1079 sighted against 0.1134 blind):

| held-out enemy | sighted | blinded | channel |
|---|---|---|---|
| oracle(open), a revealing force | 0.2412 | 0.2267 | 0.94x |
| oracle(hidden), a concealed force | 0.1370 | 0.1779 | 1.30x |
| llm / random / heuristic forces, mixed ground | 0.0751 | 0.0810 | 1.08x |

### Diagnostic phases (corrected brief)

Curriculum value against the opponent's irreducible threat, that is its median damage against a defender that already knows the laydown:

| curriculum | opponent vs perfect play | resulting defender (held-out, pooled) |
|---|---|---|
| heuristic | 0.0215 | 0.0915 |
| llm | 0.0007 | 0.1079 |
| random | 0.0000 | 0.1520 |

Phase 1c. Robust arm median irreducible threat 0.00194, that is 9% of the 0.0215 bar, best single force 0.0126 (59% of the bar), cover share 0.67. Iterative rounds run 9%, 13%, 3% of the bar with no trend. No arm reaches the bar.

Phase 1d. Grounding medians 11-18% per round, overall median about 12%. Irreducible threat moves from 6% to 11% of the bar over six rounds. Free lanes fall from 6.0 to 0.0 (B1 pattern). B3 fails, with evolved forces pooled 0.0829 against the heuristic force's 0.1650 (0.50x), round-0 forces 0.0699, and better on 3/6 fields.

Phase 1e. Exhaustive ceiling over all 165 three-slot combinations 0.0278; the median slot combination, equivalent to choosing at random, 0.0055. Thinking-off medians are 17% of ceiling for llama (best 36%) and 19% for qwen (best 36%); the thinking rider reads 29% median and 42% best. C1 (>= 60%) fails in both modes, C2 passes (best 0.0099-0.0115 against the random draw 0.0055, 1.8-2.1x), rider grounding 100%, rider T1 median 0.0082 against the 0.0107 bar and T2 29%, both failing. Urban slot uptake is 8/24 forces on the corrected brief against 0/24 on the defective one, though urban sits in the ceiling-defining optimal combination.

Phase 1f, sample efficiency in the full space (200 sites, 1,313,400 three-team forces), every arm on the same exact-evaluation budget with the same fixed doctrine. Best force found:

| arm | @8 | @16 | @32 | @48 | final (96) |
|---|---|---|---|---|---|
| random triples | 0.0110 | 0.0267 | 0.0385 | 0.0385 | 0.0385 |
| greedy top-K by site threat | 0.0374 | 0.0394 | 0.0394 | 0.0394 | 0.0404 |
| local search (seed + steepest-descent swaps) | 0.0345 | 0.0345 | 0.0410 | 0.0410 | 0.0485 |
| llm: llama-3.3-70b | 0.0394 | 0.0408 | 0.0408 | 0.0408 | 0.0433 |
| llm: qwen3-27b | 0.0309 | 0.0316 | 0.0426 | 0.0426 | 0.0426 |

S1 passes (0.0433 against 0.0385), S3 passes (2.0x the 0.0215 bar), S2 fails (local search reaches 0.0485). At 8 evaluations llama is at 0.0394 where local search is at 0.0345 and random at 0.0110.

### Step 5 (matched-budget curricula, n=3)

Positions only, doctrine frozen to gen32 in every arm, 16 exact evaluations per field, narva, 5000 sorties, validation-selected.

| arm | seed 0 | seed 1 | seed 2 | pooled +/- sd |
|---|---|---|---|---|
| llm16 (llama-proposed, 16 evals) | 0.1302 | 0.1417 | 0.1145 | 0.1288 +/- 0.0111 |
| local16 (hill-climb, 16 evals) | 0.1298 | 0.1198 | 0.1564 | 0.1353 +/- 0.0155 |
| random16 | 0.1597 | 0.1844 | 0.1421 | 0.1621 +/- 0.0173 |
| tuned (step-3 control) | 0.1717 | 0.1824 | 0.1490 | 0.1677 +/- 0.0139 |

The primary criterion is met on 3/3 seeds against the tuned control, 23% better pooled. Against random16 the clauses pass 3/3 seeds (6/6, 6/6, 4/6 cells). Against local16 they pass on 1/3 seeds, and the paired difference is -0.0066 +/- 0.0265, so llm16 and local16 are statistically indistinguishable at n=3 and no ordering between them is claimed. Curriculum strengths: llm16 0.0393, local16 0.0222, random16 0.0286, tuned 0.0278.

### Zero-shot

The twelve narva-trained checkpoints on three unseen theatres, each with its own strong test set built by the same 16-evaluation recipe. The harness self-check reproduces the narva step-5 cells to 0.00000.

| map (unseen) | llm16 | local16 | random16 | tuned | llm16 beats tuned | beats local16 |
|---|---|---|---|---|---|---|
| kgd_gvardeysk | 0.3522 | 0.3688 | 0.3720 | 0.4117 | 3/3 seeds | 3/3 seeds |
| ukraine | 0.2192 | 0.2390 | 0.2351 | 0.2654 | 3/3 seeds | 3/3 seeds |
| fulda | 0.1042 | 0.1108 | 0.1160 | 0.1099 | 3/3 seeds | 2/3 seeds |

The same twelve defenders on the fresh laydown-saved test sets (self-check reproduces the step-5 narva cells to 0.00000), mean over 3 seeds:

| map | llm16 | local16 | random16 | tuned |
|---|---|---|---|---|
| kgd_gvardeysk | 0.3837 | 0.4028 | 0.4002 | 0.4403 |
| ukraine | 0.2140 | 0.2327 | 0.2311 | 0.2620 |
| fulda | 0.1182 | 0.1271 | 0.1306 | 0.1223 |

On the fresh sets llm16 beats local16 on 8/9 map-seed pairs, random16 on 7/9 and tuned on 8/9. Oracle boundary for the whole act, stated once: no trained arm beat the best observing rule on any cell of any map (0/6 everywhere), in step 3, at step 5 n=3, and on all three unseen theatres.

### Steps 5c-5e (authoring grid)

Step 5c, the qwenthink16 arm (Qwen3.6-27B with thinking on), 3 seeds, everything else frozen to step 5.

| arm | curriculum strength | defender (narva, pooled) | seed spread |
|---|---|---|---|
| qwenthink16 (Qwen3.6-27B, thinking on) | 0.0377 | 0.1132 | +/- 0.0004 |
| llm16 (llama-3.3-70b) | 0.0386 | 0.1288 | +/- 0.0137 |
| local16 (hill-climb) | 0.0217 | 0.1353 | +/- 0.0189 |
| random16 | 0.0286 | 0.1621 | +/- 0.0212 |
| tuned (step-3 control) | 0.0267 | 0.1677 | +/- 0.0171 |

The primary criterion is met on 3/3 seeds (6/6, 5/6, 5/6 cells; paired -0.0545, paired-t p 0.032). In distribution the paired readout is suggestive only (vs llm16 -0.0155, p 0.195, cells won 12/18; vs local16 -0.0221, p 0.175). On the fresh laydown-saved sets qwenthink16 beats llm16, local16 and tuned on 9/9 map-seed pairs each, and random16 on 9/9.

| map | qwenthink16 | llm16 | local16 | random16 | tuned |
|---|---|---|---|---|---|
| fulda | 0.1026 | 0.1182 | 0.1271 | 0.1306 | 0.1223 |
| kgd_gvardeysk | 0.3643 | 0.3837 | 0.4028 | 0.4002 | 0.4403 |
| ukraine | 0.1826 | 0.2140 | 0.2327 | 0.2311 | 0.2620 |

Step 5d, a second qwenthink curriculum over the same 16 narva training fields with only the search rng changed:

| curriculum | fields | strength | distinct sites | Jaccard | reuse |
|---|---|---|---|---|---|
| qwenthink 1 (step 5c, trained) | 16 | 0.0390 | 18 | 0.153 | 62.5% |
| qwenthink 2 (new, same author, fresh search) | 16 | 0.0387 | 16 | 0.187 | 66.7% |
| llama llm16 (reference) | 16 | 0.0393 | 12 | 0.321 | 75.0% |

The two qwenthink curricula share zero of their 16 per-field top laydowns, and 5 laydowns across the full kept sets (48 each, top-3 per field). All 5d digits reproduce from the committed curricula under the pinned metric.

Step 5e, the authoring grid (2 authors x 3 rolls x 3 seeds, all scored on the same saved fresh sets). Transfer is the mean over the 9 fresh map-seed cells, lower is better.

| roll | transfer | narva pooled | strength | Jaccard / distinct |
|---|---|---|---|---|
| qwen r3 | 0.2076 | 0.1158 | 0.0381 | 0.158 / 17 |
| qwen r2 | 0.2147 | 0.1183 | 0.0387 | 0.187 / 16 |
| qwen r1 | 0.2165 | 0.1132 | 0.0390 | 0.153 / 18 |
| llama r2 | 0.2356 | 0.1209 | 0.0389 | 0.209 / 16 |
| llama r1 | 0.2386 | 0.1288 | 0.0393 | 0.321 / 12 |
| llama r3 | 0.2479 | 0.1329 | 0.0383 | 0.406 / 12 |

The primary passes at the locked bar, with author means 0.2129 +/- 0.0039 against 0.2407 +/- 0.0052, complete rank separation, exact permutation p 0.05. The narva column separates by rank as well (maximum qwen 0.1183 below minimum llama 0.1209). Curriculum strength spans 0.0381-0.0393 across the whole grid. Reported mediation row: the Spearman correlation of curriculum Jaccard against transfer loss over the six rolls is rho 0.83 (nominal p 0.042). The step-5c seed spread of +/- 0.0004 did not recur on any new roll (spreads 0.0058-0.0075).

### Thinking-mode rows

The step-2 protocol rescored under the step-2 aggregation, with the gen42 crown arms as fresh draws. All six consistency anchors reproduce the step-2 table before any new number was read.

| arm (step-2 aggregation, vs the best observing defender) | n | pooled | per-field | vs heuristic 0.0603 | relabel collapse |
|---|---|---|---|---|---|
| step-2 qwen, off (2026-07-26 draw) | 8 | 0.0613 | 0.0683 / 0.0643 / 0.0330 | pooled pass (thin), 1/3 fields | 10.8x |
| gen42 crown off (fresh draw) | 16 | 0.0343 | 0.0421 / 0.0430 / 0.0185 | pooled fail, 0/3 fields | 7.8x |
| gen42 crown on (thinking; 12/16 valid) | 12 | 0.0211 | 0.0113 / 0.0402 / 0.0059 | pooled fail, 0/3 fields | 1.8x |

Both gen42 arms stay above the random mean of 0.0179. The off-to-on gap (0.0343 against 0.0211) is smaller than the same-configuration draw-to-draw gap (0.0613 against 0.0343). The step-2 qwen partial does not replicate on the fresh draw at larger n; the step-2 llama result (0.0747, every clause) was not re-drawn.
