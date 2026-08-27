# HANDOVER.md: the master state of the SACRED project

> Rewritten from scratch on 2026-08-04, replacing the 727-line reverse-chronological banner
> stack that had accumulated since 2026-07-07. The old file is preserved verbatim at
> `docs/archive/HANDOVER_STACK_2026-07-28.md`. Nothing scientific changed in the rewrite. Every
> claim below carries its ledger pointer, and the ledgers remain the sole source of citable
> numbers. Read `CLAUDE.md` first if you have not (identity, house rules, project map).

## The state in one paragraph (2026-08-10)

The experimental campaign is COMPLETE across all three worktrees (39 chronicle entries,
gen01-gen45; the gen40-gen45 arcs of 2026-08-04/10 were post-freeze follow-ups on Kilian's
instruction: the doctrine-head transfer positive (gen41), the gen43 consolidation of the
thesis's Act 2 onto one instrument, the LLM capability-measurement chain (gen42, the aerial
gen43 exam, gen44) with the gen39 author-level transfer arc (steps 5c-5e), and the gen45
unification of Acts 4 and 5 onto one game). Nothing is
running, no launch decision is pending, and all three trees are clean and committed. The Final Activities Report deadline (30 July) and the self-imposed experimental
freeze (3 August) are past. **Writing is the critical path.** The thesis and poster are due
10:00, Friday 28 August 2026 (12,000 words excluding appendices). In the thesis repo the
Methodology chapter and its proofs appendix are drafted and being Kilianised, the Results
chapter is substantially drafted on the five-act structure agreed 2026-08-03 (Acts 2 and 4
now rebuild from the gen43 and gen45 consolidation ledgers respectively), and the
introduction, literature, conclusions and abstract are the thin remaining ends.

## Deadlines, rubric and objectives

- **Thesis + poster due 10:00, Friday 28 August 2026.** Maximum 12,000 words excluding
  appendices and references. The thesis is 70% of the project mark, the poster 5% (the assessed
  literature review was the other 25%).
- **Rubric weights.** Methodology, Analysis & Discussion 50%; Structure & Presentation 20%;
  Abstract 10%; Introduction & Problem Definition 10%; Conclusions 10%. The conclusions chapter
  must revisit each objective and confirm whether it was met.
- **AI acknowledgement is mandatory** (tool name and version, publisher, description of use,
  confirmation the work is Kilian's own). Fabricated or unverified references are treated as
  plagiarism. Full detail in the guidance PDF (`../../MSc Transport - Research Project Guidance
  2025-2026.pdf`).
- **The objectives were reworked to FOUR on 2026-07-27** (supervisor approved): ERB dropped,
  the SBO objective dropped, the LLM work moved into Obj-3, the mission-control application
  into Obj-2. The assessed survey's original five-objective wording is historical context, not
  the current contract.
- **Game framing is Nash, not Stackelberg**, everywhere in the thesis (binding decision,
  2026-07-31, with the commitment-coincidence remark carrying the equivalence).

## The thesis argument in one paragraph (the boundary map)

The thesis asks where learning pays in contested routing, and answers it with a measured map.
Below a measurable boundary it does not pay, because a two-line rule is near-optimal wherever
it has somewhere safe to go. This is proven three ways, by the negative campaign (gen03-06,
congestion adversaries have a flat attack landscape), by the disjoint-baseline finding (a
max-flow stack matches every trained static K=1 number), and by the aerial baseline-completeness
appendix. Learning pays where the cheap escape is closed, at interdiction budgets approaching
the min-cut (gen26, gen35, gen43) and against adaptive pattern-of-life adversaries (gen19,
gen27, gen31, and gen45, which rebuilt gen32's real-terrain positive on the unified game),
always scored against computable optima. Above both sits a harder wall,
conditioning capacity, where a channel is present and visible yet the policy cannot convert it
(gen29 coordination, gen34 hidden type, gen39 exposure). The LLM arc adds a measured map of
where a language model helps this pipeline (doctrine identification from prose, terrain-grounded
force composition, sample-efficient curriculum authoring) and where it is useless to harmful
(numeric mixtures, combinatorial curation), now with a measured capability axis (scale helps
cumulatively, deliberation changes answers rather than scores, and the thinking reasoner's
more varied curricula train the defenders that transfer best).

## The claims register

Citable numbers live only in the ledgers named here. Worktree prefixes: (R) roads `sacred`,
(A) `sacred-aerial`, (M) `sacred-gen29`. Ledgers shared across worktrees were reconciled to one
canonical version on 2026-08-04 and are byte-identical; if you amend a shared ledger, mirror
the change to the other trees in the same session.

### Banked positives

| claim | ledger |
|---|---|
| **gen27, the flagship.** One history-aware policy, trained on three cities, zero-shot on Gdansk beats the static cap (PRIMARY + STRONG 3/3, pooled 0.639x cap); no-window control causal. Corrected exact-optimum ratio 1.97x (yardstick-repair appendix) | (R) `experiments/gen27_dynamic_generalist.md` |
| **gen35.** Dynamic K-boundary on 71-33 m=6; K=3 beats EVERY two-line rule 3/3 seeds (-8.6%), K=2 ties; the programme's first pre-registered beats-every-naive-rule cell; tabular window-Q at matched budget fails | (R) `experiments/gen35_dyn_kboundary.md` |
| **gen43 (2026-08-08/09).** The consolidated Act-2 instrument: 71-33, both registers, one K-axis. Dynamic: SACRED beats EVERY two-line rule at K=3,4,5,6 (3/3 seeds each; margins -8.6%/-15.4%/-20.7%/-19.9%; slack collected 26->57%; 1.24x the exact optimum at K=5/6, the programme's closest; matched-budget window-Q fails outright at K<=4 and collects at most a fifth of the slack at K=5/6), the ENTIRE computable range past the K=2 tie, ending at the measured K=7/8 memory-and-cost wall (heuristic proxies barred by gen40 tier E). Static: SACRED tracks the naive frontier from behind to a tie as mixing itself dies at K~8-9 (det optimal from K=9); exact v* to K=4, inv-vuln stack exactly optimal at K=1. Reuse licence + the bit-replay-never-existed finding. The thesis's Act-2 tables rebuild from this ledger | (R) `experiments/gen43_unified_kboundary.md` |
| **gen41 Act 3 (2026-08-08).** The doctrine-head transfer positive under the three-tier fairness ladder: zero-shot on Gdansk at K=2, PRIMARY PASS 2/3 (pooled 0.783x cap; beats the cap 6/6 on every seed, every map-only rule 3/3, every outcome-earning adaptive learner 2/3, seed 0 missing by 0.0005); causal control 1.161 at 0/6; the 0.943 -> 0.783 one-flag encoder ablation against Act 2 is the mechanism claim; STRONG FAIL disclosed (mechanism-told rules at 0.656 stay ahead) | (R) `experiments/gen41_deepwindow_zst.md` |
| **gen19.** Single-instance pattern-of-life register; history-aware SACRED far below the static cap with causal no-window control (its STRONG "reaches the optimum" clause retired by the yardstick repair; rotation attains the exact optimum on m=4) | (R) `experiments/gen19_b1lite1.md` |
| **gen26.** The K-to-min-cut boundary map; SACRED beats both max-flow heuristics at K=m-1 exact; ties the best naive full-menu stack at K=m; tabular FP with the same greedy oracle also works, so the licensed sentence is "learning is required, deep RL is one sufficient method" | (R) `experiments/gen26_kboundary.md` |
| **gen31 + gen32.** The aerial dynamic positives, synthetic then REAL Kaliningrad terrain; 6/6 beats-static-cap on 3/3 fresh seeds each, blinded controls causal, one attempt each; gen32 pooled 0.451x cap, 1.30x the exact optimum; gen32 SUPERSEDED for the thesis's Act 4 by gen45 (2026-08-10), its numbers never tabulated beside gen45's | (A) `experiments/gen31_aerial_dyn.md`, `gen32_theatre_dyn.md` |
| **gen45 (2026-08-09/10).** The unified corridor game: Acts 4 and 5 rebuilt on ONE substrate (gen39 terrain v2, quota sites, multiplier field; one enemy model whose only dial is relocation range, the gen32 "searchlight" proven its flat limit). The Act-4 positive reproduced in one attempt on the pristine gated set: PRIMARY 18/18 beats-cap (pooled 0.351x), STRONG 1.46x the exact optimum, blinded control causal (1.242x at 0/6, information weights 0.000000); the whole payoff-blind family beaten 18/18; fitted doctrine-informed rules ~1.2x ahead disclosed; worst-case committing 1.52x; the doctrine column, not anti-repeat, carries the result; verified digit-for-digit | (A) `experiments/gen45_unified_corridor.md` |
| **gen39.** The concealment mechanic with an internal control (sight worth 1.26-1.37x vs exactly 1.00x concealed); the LLM composition positive with the terrain-relabel control PASSING (the LLM arc's first licensed terrain-reasoning claim); the curriculum negative diagnosed (irreducible threat) and FIXED at step 5 (llm16 23% better than the tuned control 3/3, zero-shot to three unseen theatres); extended 2026-08-06/10 by the brief repair (1c/1d/1e citable only from corrected artefacts) and steps 5c-5e (own row below) | (A) `experiments/gen39_concealment.md` |
| **gen39 steps 5c-5e (2026-08-08/10).** The author-level transfer effect: at a matched 16-evaluation budget and matched curriculum strength (~3% grid-wide), curricula authored by Qwen3.6-27B thinking-on train defenders that transfer better to unseen theatres than llama's on ALL rolls (2 authors x 3 rolls x 3 seeds, complete rank separation, exact permutation p 0.05); curriculum DIVERSITY is the supported carrier candidate (Jaccard-vs-transfer rho 0.83 over six rolls; authors have reproducible style without reproducible choices); in-distribution suggestive only; per-model as always | (A) `experiments/gen39_concealment.md` (steps 5c-5e) |
| **gen38.** LLM enemy identification; V1 reads intel prose, classifies five doctrines at 100%, crosses the gen34 type-blind wall 6/6; V2 trained type-conditioned SACRED crosses 3/3 seeds at 0.664x the blind cap, LLM-supplied type indistinguishable from truth | (R) `experiments/gen38_llm_enemy_id.md` |
| **gen30.** Security-aware facility location, oracle-only; the (cost, security) depot frontier; redundancy-coordination complementarity (median 25% at the coordinated optimum, negative under napkin play); surrogate Spearman 0.870 | (R) `experiments/gen30_secure_flp.md` |
| **gen16 + gen15.** The static ZST arc (cross-city transfer with causal controls); wording bound by the disjoint-baseline rules below | (R) `experiments/gen16_multicity.md`, `gen15_generalist.md` |
| **gen20.** A learned interdictor co-evolves to 0.81x the oracle's strength (the campaign reversal) | (R) `experiments/gen20_f2_learned.md` |
| **The negative campaign.** Adversarial co-training against congestion confers no robustness and worsens it, with a complete mechanism chain; the motivating negative of the whole thesis | (R) `experiments/gen03_robustness_dynassign.md` through `gen06_dynassign_matrix.md`, `gen07_contested_matrix.md` |

### Boundaries and negatives (equally load-bearing)

| finding | record |
|---|---|
| **The disjoint-baseline finding.** A two-line max-flow stack matches or beats every trained static K=1 number, including zero-shot at 1.13x eq; the old "uniform" anchors were padded-menu strawmen | `docs/archive/CRITIQUE_16-07-26.md` + R0a appendices in seven (R) ledgers |
| **The coordination wall.** The three-stream oracle moat is real (median 31% vs the fitted cap over 55 cells) and survives a complete hostile family; the trained half failed both tiers with the blinded control equal to sighted; gen36 separated the mechanism as CONDITIONING CAPACITY, not training dynamics | (M) `experiments/gen29_multiod.md`, `gen36_multiod_rescue.md` |
| **The hidden-type wall.** No type inference from realised-attack observations (pooled 1.373x the type-blind cap, 0/18); the channel is causally useful short of inference | (R) `experiments/gen34_hidden_adversary.md` |
| **FP transience + coordination.** The equilibrium is a reproducible transient of last-iterate smooth FP (best-checkpoint discipline is the resolution); independent followers never learn to stack | (R) `experiments/gen17_lastiterate.md`, `gen18_learnedfollower.md` |
| **Aerial static register.** The fleet Tier-1 "positive" is RETIRED (tie with the naive frontier); the surviving zero-shot claim is payoff-blind-frontier MATCHING only, and the permuted-field row must run before any "on sight" sentence | (A) `experiments/gen28_aerial.md` (2026-07-19 appendix first) |
| **LLM negatives.** Neither open-weight model calibrates a mixed strategy (B2, the knowledge/application dissociation); gen33's terrain control failed; gen37 route curation is worse than random at every prune size | (R) `experiments/b2_llm_benchmark.md`; (A) `gen33_llm_adversary.md`; (M) `gen37_reasoning_curation.md` |
| **The capability axis (gen42, the aerial gen43 exam, gen44; all eval-only).** Size helps CUMULATIVELY (2B->27B +0.300 share of ceiling, 4B->27B +0.134, CIs excluding zero; no single step separates); generation and thinking unresolved with tight CIs (thinking changes the answer on 23/40 items and moves the score about as much as re-rolling the sampling seed); llama-3.3-70b measurably BELOW the 2026 27Bs; with repeats, every LLM author beats hill-climbing at the CURRICULUM level at every budget while the defender-level tie stands; gen42's B-COMP is underpowered at n=16 (no between-model composition sentence) | (A) `experiments/gen42_capability_ladder.md`, `gen43_exam.md`, `gen44_budget_sweep.md` |
| **gen39 boundary.** No arm beats the best simple OBSERVING rule on any cell of any of the four maps | (A) `experiments/gen39_concealment.md` |
| **Yardstick repair.** The roads dynamic-optimum yardstick (`oracle_refs` undamped RVI) was wrong on every cell tested; exact truth is Karp min-mean-cycle, `scratch/dyn_exact.py`; corrected appendices in the gen19/gen27 ledgers | (R) chronicle entry 31 |
| **The gen40 structure laws + the gen41 boundaries.** Rotation exactly optimal at w = m-1; rules fail deepest at w a multiple of m; rule failure grows with K/m to the exact wall (~K=4-5), past which NO computable adversary preserves the game; at w = 2m nothing tested collects the (real, 0.58x-cap) optimum, a channel-content boundary; at transfer the full net's encoder is the calibration bottleneck (Act 2 FAIL 0/3, fixed by the Act-3 mask) | (R) `experiments/gen40_dyn_sensitivity.md`, `gen41_deepwindow_zst.md`; chronicle entry 35 |

### The where-LLMs-help map (per-model, never pooled)

Useless to harmful at the quantitative registers (numeric mixtures B2, terrain-grounded
composition gen33, combinatorial curation gen37). Decisively valuable at the language-to-decision
register (gen38 doctrine ID) and at composition once the mechanics were real (gen39 step 2,
terrain control passing). Its distinctive advantage is SAMPLE EFFICIENCY at small evaluation
budgets (leads every method at 8-16 evaluations in a 1.3M-force space, overtaken by
hill-climbing by 96 at the DEFENDER level; at the CURRICULUM level gen44's repeats show every
LLM author beating hill-climbing at every budget, and the two levels are never elided). The
capability axis is measured (gen42, the aerial gen43 exam, gen44): performance climbs
cumulatively with size, generation and deliberation move nothing on score (thinking changes
answers, not outcomes, within the seed-re-roll noise floor), and a 2024 70B sits below the
2026 27Bs. The strongest positive shape is the author-level transfer effect (gen39 steps
5c-5e): thinking-authored curricula, matched in strength, train better-transferring
defenders, with diversity the supported carrier. The models still reverse between tasks
(llama leads composition; the 1e slot-task reversal was RETIRED by the 2026-08-06 brief
repair, near parity). Synthesis table: `experiments/regime_decision_table.md`.

## Binding wording rules (consolidated; each earned by a named finding)

1. No comparative claim without the complete baseline family beside it; the simple-rule
   concession is stated FIRST, on our terms.
2. Never "only self-play can train there"; the licensed form is "only best-response-oracle
   methods" (tabular FP with the same oracle also works).
3. Claims are per-register and regime-conditional; the regime decision table is the map.
4. LLM claims are per-model, never pooled; no "the LLM curriculum is best" (llm16 and local16
   are indistinguishable in-distribution); no "trained policy beats the rules" sentence from
   gen39.
5. Never mix game versions, pre-fix and post-fix ladders, or differently-sampled variants in
   one figure or table.
6. ZST wording: adversarial training's transfer value is label-free and self-stopping, NOT
   superior transfer (distillation with validation stopping beats the generalist); the hedge is
   "geometry-informed and threat-robust", per-edge map-reading is not the mechanism; gap
   closure decays 0.90 to 0.04 across transfer distance and any transfer figure shows it.
7. ALNS is never called "SOTA" unqualified; its defence is that it provably attains loss_det,
   the optimum of the whole deterministic class.
8. Dynamic optimum yardsticks come from `scratch/dyn_exact.py` (Karp), never from the old
   `oracle_refs` history_opt.
9. Nash framing throughout the thesis; no Stackelberg anywhere.
10. n=3 results say "mean +/- population std, per-seed values shown"; no significance language
    from pooled dependent cells.
11. Act-4 real-corridor numbers come from gen45; gen32 is superseded and its numbers never
    sit in a table beside gen45's (prose may say the claim SHAPE reproduces, each number
    attributed to its own game). The gen45 mechanism sentence is the doctrine/threat column,
    not anti-repeat (the recency channel is largely inert there).
12. Curriculum-level and defender-level claims are stated separately (gen44: hill-climb ties
    the LLM at the defender level yet is clearly worse at the curriculum level); the 5c-5e
    transfer sentence is author-level, per-model, with diversity "supported, not proven".
13. No thinking-mode score claim in either direction (deliberation changes answers, not
    scores, within the seed-re-roll noise floor); no between-model composition sentence from
    gen42's B-COMP (underpowered at n=16); the gen39 1c/1d/1e diagnostic chain is citable
    only from the corrected-brief artefacts (the 2026-08-06 repair).

## Open threads (all small, none blocking writing)

1. **gen39 validation-cache rebuild** (cheap): step 5's validation cache is inherited from step
   3 and built from tuned-family enemies; a rebuild would tighten every step-5 number.
2. **The aerial permuted-field row** (eval-only, minutes): must run before the thesis cites the
   aerial amortisation sentence.
3. **The 5d replication training + the diversity-manipulation test** (aerial): the one-command
   training of the second qwenthink curriculum's defenders, and the designed test (author at
   matched strength with deliberately varied diversity) that would turn the diversity carrier
   from supported into proven; both recorded, both need Kilian's go.
4. **gen37 restriction follow-up**: recorded future work only.
5. **imperial-sacred** carries uncommitted Story/UX work dated 2026-08-04; Kilian has parked it
   ("done for now").

## The project map

Three sacred worktrees (see `CLAUDE.md` for the table): roads `gen08-interdiction` (this tree,
master docs, suite last recorded 171; the "224" previously recorded here was a doc slip, the
aerial-era count), aerial `gen28-aerial` (suite last recorded 246), multi-OD
`gen29-multiod` (closed, suite last recorded 173). Branch `gen07-contested` is frozen history
at the gen07 close. All of `gen08-interdiction`, `gen28-aerial`, `gen29-multiod` and `public`
are pushed to `origin` (github.com/Kilian-S/sacred). A fourth worktree, `code/sacred-public`
(branch `public`, built 2026-08-21/22), is the reader-facing layout named in thesis Appendix A:
`roads/` and `aerial/` arms with `src`, `scripts`, `analysis` (the former `scratch`), `tests`,
`experiments` and `data/maps`, GPL-3.0, a README in Kilian's own prose, pinned requirements and
a `pytest.ini` per arm. GitHub's default branch `main` is kept equal to `public` by
fast-forward; the dev branches carry the full history and the untrimmed ledgers.

**imperial-sacred** (`code/imperial-sacred`, branch `expansion-gen26-39`, unpushed) is the
shareable restructured repo: engine package `sacred/`, historical harnesses `training/`, the
Mission Control web app (`api/` + `web/`), record synced to gen39 with Theatre, Playbook, Basing
and LLM surfaces. Entry points `README.md`, `AGENTS.md`, `docs/notes/HANDOVER.md`. Its anchor
tests (0.699/0.206 on 35-159) must never move.

**The thesis repo** (`Thesis/thesis/`, Overleaf-synced) is where all writing happens; this repo
is its read-only evidence base. Its own briefs are `THESIS_FRAME.md` (the spine) and
`THESIS_PLANNER_HANDOFF.md`. Current chapter state is summarised in the state paragraph above;
`SESSION_REPORT_methodology_2026-07-31.md` records the methodology draft.

**External references.** The guidance PDF and the assessed literature survey sit two directories
above the repo root (`../../`). The LLM workbench (Prof Angeloudis's box `cv-iits-w05`) is
reachable over Tailscale through an OpenAI-compatible gateway on port 8080, models
`llama-3.3-70b` and `qwen3-27b`. The endpoint, gateway key and SSH credentials are held by
Kilian and were scrubbed from every tree on 2026-08-27 (the public-branch scripts read
`SACRED_LLM_BASE` and `SACRED_LLM_KEY` from the environment); every call is audit-logged on
the box. Any use of it needs Kilian's go (shared hardware).

## Machine and operations facts (measured, not guessed)

M4 Mac, 10 cores (4P + 6E), 24 GB RAM. Training is CPU-locked (MPS 2.4-4x slower, settled).
4 torch threads solo; cap ALL thread pools on multi-process launches (`OMP_NUM_THREADS=1
VECLIB_MAXIMUM_THREADS=1` plus torch caps). Do not `nice` training runs (3x efficiency-core
penalty). RAM before cores (a duplicate-graph replay bug once drove the machine into memory
compression). Aerial trainer ~1.83 s/flight solo, ~3.1 GB per run, four concurrent is the safe
shape. Kill by explicit PID with a self-excluding pattern, never bare `pkill -f`, and verify
over 30 seconds. Run everything through `.venv/bin/python` with `PYTHONPATH=.`.

## The archive

Everything that used to clutter the top level lives in `docs/archive/` with one-line
descriptions in `docs/archive/INDEX.md`: the seven-critique series, the direction and redesign
documents, the superseded plans and checklists, the old banner-stack HANDOVER and SYSTEM, and
the historical build briefs. Older ledgers and chronicle entries reference these files by their
old top-level names; nothing in the archive is current guidance, but the critiques remain the
record of what each claim survived.
