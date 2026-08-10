# HANDOVER.md: the AERIAL worktree (branch `gen28-aerial`)

> Rewritten 2026-08-04 in the project-wide documentation refactor and updated 2026-08-10
> after the gen42-gen45 arcs and the gen39 extension steps landed. This file absorbs
> `HANDOVER_AERIAL_28-07-26.md` (now in `docs/archive/`, content preserved) and is the
> authoritative state of the aerial branch. **The master project state, the claims register and
> the consolidated wording rules live in `../sacred/HANDOVER.md`; read `CLAUDE.md` first for
> identity, house rules and the project map.** Citable numbers live only in the `experiments/`
> ledgers of this worktree.

## The state in one paragraph

The aerial campaign is COMPLETE through gen45 (2026-08-10). Nothing is running, the tree is
clean, and no decision is pending; the one recorded option awaiting Kilian's go is the
step-5d replication training (a single command, see future work). The branch holds nine
acts: gen28 (the free-flight negatives, the retired Tier-1 tie, and the binding
baseline-completeness appendix), gen31 (the aerial dynamic positive on the synthetic
lattice), gen32 (the same positive reproduced on the real Kaliningrad-Gvardeysk terrain, now
SUPERSEDED for the thesis's Act 4 by gen45), gen33 (the LLM red-force act, its terrain
control failing), gen39 (the concealment mechanic, the LLM composition positive, the
curriculum negative diagnosed then fixed, and, added 2026-08-06/10, the brief repair and the
step 5c-5e author-level transfer arc), gen42 (the capability ladder), gen43 (the
forty-question exam; the roads tree has a DIFFERENT gen43, the unified K-boundary), gen44
(the authoring budget sweep), and gen45 (the unified corridor game, Acts 4 and 5 on one
substrate).

## What each act banks (ledger pointers)

| act | verdict | ledger |
|---|---|---|
| gen28 | free-flight negatives (menu bandit, credit starvation, corridor collapse); Tier-1 fleet claim RETIRED as a tie by the 2026-07-19 appendix (read the appendix FIRST); surviving claim = payoff-blind frontier MATCHING; the permuted-field row must run before any "on sight" sentence | `experiments/gen28_aerial.md` |
| gen31 | the aerial trained positive, one attempt, blind-confirmed (6/6 beats-cap on 3/3 fresh seeds, blinded control causal) | `experiments/gen31_aerial_dyn.md` |
| gen32 | gen31 reproduced on REAL terrain (pooled 0.451x cap, 1.30x the exact optimum); **SUPERSEDED for the thesis's Act 4 by gen45 (2026-08-10)**: stays banked in its own ledger, its numbers never tabulated beside gen45's | `experiments/gen32_theatre_dyn.md` |
| gen33 | LLM red force; K=3 coordinated passes for both models but the binding scrambled-terrain control FAILS, so no terrain-grounding claim; metric 2 left open, superseded by gen39 | `experiments/gen33_llm_adversary.md` |
| gen39 | the concealment mechanic with an internal control; the LLM composition positive with the terrain control PASSING; the curriculum negative diagnosed (irreducible threat) and FIXED at step 5 with zero-shot transfer; EXTENDED 2026-08-06/10: the v1-brief defect repaired (1c/1d/1e citable only from corrected artefacts; the 1e slot-task reversal RETIRED) and steps 5c-5e (the author-level transfer effect, complete rank separation at exact p 0.05, diversity the supported carrier) | `experiments/gen39_concealment.md` (reader's summary at top; superseded blocks deliberately visible) |
| gen42 | the capability ladder (eval-only): endpoints separate (4B below 27B at the search-bound slot register), every finer contrast drowns in n=8 sampling noise; the two amendments are binding (B-COMP underpowered at n=16, no between-model composition sentence; only the staircase's endpoints survive bootstrap) | `experiments/gen42_capability_ladder.md` |
| gen43 (exam) | the forty-question placement exam, zero format failures on 7/7 configs: size helps CUMULATIVELY (2B->27B +0.300 share of ceiling, 4B->27B +0.134, CIs excluding zero; no single step separates); generation and thinking unresolved with tight CIs (thinking changes the answer on 23/40 items and moves the score about as much as a seed re-roll, the amendment's noise-floor rows); llama-3.3-70b measurably BELOW both 2026 27Bs; the 2B's gen42 format problem was harness-scoped | `experiments/gen43_exam.md` |
| gen44 | the authoring budget sweep with repeats (9 searches per config): 2B/4B SEPARATE from 27B as curriculum authors at usable budgets (correcting gen42's n=1 flat-band reading); every LLM author beats hill-climbing at the CURRICULUM level at every budget (the defender-level tie stands; the two levels are never elided); llama vs crown-thinking indistinguishable at budget 16, excluding curriculum strength as the 5c transfer carrier | `experiments/gen44_budget_sweep.md` |
| gen45 | the unified corridor game: the Act-4 positive reproduced in ONE attempt on the exact gen39 substrate (PRIMARY 18/18 beats-cap, pooled 0.351x; STRONG 1.46x the exact optimum; blinded control causal at 1.242x, 0/6; payoff-blind family beaten 18/18; fitted rules ~1.2x ahead disclosed; worst-case committing 1.52x); verified digit-for-digit; Acts 4 and 5 now share ONE game | `experiments/gen45_unified_corridor.md` |
| theatre atlas | the four scored theatres (kgd, ukraine, narva, fulda) | `experiments/theatre_atlas.md` |

## gen39 in brief

Step 1 passed (gates clear on 86% of real cells; sight worth 1.26-1.37x against a revealing
force and exactly 1.00x against a concealed one). Step 2 passed for llama on every clause
(0.0747 vs the tuned doctrine 0.0603 against the best simple defender; qwen partial), with the
relabel control collapsing both models' forces 10-13x, the LLM arc's first licensed
terrain-reasoning claim. Step 3 (training against LLM-composed enemies) FAILED 0/3; Phase 1
diagnosed curriculum value as tracking the enemy's IRREDUCIBLE THREAT; step 5 rebuilt every
curriculum with a matched 16-evaluation search and PASSED 3/3 (llm16 0.1288 vs the tuned
control 0.1677, 23% better), transferring zero-shot to all three unseen theatres. Phases 1c-1f
located the LLM's edge at grounding + sample efficiency (leads at 8-16 evaluations, overtaken
by hill-climbing by 96).

**The 2026-08-06/10 extensions.** The v1-brief defect (1c/1d/1e briefed the v1 physics table
while every scorer used v2) was repaired and re-run at pinned bars: "briefing is not the
constraint" softened (a truthful brief roughly doubles the median yet every arm stays 5-10x
short of the bar), grounding ~12% and unmoved by truthful physics, the 1e slot-task reversal
RETIRED, and "what remains is combinatorial search" re-established on corrected artefacts,
the only citable ones for that chain. Steps 5c-5e then produced the LLM arc's strongest
positive: the qwenthink16 arm PASSED primary 3/3 and its defender transferred better on 9/9
fresh map-seed pairs (the pre-registered null prediction failed, recorded made-and-wrong);
5d showed the author's diversity signature reproduces with zero shared top choices (style
without choices); 5e's grid (2 authors x 3 rolls x 3 seeds) rank-separated the authors
completely (qwen 0.2129 +/- 0.0039 vs llama 0.2407 +/- 0.0052, exact permutation p 0.05)
with the diversity mediation row firing (rho 0.83 over six rolls). Licensed: at matched
budget and matched strength the thinking-mode author's curricula train better-transferring
defenders, an author-level finding; diversity is the supported, not proven, carrier.

## gen45 in brief (the branch's most recent act)

The gen32 real-corridor positive rebuilt on the exact gen39 substrate (terrain v2 with reach
AND lethality, 200 quota non-grid sites, multiplier field 0.55-1.0, DOC32 enemy at w=2 tau
0.10), the enemy's full-map relocation proven to be the gen39 machinery's flat limit (anchor
3.9e-12), so Acts 4 and 5 differ only in how far a team may relocate between serials. Hunt
gates passed at the preferred pin (G1 min 3.71, G2 12/12); the attempt wave passed its gate
3/3; the confirmation wave on the pristine gated set passed every bar (18/18 beats-cap,
pooled 0.351x; 1.46x the exact optimum; blinded control 1.242x at 0/6 with information
weights 0.000000). Mechanism note for the write-up: the recency channel is largely INERT on
this game (a seed sweeps 6/6 with a positive recency weight); the doctrine column (~-20 on
every seed) carries the result. Fitted doctrine-informed rules stay ~1.2x ahead, so the
licensed sentence remains "beats every static object and every payoff-blind dynamic rule,
discovered unaided"; the worst-case committing premium is 1.52x pooled.

**Binding boundaries.** No gen39 arm beats the best simple OBSERVING rule on any cell of any
map, so no "trained policy beats the rules" sentence. llm16 and local16 are indistinguishable
in-distribution; the 5c-5e transfer sentence is author-level and per-model, with diversity
"supported, not proven". The models reverse between tasks at composition (llama leads); the
1e slot-task reversal is RETIRED (a brief-repair artefact). No thinking-mode score claim in
either direction (deliberation changes answers; scores move within the seed-re-roll noise
floor); no between-model composition sentence from gen42's B-COMP. Curriculum-level and
defender-level claims are stated separately (gen44). gen45 and gen32 numbers never share a
table (different games), and nothing pools across the gen39/gen42/gen43-exam instruments.
Never mix game versions (symmetric/asymmetric forest, leaked/masked concentration, grid/quota
sampling) in one figure.

## Code map (what is new on this branch)

`src/envs/aerial_sector.py` (gen28 lattice), `src/envs/aerial_theatre_vec.py` (vec-theatres,
terrain v2, quota sampler), `src/envs/aerial_theatre_env.py` (any vec-theatre SAC-trainable),
`src/envs/aerial_conceal.py` (gen39 concealment; carries the gen45 flat-limit regression
anchor), `src/redforce.py` (LLM briefs, `force_schema`); trainers `scripts/train_aerial_dyn31.py`,
`train_aerial_dyn32.py`, `train_gen39_conceal.py`, `scripts/train_gen45_unified.py` (the
gen32 trainer on the unified substrate); the gen39 probe chain in `scratch/gen39_*.py`; the
gen42-45 drivers (`scratch/gen42_battery.py`, `gen43_{bank,exam,mark}.py`,
`gen44_budget_sweep.py`, `gen45_{corridor_hunt,gate,score,verify,worstcase}.py` plus
`gen45_batch.sh`/`gen45_chain.sh`); artefacts under `models/runs/` and `assets/`. Suite last
recorded 246 green.

## Running things on this machine (measured)

The gen39 trainer costs ~1.83 s/flight solo, ~3.3 s at 4-way; ~3.1 GB per run; FOUR concurrent
is the safe shape on 24 GB. The gen45 trainer runs 0.79-1.02 s/sortie at THREADS=2, ~1.7 GB
per run, three concurrent comfortable. Do not `nice` training runs, and mind the measured zsh
trap: backgrounding from an interactive zsh yields nice 5 (the efficiency-core penalty) while
`bash -c '... &'` yields nice 0, so launch through bash and verify `ps -o nice` after start.
This sandbox's `python3` lacks site packages; run every aerial command through
`../sacred/.venv/bin/python` with `PYTHONPATH=.`. Cap all maths thread pools to 1 and set
`OMP_WAIT_POLICY=PASSIVE`. Oracle probes use a 9-10 worker pool. Kill by explicit PID with a
self-excluding pattern and verify over 30 seconds.

## Recorded future work (nothing scheduled)

1. The step-5d replication training: one command (the step-5 trainer's qwenthink16 arm on
   `curricula_qwenthink2.json`, 3 seeds, scored on the fresh zero-shot sets); would give the
   diversity candidate its second matched pair. Needs Kilian's explicit go.
2. The designed diversity-manipulation test (author curricula at matched strength with
   deliberately varied diversity); would turn "supported" into "proven" or kill the carrier.
3. The step-5 validation-cache rebuild (cheap; would tighten every step-5 number).
4. The LLM-proposes, local-search-refines hybrid (both halves measured, combination untested).
5. The LLM-doctrine arm stays excluded on evidence (free gate 0.53-0.75x the tuned recipe).
6. The gen28 permuted-field row (eval-only, minutes) must run before any thesis "on sight" or
   amortisation sentence from the static aerial register.

## Archive

This worktree's stale copies of the shared documentation (frozen around 2026-07-16), the two
dated aerial handovers and the 2026-07-28 doc register live in `docs/archive/` (see its
`INDEX.md`). The chronicle lives on the roads worktree; this tree's `SACRED_PROGRESS.md` is a
pointer stub, with the old stale copy (which carried entry 34) archived. The ledgers shared
across worktrees (`b2_llm_benchmark`, `gen19_b1lite1`, `gen27_dynamic_generalist`,
`regime_decision_table` and the gen01-gen25 set) were reconciled to one canonical version on
2026-08-04; if you amend a shared ledger, mirror the change on all trees in the same session.
