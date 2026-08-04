# HANDOVER.md: the AERIAL worktree (branch `gen28-aerial`)

> Rewritten 2026-08-04 in the project-wide documentation refactor. This file absorbs
> `HANDOVER_AERIAL_28-07-26.md` (now in `docs/archive/`, content preserved) and is the
> authoritative state of the aerial branch. **The master project state, the claims register and
> the consolidated wording rules live in `../sacred/HANDOVER.md`; read `CLAUDE.md` first for
> identity, house rules and the project map.** Citable numbers live only in the `experiments/`
> ledgers of this worktree.

## The state in one paragraph

The aerial campaign is COMPLETE. gen39 (concealment + the LLM curriculum arc) closed 2026-07-28
through step 5 with zero-shot rows on all four theatres; nothing is running, the tree is clean,
no decision is pending. The branch holds five acts: gen28 (the free-flight negatives, the
retired Tier-1 tie, and the binding baseline-completeness appendix), gen31 (the aerial dynamic
positive on the synthetic lattice), gen32 (the same positive reproduced TIGHTER on the real
Kaliningrad-Gvardeysk terrain, with the ops-map exhibit), gen33 (the LLM red-force act, its
terrain control failing), and gen39 (the concealment mechanic, the LLM composition positive
with its terrain control passing, and the curriculum negative diagnosed then fixed).

## What each act banks (ledger pointers)

| act | verdict | ledger |
|---|---|---|
| gen28 | free-flight negatives (menu bandit, credit starvation, corridor collapse); Tier-1 fleet claim RETIRED as a tie by the 2026-07-19 appendix (read the appendix FIRST); surviving claim = payoff-blind frontier MATCHING; the permuted-field row must run before any "on sight" sentence | `experiments/gen28_aerial.md` |
| gen31 | the aerial trained positive, one attempt, blind-confirmed (6/6 beats-cap on 3/3 fresh seeds, blinded control causal) | `experiments/gen31_aerial_dyn.md` |
| gen32 | gen31 reproduced on REAL terrain, tighter (pooled 0.451x cap, 1.30x the exact optimum); deliverable `scratch/gen32_ops_map.html` | `experiments/gen32_theatre_dyn.md` |
| gen33 | LLM red force; K=3 coordinated passes for both models but the binding scrambled-terrain control FAILS, so no terrain-grounding claim; metric 2 left open, superseded by gen39 | `experiments/gen33_llm_adversary.md` |
| gen39 | the concealment mechanic with an internal control; the LLM composition positive with the terrain control PASSING; the curriculum negative diagnosed (irreducible threat) and FIXED at step 5 with zero-shot transfer | `experiments/gen39_concealment.md` (reader's summary at top; nine superseded blocks deliberately visible) |
| theatre atlas | the four scored theatres (kgd, ukraine, narva, fulda) | `experiments/theatre_atlas.md` |

## gen39 in brief (the branch's most recent act)

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

**Binding boundaries.** No arm beats the best simple OBSERVING rule on any cell of any map, so
no "trained policy beats the rules" sentence. llm16 and local16 are indistinguishable
in-distribution, so no "the LLM curriculum is best" sentence. Everything is per-model and the
models reverse between tasks. Never mix game versions (symmetric/asymmetric forest,
leaked/masked concentration, grid/quota sampling) in one figure.

## Code map (what is new on this branch)

`src/envs/aerial_sector.py` (gen28 lattice), `src/envs/aerial_theatre_vec.py` (vec-theatres,
terrain v2, quota sampler), `src/envs/aerial_theatre_env.py` (any vec-theatre SAC-trainable),
`src/envs/aerial_conceal.py` (gen39 concealment), `src/redforce.py` (LLM briefs,
`force_schema`); trainers `scripts/train_aerial_dyn31.py`, `train_aerial_dyn32.py`,
`train_gen39_conceal.py`; the gen39 probe chain in `scratch/gen39_*.py`; artefacts under
`models/runs/` and `assets/`. Suite last recorded 240+ green.

## Running things on this machine (measured)

The gen39 trainer costs ~1.83 s/flight solo, ~3.3 s at 4-way; ~3.1 GB per run; FOUR concurrent
is the safe shape on 24 GB. Do not `nice` training runs. Cap all maths thread pools to 1 and
set `OMP_WAIT_POLICY=PASSIVE`. Oracle probes use a 9-10 worker pool. Kill by explicit PID with
a self-excluding pattern and verify over 30 seconds.

## Recorded future work (nothing scheduled)

1. The LLM-proposes, local-search-refines hybrid (both halves measured, combination untested).
2. The step-5 validation-cache rebuild (cheap; would tighten every step-5 number).
3. The LLM-doctrine arm stays excluded on evidence (free gate 0.53-0.75x the tuned recipe).

## Archive

This worktree's stale copies of the shared documentation (frozen around 2026-07-16), the two
dated aerial handovers and the 2026-07-28 doc register live in `docs/archive/` (see its
`INDEX.md`). The chronicle lives on the roads worktree; this tree's `SACRED_PROGRESS.md` is a
pointer stub, with the old stale copy (which carried entry 34) archived.
