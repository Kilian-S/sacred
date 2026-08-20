# SACRED

**Adversarial reinforcement learning for contested resupply routing.** Code and experiment
records for an MSc thesis at Imperial College London (Kilian Schwarz, 2026).

<img src="assets/theatre.png" alt="The Kaliningrad to Gvardeysk corridor in Mission Control, with line-of-sight-masked air defences and the trained policy's route" width="100%">

Send convoys down the same road often enough and an ambusher learns it. Fly the same corridor
twice and the air defence is already aimed there. Classical planners are built to find the one
best route, which is exactly the habit that gets a convoy caught. SACRED is a Soft Actor-Critic
policy trained by fictitious self-play against best-responding adversaries. It learns calibrated
route randomisation for convoy fleets and drone flights moving through networks under hidden,
pre-committed interdiction, and every result is scored against exactly computable optima.

The thesis asks a question the field had not answered with measurement, namely when a learned
routing policy is worth using at all, and answers it as a map. Below a measurable boundary,
simple rules are provably optimal and the thesis says so. Beyond it, where exact solvers and
hand-written rules both give out, SACRED delivers value no competing method matches.

## Findings

**One policy, never-seen city.** Trained by adversarial self-play on three cities and deployed
zero-shot on a fourth, SACRED beats every fixed strategy the city admits, at 0.639 of the best
static bound. An identical twin with its route memory deleted collapses to 1.434, so the learned
use of history carries the entire gain.
([record](roads/experiments/gen27_dynamic_generalist.md))

<img src="assets/flagship_transfer.png" alt="Zero-shot performance on the held-out city against the static cap, with the memory-deleted control" width="100%">

**Learning pays exactly where computation fails.** Against an adversary that studies the
defender's recent pattern, SACRED beats every heuristic at interdiction budgets K=3 to K=6, with
margins widening to 20.7%, across the entire range where the exact optimum is still computable
and beyond the reach of any solver.
([record](roads/experiments/gen43_unified_kboundary.md))

<img src="assets/dynamic_ladder.png" alt="The dynamic ladder: SACRED below every heuristic from K=3 to the computability wall" width="100%">

**And the map is honest about where it does not.** Against an adversary that commits in advance,
a two-line stack over the edge-disjoint routes equals the exact game value at low budgets and is
never beaten by training on the instrument. The thesis's opening act measures adversarial
training against congestion actively worsening robustness. Knowing when not to deploy learning
is part of the result.
([record](roads/experiments/gen43_unified_kboundary.md),
[campaign](roads/experiments/gen06_dynassign_matrix.md))

<img src="assets/static_ladder.png" alt="The static ladder: the naive stacks stay ahead of the trained policy at every committed-adversary budget" width="100%">

**The mechanism travels from roads to real terrain.** Rebuilt for aerial resupply on the real
Kaliningrad to Gvardeysk corridor, with terrain-dependent reach, lethality and line-of-sight
masking, the policy beats the static reference on 18 of 18 held-out cells at 0.351 of it, 1.46
times the exact optimum, discovered zero-shot on threat fields it never trained on.
([record](aerial/experiments/gen45_unified_corridor.md))

**Language models slot in where language enters the loop.** Reading short, sometimes
contradictory intelligence prose, an LLM identifies the enemy's doctrine 60 of 60 times where
keyword matching collapses, and hands a type-conditioned SACRED the ability to cross a
performance wall it provably cannot cross alone (0.664 against the type-blind 1.373). Curricula
authored by a thinking-mode reasoner train the defenders that transfer best.
([record](roads/experiments/gen38_llm_enemy_id.md),
[record](aerial/experiments/gen39_concealment.md))

<img src="assets/enemy_id.png" alt="Doctrine identification under degrading intelligence, LLM against keyword control, and the value of a correct call" width="100%">

## Mission Control

The interactive counterpart to the thesis. A web application over the same engine and the same
records, where the games are played live: watch strategies duel across a real city, take either
side yourself, command the air defence against the trained policy, or build a playable theatre
from any rectangle of the real world.

<img src="assets/play.png" alt="The placement game in Mission Control: a hand-placed air defence laydown mid-mission against the trained policy" width="100%">

Repository: [sacred-mission-control](https://github.com/Kilian-S/sacred-mission-control)

## Layout

The repository holds two self-contained arms.

- `roads/` covers the road-based interdiction games (thesis Acts 1 to 3 and the road half of
  Act 5). `src/` holds the environments, agents and oracles, `scripts/` the trainers and
  evaluators, `analysis/` the exact solvers, probes and scoring tools, `tests/` the suite
  (171 tests), `experiments/` the experiment records, `data/maps/` the extracted city graphs.
- `aerial/` covers the aerial theatre games (thesis Acts 4 and 5) in the same layout, with 246
  tests and the theatre geometry under `data/maps/`.

## Experiment records

Every experiment has one record in `experiments/`. A record states the question, the pinned game
configuration, the pre-registered decision criteria, the baseline family and the results, and
nothing else. All numbers cited in the thesis trace to these records.

| Thesis Act | Records |
|---|---|
| Act 1: Congestion | `roads/experiments/gen03_robustness_dynassign.md` to `gen07_contested_matrix.md` |
| Act 2: Incomputability at Scale | `roads/experiments/gen43_unified_kboundary.md` |
| Act 3: Zero-Shot Transfer | `roads/experiments/gen27_dynamic_generalist.md`, with controls in `gen16_multicity.md`, `gen21_vanilla_transfer.md`, `gen22_rotation.md`, `gen25_dr_control.md`, `zst_map_robustness.md` |
| Act 4: From Road-Based to Air-Based Resupply | `aerial/experiments/gen31_aerial_dyn.md`, `gen45_unified_corridor.md` |
| Act 5: LLM-Assisted SACRED | `roads/experiments/b2_llm_benchmark.md`, `gen34_hidden_adversary.md`, `gen38_llm_enemy_id.md`; `aerial/experiments/gen39_concealment.md`, `gen43_exam.md`, `gen44_budget_sweep.md` |
| Synthesis and References | `roads/experiments/regime_decision_table.md`, `ref_artificialanalysis_llm_index.md`; `aerial/experiments/theatre_atlas.md` |

## Reproduction

Each arm runs independently on Python 3.13 with its `requirements.txt`. From inside an arm
directory,

```
pip install -r requirements.txt
PYTHONPATH=. python -m pytest tests/
```

Training and evaluation commands are pinned in the experiment records; run outputs land under
`models/runs/`, which is not shipped (the records name the artefact paths that a rerun
regenerates). All results were produced on a single laptop-class CPU. The language-model
experiments call locally served open-weight models (Llama-3.3-70B and Qwen3.6-27B, plus the
Qwen3.5 family) through an OpenAI-compatible endpoint and log every request and reply; an
equivalent endpoint is required to rerun them.

## Data

Road graphs and theatre geometry derive from OpenStreetMap data, © OpenStreetMap contributors,
available under the Open Database Licence (ODbL). The maps carry network geometry only.
