# SACRED

Code and experiment records for the MSc thesis on adversarial reinforcement learning for
contested resupply routing (Kilian Schwarz, Imperial College London, 2026). SACRED is a
Soft Actor-Critic policy trained by smooth fictitious play against best-responding
adversaries. It learns calibrated route randomisation for convoy fleets and drone flights
moving through networks under hidden, pre-committed interdiction, and every result is
scored against exactly computable optima.

## Layout

The repository holds two self-contained arms.

- `roads/` covers the road-based interdiction games (thesis Acts 1 to 3 and the road half
  of Act 5). `src/` holds the environments, agents and oracles, `scripts/` the trainers
  and evaluators, `analysis/` the exact solvers, probes and scoring tools, `tests/` the
  suite (171 tests), `experiments/` the experiment records, `data/maps/` the extracted
  city graphs.
- `aerial/` covers the aerial theatre games (thesis Acts 4 and 5) in the same layout,
  with 246 tests and the theatre geometry under `data/maps/`.

## Experiment records

Every experiment has one record in `experiments/`. A record states the question, the
pinned game configuration, the pre-registered decision criteria, the baseline family and
the results, and nothing else. All numbers cited in the thesis trace to these records.

| Thesis Act | Records |
|---|---|
| Act 1: Congestion | `roads/experiments/gen03_robustness_dynassign.md` to `gen07_contested_matrix.md` |
| Act 2: Incomputability at Scale | `roads/experiments/gen43_unified_kboundary.md` |
| Act 3: Zero-Shot Transfer | `roads/experiments/gen27_dynamic_generalist.md`, with controls in `gen16_multicity.md`, `gen21_vanilla_transfer.md`, `gen22_rotation.md`, `gen25_dr_control.md`, `zst_map_robustness.md` |
| Act 4: From Road-Based to Air-Based Resupply | `aerial/experiments/gen31_aerial_dyn.md`, `gen45_unified_corridor.md` |
| Act 5: LLM-Assisted SACRED | `roads/experiments/b2_llm_benchmark.md`, `gen34_hidden_adversary.md`, `gen38_llm_enemy_id.md`; `aerial/experiments/gen39_concealment.md`, `gen43_exam.md`, `gen44_budget_sweep.md` |
| Synthesis and References | `roads/experiments/regime_decision_table.md`, `ref_artificialanalysis_llm_index.md`; `aerial/experiments/theatre_atlas.md` |

## Reproduction

Each arm runs independently on Python 3.13 with its `requirements.txt`. From inside an
arm directory,

```
pip install -r requirements.txt
PYTHONPATH=. python -m pytest tests/
```

Training and evaluation commands are pinned in the experiment records; run outputs land
under `models/runs/`, which is not shipped (the records name the artefact paths that a
rerun regenerates). All results were produced on a single laptop-class CPU. The
language-model experiments call locally served open-weight models (Llama-3.3-70B and
Qwen3.6-27B, plus the Qwen3.5 family) through an OpenAI-compatible endpoint and log every
request and reply; an equivalent endpoint is required to rerun them.

## Data

Road graphs and theatre geometry derive from OpenStreetMap data, © OpenStreetMap
contributors, available under the Open Database Licence (ODbL). The maps carry network
geometry only.
