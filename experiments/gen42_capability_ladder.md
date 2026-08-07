# Generation: gen42_capability_ladder (does model capability predict task performance?)

- **status: PRE-REGISTERED 2026-08-07, BEFORE any ladder call or box change. Kilian's
  in-conversation go 2026-08-06/07: ladder design agreed (rungs, battery, llama as
  off-ladder reference), llama may be stopped whenever (agreed with Pan), eval-only, no
  training anywhere.**
- **git SHA at registration: aerial `7335ca2` (+ this commit).**

## Question

Across a capability ladder that holds the model FAMILY, quantisation recipe and serving
stack fixed, does general model capability predict performance on this pipeline's three
measured LLM registers, and where are the thresholds? The gen39 arc measured two models
and found a task-dependent reversal; the ladder turns those two points per register into a
curve. Pre-committed sub-questions: (i) is performance monotone in parameter count within
Qwen3.5 (2B -> 4B -> 9B -> 27B) per register; (ii) does one model GENERATION at fixed size
(Qwen3.5-27B vs Qwen3.6-27B, architecture-identical configs) move performance more or
less than a size step; (iii) does the search-bound register (1e slot choice) stay flat
while language/composition-bound registers climb, which is the where-LLMs-help map's
sharpest prediction; (iv) does the crown's thinking mode change any of it (its on/off pair
is free at fixed weights).

## The ladder (identity facts verified on-box 2026-08-06, stagehand report
## `/home/killian/ladder_prep/REPORT.md`)

Rungs, all cyankiwi compressed-tensors pack-quantized W4A16 group-32 (the resident
crown's recipe; the only recipe delta is symmetric zero points on the 3.5 line vs
asymmetric on the crown), all `Qwen3_5ForConditionalGeneration`:

1. `cyankiwi/Qwen3.5-2B-AWQ-4bit`
2. `cyankiwi/Qwen3.5-4B-AWQ-4bit`
3. `cyankiwi/Qwen3.5-9B-AWQ-4bit`
4. `cyankiwi/Qwen3.5-27B-AWQ-4bit`
5. `cyankiwi/Qwen3.6-27B-AWQ-INT4` (the resident crown, served as `qwen3-27b`), thinking
   OFF and thinking ON arms.

`llama-3.3-70b` (AWQ-INT4, Llama 3.3, late 2024) is the OFF-LADDER reference line, not
rung zero: cross-family and cross-vintage, its banked and corrected-brief values are
reported beside the curves and never enter the monotonicity statistics. Chat-template
divergence is neutralised by the gateway (`default_thinking: false` on every rung; the
2B's template defaults thinking off while 4B/9B/27B default on, so the flag is
load-bearing and is pinned in every draft entry).

## The battery (identical for every rung; corrected-brief instruments ONLY, post the
## 2026-08-06 gen39 repair; qwen thinking OFF except the crown's ON arm)

- **B-COMP (composition, the register LLMs won):** the step-2 protocol verbatim
  (`scratch/gen39_compose.py` machinery, narva, K=3, v2 table passed), n=16 forces per
  rung + the 16-force RELABEL control (the binding terrain-grounding control); scored
  exactly (irreducible + vs-searcher + cover).
- **B-SLOT (grounded slot choice, the search-bound register):** the corrected Phase-1e
  protocol verbatim, n=4 lineages x 2 rounds = 8 calls per rung; ceiling 0.0278 fixed by
  the same exhaustive machinery; grounding, free lanes, urban uptake reported.
- **B-EFF (sample efficiency):** the Phase-1f protocol at evaluation budgets 8 / 16 / 96,
  one lineage per rung, against the banked hill-climbing/random curves (which are
  model-independent and reused, never re-run).

Estimated calls per rung ~45 (16 + 16 + 8 + ~5); six rungs (incl. both crown arms)
~270 calls total, all logged by the box's audit layer.

## Data-validity bars (pre-registered; the act itself is DESCRIPTIVE, no pass/fail
## verdict hangs on a curve's shape)

1. A rung's cell is VALID only if >= 6/8 (B-SLOT) and >= 12/16 (B-COMP) calls return
   schema-valid forces; below that the cell is marked FORMAT-LIMITED and excluded from
   curves (reported, never dropped).
2. A B-SLOT cell with grounding < 80% is marked INTERFACE-LIMITED (the gen39 lesson:
   check the interface before concluding capability); its score still plots, flagged.
   *(Amended 2026-08-07 before any cross-rung comparison, after the crown off-arm run:
   the grounding metric needs >= 4/8 parsable INTENDED_ROUTES declarations to be read at
   all; below that the cell is PARSE-LIMITED, a distinct marking, since qwen-family
   models omit the declaration line in 6-8 of 8 responses across every 1e run to date.
   Declaration-compliance itself is reported per rung as a row.)*
3. Pre-committed readings: per-register Spearman of score vs parameter count over the four
   3.5 rungs; the generation delta (3.5-27B vs 3.6-27B, same size) compared against the
   largest size step; the flat-vs-climbing register contrast; the crown on/off delta
   against the (repaired) rider result. Both directions of every reading are reportable;
   no wording is licensed beyond the measured curves and the standing per-model rules.
4. The banked llama/qwen gen39 numbers are never mixed into ladder curves (vintage and
   family confounds); they appear as labelled reference lines only.

## Operations (the box plan, agreed with Kilian; Pan informed)

One rung at a time (global tensor-parallel 2 spans both GPUs): stop `llama-3.3-70b`
(Kilian's authority, 2026-08-07 "you can turn llama off whenever"), mount a rung from the
pre-validated draft entries (`/home/killian/ladder_prep/models_draft.json`), health-check,
run the battery from the Mac, rotate to the next rung, and at the end RESTORE the standing
pair (llama + qwen3-27b). The crown's two arms need no mounting (resident weights, thinking
is a per-request flag). All swaps are performed by the Opus stagehand agent under written
checklists; every step additive and reversible; `nvidia-smi` is broken box-wide (NVML
skew), VRAM telemetry via `torch.cuda.mem_get_info`.

Artefacts: `models/runs/gen42_ladder/<rung>/{comp,slot,eff}.json` (naming reconciled to the
driver 2026-08-07, before any battery result existed) + a consolidated
`models/runs/gen42_ladder/ladder.json`; battery driver `scratch/gen42_battery.py` (wraps
the three gen39 harnesses with MODELS/OUT rebound per rung, the phase1e_thinking pattern;
committed before the first battery call).

## RESULTS (appended per rung; nothing above changes after results exist)
