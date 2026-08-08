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

### RUNGS 5a/5b, THE CROWN (2026-08-07 morning; Qwen3.6-27B resident, thinking off then on;
### driver `scratch/gen42_battery.py`; artefacts `models/runs/gen42_ladder/qwen3-27b[_think]/`;
### off arm 8 min, on arm ~34 min; no phase failures)

| row | crown OFF | crown ON |
|---|---|---|
| B-COMP valid forces (bar >= 12/16 per arm) | 16/16 + 16/16 relabel | 12/16 + 13/16 relabel |
| B-COMP med irreducible (e0) llm / relabel | 0.0009 / 0.0003 (3.0x collapse) | 0.0012 / 0.0005 (2.4x) |
| B-COMP med vs-searcher (e1) llm / relabel | 0.0739 / 0.0896 (no collapse) | 0.0904 / 0.0918 (no collapse) |
| B-SLOT median (% of 0.0278 ceiling) | 0.0038 (14%) | 0.0040 (14%) |
| B-SLOT best / round trajectory | 0.0126 (45%); r0 28% -> r1 7% | 0.0127 (46%); r0 14% -> r1 23% |
| B-SLOT declarations parsed / urban uptake | 0/8 (PARSE-LIMITED) / 0/8 | 5/8 (grounding 100%) / 1/8 |
| B-EFF llm at 8 / 16 / 96 evals | 0.0352 / 0.0352 / 0.0442 | 0.0365 / 0.0365 / 0.0428 |
| B-EFF baselines | random/greedy/local reproduce the banked curves EXACTLY (seeded) | same |

**Flags, disclosed before any cross-rung reading.** *(kept in place; see the full-ladder
section below for the readings)* (i) The e1 medians are the compose
scorer's vs-searcher column aggregated median-of-field-medians; this is NOT the banked
step-2 headline aggregation (which used the best-simple-defender construction and read
0.0747/0.0613), so the consolidation step must recompute the banked aggregation per rung
before any comparison to step-2 sentences; e0/e1 are interim battery-internal rows.
(ii) The relabel control collapses the crown's forces on the irreducible metric (2.4-3.0x)
but NOT on e1 in either mode in this fresh sample; the banked 10-13x collapse was on the
banked aggregation. (iii) B-SLOT medians show large n=8 sampling variance against the
repair-day samples (off 14% here vs 19% then; on 14% here vs 29% then; round trajectories
disagree in direction), so cross-rung slot readings must pool or interval these, never
cite one sample. (iv) Thinking mode makes the crown COMPLY with the declaration format
(5/8 vs 0-2/8 in every off-mode run; grounding 100% where parsable), a per-mode interface
fact banked beside the scores. (v) Thinking's B-COMP validity dropped to 12/16, exactly
at bar.

### RUNGS 1-4 + THE LADDER READINGS (2026-08-07; runner take 3 + the supervisor's ops
### driver; ladder wall clock 10:57-12:04, box restored and verified 12:36; ops record
### `models/runs/gen42_ladder/OPS_REPORT.md`; zero battery-phase failures)

Per-rung rows (same battery, same corrected-brief instruments; validity bars applied):

| row | 2B | 4B | 9B | 27B (3.5) | crown 3.6 off | crown 3.6 on |
|---|---|---|---|---|---|---|
| B-COMP valid llm / relabel (bar 12/16) | **11 / 8 FORMAT-LIMITED** | 16 / 16 | 16 / 16 | 16 / 16 | 16 / 16 | 12 / 13 |
| B-COMP med e1 llm (vs-searcher) | (0.0922) | 0.0756 | 0.0618 | 0.0726 | 0.0739 | 0.0904 |
| B-COMP med e0 llm / relabel | (0.0002/0.0032) | 0.0001/0.0011 | 0.0001/0.0000 | 0.0002/0.0002 | 0.0009/0.0003 | 0.0012/0.0005 |
| B-SLOT valid n (bar 6/8) | **5 FORMAT-LIMITED** | 8 | 7 | 8 | 8 | 8 |
| B-SLOT median (% of 0.0278 ceiling) | (3%) | 3% | 6% | **15%** | 14% | 14% |
| B-SLOT best | (47%) | 14% | 22% | 62% | 45% | 46% |
| B-SLOT declarations / urban forces | 1/8 / 0 | 0/8 / 1 | 0/8 / 4 | 1/8 / 0 | 0/8 / 0 | 5/8 / 1 |
| B-EFF llm at 8 / 16 / 96 | .0359/.0359/.0364 | .0394/.0394/.0426 | .0193/.0341/.0433 | .0280/.0384/.0479 | .0352/.0352/.0442 | .0365/.0365/.0428 |

**Reading (i), monotonicity in size (valid 3.5 rungs).** B-SLOT climbs monotonically,
3% -> 6% -> 15% of ceiling (4B -> 9B -> 27B), and B-EFF at the full budget climbs
monotonically across all four rungs (0.0364 -> 0.0426 -> 0.0433 -> 0.0479). B-COMP is
NON-monotone, a flat 0.062-0.090 band from 4B to the crown: composition quality does not
track parameter count in this family. The 2B is FORMAT-LIMITED on both gated phases (the
smallest rung fails at output discipline before capability can be read).

**AMENDMENT (2026-08-08, before any B-COMP sentence is used anywhere): B-COMP AS RUN
CANNOT RESOLVE BETWEEN-MODEL DIFFERENCES, so reading (i)'s "flat" clause is restated as
UNDERPOWERED, not as measured sameness.** Per-force values (median over the 3 fields, the
`llm` arm) spread far more WITHIN one rung than the rung medians spread between rungs: the
3.5-27B alone spans 0.0594-0.1296 (spread 0.0702) while the six rung medians span
0.0618-0.0922 (spread 0.0303). Bootstrap 95% CIs on the difference of medians (20,000
resamples, rng(0)) contain zero for 4B-vs-3.5-27B [-0.0129, +0.0436], 4B-vs-crown
[-0.0206, +0.0423] and 3.5-27B-vs-crown [-0.0178, +0.0100]; only 9B-vs-crown separates
[-0.0284, -0.0028], which at 6 comparisons is what one expects by chance. **Binding
consequence:** no gen42 sentence may say composition quality is equal, better or worse
across rungs; the licensed sentence is that at n=16 per rung the instrument's
between-attempt noise exceeds any between-model effect it could detect, and the
discriminating gap gen39 step 2 relied on (0.0747 vs 0.0603, ~24%) is itself of the same
order as this noise, which is a second reason the standing banked-aggregation
recomputation flag must be cleared before any step-2 comparison. A powered B-COMP would
need a pre-run discrimination check (two known-different composers separated at the chosen
n) and materially more samples; recorded as the method fix, not run.

**AMENDMENT 2 (2026-08-08, same challenge applied to B-SLOT): the staircase's ENDPOINTS
survive, its intermediate steps and its top group do NOT.** Bootstrap 95% CIs on each
arm's slot median (20,000 resamples, rng(0), per-force values as fractions of the 0.0278
ceiling):

| arm | n | median | best | 95% CI on median |
|---|---|---|---|---|
| 4B | 8 | 3% | 14% | [0%, 7%] |
| 9B | 7 | 6% | 22% | [3%, 13%] |
| 3.5-27B | 8 | 15% | 62% | [12%, 49%] |
| crown 3.6 OFF (gen42) | 8 | 14% | 45% | [4%, 35%] |
| crown 3.6 OFF (gen39 repair) | 8 | 19% | 36% | [1%, 35%] |
| crown 3.6 ON (gen42) | 8 | 14% | 46% | [5%, 44%] |
| crown 3.6 ON (gen39 repair) | 8 | 29% | 42% | [9%, 37%] |
| llama-3.3-70b (gen39 repair, off-ladder) | 8 | 17% | 36% | [0%, 28%] |
| 2B | 5 | 3% | 47% | [0%, 47%] |

Licensed: **4B vs 3.5-27B separates** ([0,7] vs [12,49], non-overlapping), so "the smallest
competent rung is genuinely below the largest" stands. NOT licensed: the 4B-to-9B and
9B-to-27B steps individually (overlapping CIs), and any ordering within the top group
(3.5-27B, both crown modes, llama-70B all mutually overlapping). **Replication scale,
measured:** the crown ON arm was run twice on the same instrument and corrected brief and
returned 29% then 14%, a factor of two, which is the honest noise scale for every single
n=8 cell in this ledger.

**Crown pooled across its two samples (n=16 per mode) and the thinking test:** OFF median
16% [8%, 28%], ON median 20% [11%, 37%]; difference +5 points, CI [-12, +25],
**INDISTINGUISHABLE**. Binding consequence: no score claim may be made for thinking mode
in either direction. The thinking finding that DOES survive is the format-compliance count
(declarations 5/8 with thinking vs 0-2/8 without, across every run to date), which is a
count rather than a noisy median.

**2B FAILURE MODE (measured from `calls_comp.json`, 32 calls per rung): discipline, not
comprehension.** The 2B returned correctly structured plans with the right fields
(archetype, emplacement zone, doctrine, rationale); it failed by sending the wrong number
of teams (1 team on 6 calls, 5 teams on 2, against the required 3) and by running out of
the token budget mid-answer on 5 calls (finish_reason `length`). The 4B and 3.5-27B were
32/32 on both counts. The FORMAT-LIMITED marking therefore means "cannot hold the answer
format", NOT "cannot read the task"; the thesis sentence must say so.

**Reading (ii), generation vs size at fixed 27B (both thinking off).** Scores are flat
across the generation step: slot 15% -> 14%, comp e1 0.0726 -> 0.0739, eff@96 0.0479 ->
0.0442. The generation shows up in CONTROL behaviour instead: the relabel collapse on the
irreducible metric appears ONLY at the crown (3.6: 3.0x off / 2.4x on; 3.5-27B: none,
0.0002/0.0002), and thinking-mode format compliance exists only in 3.6. Suggestive, not
gating: e0 magnitudes are small and the banked step-2 aggregation recomputation (the
standing flag) must precede any terrain-grounding sentence.

**Reading (iii), the register contrast.** The search-bound register (B-SLOT) climbs with
size yet stays FAR below its 60% bar everywhere, topping out at 15% median (best single
force 62% at 27B); no rung, mode or generation approaches the ceiling. The where-LLMs-help
map's core prediction survives the whole family: scale buys a shallow gradient in the
search-bound register, not a crossing. Composition, the register LLMs won in gen39, is
already saturated-flat by 4B. Off-ladder reference: corrected-brief llama-3.3-70b sits at
17% median on B-SLOT, i.e. the 2024 cross-family 70B lands where the 2026 27Bs land.

**Compliance row (per the amended bar 2):** INTENDED_ROUTES declarations 1/0/0/1 of 8
across the 3.5 family and 0/8 crown-off: every off-mode cell is PARSE-LIMITED for
grounding; crown-on (5/8, grounding 100%) remains the only readable cell.

**Ops disclosure (analysis-relevant facts from OPS_REPORT.md).** Rung calls went
direct-to-port with per-call traces banked (`calls_*.json`) in lieu of the gateway audit
log. A stale duplicate 2B server on :8005 was removed BEFORE any battery request, so no
measurement saw split traffic. Three defects in `scratch/gen42_run_rungs.sh` are recorded
as KNOWN (mount-ssh blocks on healthy starts; pidfile captures the wrapper, not vllm;
`start.sh`'s 300s health timeout skips the gateway when a big model loads slowly), the
run was completed by the supervisor's ops driver (`gen42_ops_driver.sh`, archived beside
the logs) with byte-identical serve flags and battery invocations; fix before any reuse.
Box end state verified: gateway, llama and qwen all serving, rung ports clear, no sudo,
no other user's process touched.
