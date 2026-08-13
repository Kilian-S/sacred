# B2: the agentic-LLM exploitability benchmark (harness READY; live runs await API credentials)

- **status: PRE-REGISTERED 2026-07-12 (NEXT_STEPS_MASTER Block B item B2). Harness built and
  dry-run-validated (`scratch/b2_llm_benchmark.py`). UNBLOCKED 2026-07-16: a LOCAL LLM workbench
  is available (no commercial API keys / no spend needed) — Prof. Angeloudis's GPU box.**
  - **Endpoint:** OpenAI-compatible gateway on the box's port 8080, key `iits-local-key`. NOT
    directly reachable from Kilian's Mac (shared Tailscale node, ACL blocks 8080); reach it via SSH
    tunnel `ssh -N -L 18080:localhost:8080 killian@100.88.32.88` (username DOUBLE-L `killian`, pw
    `tsl2026`), then `http://localhost:18080/v1`. Verified working end-to-end 2026-07-16.
  - **Models:** live = `llama-3.3-70b` (AWQ-INT4, 32K ctx); `qwen3-27b` (27B, 64K ctx) one
    `./start.sh start qwen3-27b` away (fits the VRAM budget alongside the 70B). Pinned open-weight
    models = more reproducible than a moving API; NO tools available = the informative no-tools
    register this pre-registration wants; the gateway auto-logs every call to
    `/home/llm/vllm-server/audit/YYYY-MM-DD.jsonl` = the transcript record for free.
  - **Two TODOs before the live run:** (1) Kilian's explicit go (it uses the shared GPU box);
    (2) add a generic OpenAI-compatible `--base`/`--key` path to `scratch/b2_llm_benchmark.py`
    (the `openai` provider branch already POSTs to `/v1/chat/completions`; just parameterise the
    host instead of hardcoding api.openai.com). Then: run llama-3.3-70b, enable + run qwen3-27b.
  - Full box inventory: HANDOVER top banner (2026-07-16); instructions
    `../../Connecting to local LLM workbench.txt`.
- **git SHA:** the commit landing this ledger + the harness.

## Why (Kilian's named idea; CRITIQUE_EXAMINER §6 item 7; CRITIQUE_12-07-26 §6 item 5)

The security game has a computable optimum, a computable deterministic trap, and a graded ladder
between them: an unusually clean capability probe for language agents, on a yardstick nobody has
scored them against. It also independently supports the thesis mechanism: calibrated
randomisation is exactly what language agents are believed to lack unaided.

## Design (fixed before any live call)

Game: the 35-159 headline instance (N=3, K=1, band 0.15-0.95, 12 routes), full specification in
the prompt (per-edge lengths + interception probabilities, the mission objective, the adversary
model per register). NO tools (with code execution a frontier model would simply solve the LP;
tools-allowed may be reported later as a separate ceiling row). Prompts, model ids, and raw
transcripts are logged verbatim (`models/runs/b2_llm/<model>.json`).

| register | prompt contract | scored as | anchors |
|---|---|---|---|
| (a) deterministic | "choose ONE route; adversary best-responds" | worst-case of the chosen route | loss_det 0.699 (any fixed route >= this game's per-route worst) |
| (b) stated-strategy | "commit to a probability distribution" | EXACT stated mixture under oracle BR | equilibrium 0.206; uniform-stack 0.442; SACRED 0.256 |
| (c) agentic-sequential | T=30 sorties vs the gen19 pattern-of-life adversary (w=3, tau=0.15), per-sortie outcome feedback | realised mean mission-failure | static_det 0.613; iid_eq 0.147; SACRED 0.050; history_opt 0.049; dry-run uniform agent 0.356 |

**Pre-registered hypotheses:** (a) lands at or above the per-route worst-case floor (deterministic
= exploitable, whatever route is argued); (b) lands between uniform (0.442) and the equilibrium
(0.206), miscalibrated toward uniform or toward cost; (c) is the open question: does in-context
adaptation discover anti-repeat hedging (repeat-rate below the uniform agent's ~0.32 and mean
failure materially below iid_eq 0.147)? Any model matching SACRED's 0.050 would be a headline.

**Scope guard (agreed):** at most one thesis subsection + one ladder column; the natural home is
a workshop-paper spin-out. If it threatens the writing calendar it moves whole to post-freeze.

## Dry-run validation (2026-07-12, no API): the pipeline scores end-to-end

Uniform synthetic agent: register (c) mean mission-failure 0.356 over 20 sorties (between iid_eq
0.147 and static_det 0.613, as it must be), repeat-rate 0.32; registers (a)/(b) parse and score.

## LIVE RESULTS (appended per model once credentials exist)

### DISJOINT-BASELINE APPENDIX (2026-07-16, Block R0; oracle/eval-only)

> **Context (binding wording rule; CRITIQUE_16-07-26.md §1; probes
> `scratch/disjoint_baseline_probe.py`, `scratch/r0_screen.py`, artefacts
> `models/runs/r0_screen.json`):** the candidate menus' first routes ARE the max-flow
> decomposition, and "uniform-stack over the edge-disjoint routes" (2 lines) is the strongest
> NAIVE baseline. No comparative sentence in this ledger may claim SACRED beats "every
> uncalibrated strategy class" or that its transfer is something "standard algorithms cannot
> achieve" without the rows below beside it.

**Anchor amendment (BEFORE any live run):** register (b) gains the heuristic anchor
**uniform-disjoint-stack 0.250** (35-159; inv-vuln 0.241) between uniform-stack 0.442 and
SACRED 0.256; and a new PRE-REGISTERED scored question: **does the model discover
independent-route (max-flow) reasoning?** — scored (i) behaviourally (stated mixture's mass on
the disjoint core; distance to the heuristic vs to uniform) and (ii) by transcript annotation
(does the rationale mention route independence/shared edges?). A model that reasons its way to
~0.25 has matched the trained headline with language alone — a headline finding for the
benchmark EITHER way, and exactly why the anchor must be pre-registered rather than discovered
in review.

## LAUNCH RECORD: the 35-159 overnight cell (2026-07-16 ~23:50, Kilian's explicit go)

**Live-test calibration (both transcripts committed as the validation record,
`scratch/b2_livetest_{llama,qwen}_transcript.txt`):** three harness defects found and fixed
BEFORE any batch data: reply token budget 1000 -> 12000 (qwen3-27b reasons at length; its
decision turn truncated mid-derivation), HTTP timeout 120 s -> 900 s, and the comprehension-gate
checker moved to LAST-match regex (first-match misgraded a reasoning model's correct final
answers: qwen's true gate is 3/3, llama's 1/3 stands). Live-test scores (n=1 each, NOT batch
data): both models committed the SAME distribution — uniform over routes 4-11, the
lowest-per-route-risk but maximally-shared cluster — scoring 0.663 (3.2x eq, worse than
uniform-menu-stack 0.442), while the post-probe shows the independence concept is available on
demand (qwen named {0,1,2,3} exactly). Design decisions finalised with Kilian in-conversation:
UNHINTED only; llama-3.3-70b + qwen3-27b; three instances (35-159 tonight; Gdansk OD + 71-33
K=5 after a harness extension tomorrow); direct gateway (no tunnel; Tailscale stays up);
sequential per model, both models concurrent (one in-flight request each); one retry per
conversation on transport failure; per-model wording rule ("strong open-weight models").

**Tonight's cell:** per model: register (a) x 10 seeds, (b) x 10 seeds, (c) x 5 episodes
(T=30, gen19 adversary w=3 tau=0.15); llama additionally the temperature-sensitivity row
((b) x 5 at T=0.3 and T=0.8). Seeds drive the pre-registered label permutation. Outputs
`models/runs/b2_llm/batch_35159/` (full transcripts in every JSON); runner
`scratch/b2_batch_35159.sh`; SHA = the commit landing this record.

## RESULT: the 35-159 cell (2026-07-17 ~00:40, both models, all three registers)

Scored: `models/runs/b2_llm/batch_35159_scored.json`; transcripts verbatim in every per-run JSON.
Register (b)/(a) n=10 seeds each; register (c) n=5 episodes (T=30 sorties, gen19 pattern-of-life
adversary w=3 tau=0.15). Gate checker = last-match (the live-test fix).

| register | anchor context | llama-3.3-70b | qwen3-27b |
|---|---|---|---|
| (a) deterministic (worst-cased) | loss_det 0.699 | 0.978 | 0.841 |
| (b) stated-strategy | uniform-menu 0.442 · disjoint-heur 0.250 · **SACRED 0.256** · eq 0.206 | **0.604 +/- 0.100** (gate 1.0/3) | **0.523 +/- 0.161** (gate 2.1/3) |
| (c) sequential vs pattern-of-life | iid_eq 0.147 · **SACRED 0.050** · history_opt 0.049 | **0.177 +/- 0.018** (best 0.149) | **0.297 +/- 0.176** (best 0.059) |

**Reading (the pre-registered hypotheses, resolved):**
1. **Register (a) lands ABOVE loss_det** for both (0.84-0.98): asked for one route with the
   adversary best-responding, both pick a route whose worst case exceeds even the best
   deterministic plan — they do not compute the minimax-safe single route. As pre-registered.
2. **Register (b) is the headline: neither model calibrates.** Both land 0.52-0.60 — WORSE than
   naive uniform-menu-stacking (0.442), 2x worse than the 2-line disjoint heuristic (0.250), and
   ~2.4x SACRED (0.256). The dominant failure (seen in every transcript) is spreading mass over
   the lowest-individual-risk routes, which are the MOST edge-shared cluster, so one ambush
   covers most of the mass. Language models minimise per-route risk and diversify naively while
   ignoring route CORRELATION — precisely the "predictability with extra steps" failure the
   thesis documents for the non-adversarial control. The gate shows this is NOT a comprehension
   failure (qwen 2.1/3, and both name a correct independent set in the post-probe): knowledge
   present, strategic application absent. **The dissociation is the finding.**
3. **Register (c) is where in-context adaptation partially works.** Given per-sortie feedback,
   both models drop far below their own static register-(b) play (llama 0.177, qwen best 0.059)
   and below iid_eq (0.147) on their best episodes — they DISCOVER round-robin / anti-repeat
   cycling from the feedback loop (the raw sequences show explicit rotation, e.g. qwen seed 2:
   2-0-1-3 repeating). But it is brittle: qwen's variance is huge (0.059 to 0.605 — one episode
   locks onto a fixed 4-cycle 6-7-10-11 and gets punished), and NEITHER approaches SACRED's 0.050
   or history_opt 0.049 reliably. The measured anti-repeat rate ~0.00 confirms they avoid
   immediate repeats but do not find the calibrated hedge.

**What the benchmark banks (binding wording):** *strong open-weight language models (Llama-3.3-70B,
Qwen3-27B; pinned revisions, no tools), given the security game in full, fail to design a
calibrated mixed strategy unaided — they play worse than a two-line heuristic and worse than
naive uniform stacking despite demonstrably understanding the route structure when asked directly
— but in-context sortie-by-sortie feedback lets them partially recover through emergent anti-repeat
cycling, still short of the trained policy and the computable optima. Calibrated randomisation is
exactly the capability language agents lack unaided; adaptive feedback narrows but does not close
the gap.* Two open-weight models only (per-model wording; not "LLMs" in general). Instances
Gdansk + 71-33 K=5 follow the harness extension (register the extension before running them).

**Harness note (disclosed):** the `--print-prompts` / scoring keys are `a_deterministic`,
`b_stated`, `c_agentic`; the c-register carries `mean_mission_failure`, `choices`, `repeat_rate_w`,
`gate`. Live-test calibration fixes (max-tokens, timeout, gate last-match) are in the launch record.

## LAUNCH: the held-out Gdansk cell (2026-07-17, OD 249-95, the ZERO-SHOT comparability instance)

Harness parameterised (`--od --K --city`; per-instance gen19 dynamic anchors via `oracle_refs`;
reg-(b) scored via `env.exploitability_of_occupancy_dist`, robust to the greedy mode). OD 249-95 is
one of gen27's SIX held-out Gdansk test ODs (R=10, eq 0.302, det/eq 2.45), so the LLM sits in the
IDENTICAL zero-shot position as the gen16 amortiser and the gen27 dynamic policy — the direct
"can a language agent do what the trained transfer policy does, on a city it was never given
before?" comparison. Same footprint as the headline cell: reg (a)/(b) x10 seeds, reg (c) x5
episodes, both models, concurrent, one retry/turn. Runner `scratch/b2_batch_gdansk.sh`.

## RESULT: the held-out Gdansk cell (2026-07-17, OD 249-95, both models, all registers): the finding REPLICATES zero-shot

Anchors (this OD). STATIC reg (b): det 0.740 · uniform-menu-stack 0.694 · **disjoint heuristic
0.333** · equilibrium 0.302. DYNAMIC reg (c): static_det 0.692 · **iid_eq cap 0.223** ·
history_opt 0.079 · **gen27 trained policy zero-shot ~0.098 abs (0.44x cap)**.

| register | llama-3.3-70b | qwen3-27b |
|---|---|---|
| (a) deterministic | 0.867 | 0.867 |
| (b) stated-strategy | **0.798 +/- 0.072** (gate 2.0/3) | **0.354 +/- 0.066** (gate 2.1/3) |
| (c) sequential vs pattern-of-life | 0.325 +/- 0.059 (best 0.214) | 0.394 +/- 0.047 (best 0.346) |

**Reading (replication + one sharp new result):**
1. **The calibration failure REPLICATES on a never-seen city.** In register (b) llama lands 0.798
   (worse than deterministic 0.740 — actively harmful randomisation) and qwen 0.354; BOTH miss the
   2-line disjoint heuristic (0.333) and the equilibrium (0.302). Same mechanism as 35-159:
   diversify over low-per-route-risk routes, ignore correlation. So "language agents cannot design
   calibrated mixed strategies unaided" is not a one-instance artefact — it holds zero-shot.
2. **NEW — the models split, and qwen is genuinely close to the heuristic here.** Unlike 35-159
   (where both tied ~0.6), on 249-95 qwen (0.354) approaches the disjoint heuristic (0.333) and
   its gate is 2.1/3, while llama collapses to 0.798. The smaller model reasons its way nearer the
   independence structure on this instance; a real model x instance interaction (report per-model,
   per-instance, no pooling — the standing rule).
3. **Register (c): neither approaches the trained policy.** Both drop below their static play via
   in-context anti-repeat (0.325 / 0.394) but sit at 1.5-1.8x the iid_eq cap (0.223) and ~3.5-4x
   the gen27 zero-shot policy (~0.098) and ~4-5x history_opt (0.079). The trained history-aware
   policy's zero-shot dynamic hedging is NOT reachable by in-context adaptation here.

**What the two-instance benchmark now banks (binding):** *across a headline and a held-out-city
game, strong open-weight language models fail to design calibrated mixed strategies unaided —
matching or losing to a two-line max-flow heuristic and, on the transfer instance, playing worse
than deterministic routing — despite passing comprehension; in-context feedback yields emergent
anti-repeat that stays well short of the trained policy and the computable dynamic optimum. The
capability gap is calibrated randomisation, and it does not close with model reasoning or with
adaptive feedback at these scales.* Two open-weight models, two instances (per-model, per-instance
reporting; no "LLMs in general"). REMAINING (optional): 71-33 K=5 (past-the-wall reasoning) needs
the greedy-yardstick scoring path + a design decision on its dynamic anchors; flagged, not blocking.

## LAUNCH: the 71-33 cell (PRE-REGISTERED 2026-08-12, BEFORE any call; Kilian's instruction)

**Why.** The thesis's Act 2 was consolidated onto the 71-33 six-corridor instance (gen43,
`experiments/gen43_unified_kboundary.md`), so its Act-5 LLM subsection (4.5.1) currently quotes
a game the rest of the results chapter no longer uses. Kilian's direction (2026-08-12): re-run
the B2 test on 71-33 so the LLM ladder sits on the standard instrument. K=1 (the banked B2
protocol, like-for-like with the 35-159 cell); the old optional K=5 idea above stays unrun. The
35-159 and Gdansk cells stay banked; the Gdansk zero-shot sentence in the thesis stays as is.

**Design (harness byte-identical; instance is the only moved variable).** Harness
`scratch/b2_llm_benchmark.py` UNCHANGED at SHA `83781ff` (+ this fold); flags
`--od 71-33 --city kaliningrad --K 1` (N=3, k-extra 8, menu-select, band 0.15-0.95, R=11,
exact attacker). Footprint per model exactly the banked cells': registers (a) x10 seeds,
(b) x10 seeds, (c) x5 episodes (T=30, gen19 pattern-of-life adversary w=3 tau=0.15);
temperature 0.7, max-tokens 12000, one retry per conversation. Models `llama-3.3-70b` and
`qwen3-27b` (served alias; identity **Qwen3.6-27B** per the 2026-08-06 on-box identity check;
thinking OFF, the gateway default, matching every banked B2 call). Endpoint pinned to the
MagicDNS name `http://cv-iits-w05.tail5b8d80.ts.net:8080/v1` (the 2026-08-06 transport repair;
both endpoints verified alive from Python today, both models served). Runner
`scratch/b2_batch_7133.sh`; outputs `models/runs/b2_llm/batch_7133/` (full transcripts in
every JSON). Eval-only, no training anywhere.

**Anchors (ALL reproduced 2026-08-12 by `scratch/b2_7133_anchor_probe.py`, artefact
`models/runs/b2_llm/b2_7133_anchors.json`; every banked value to 4 dp before any call).**
- Register (a): loss_det **0.4199**.
- Register (b), one-shot stacked: equilibrium v* **0.1276**, attained EXACTLY by the
  inverse-vulnerability disjoint stack (the worst-edge and budget-max definitions coincide at
  K=1); uniform-disjoint **0.1666**; uniform-full-menu **0.2252**; inv-vuln-full 0.2502;
  trained SACRED (gen43 static K=1) **0.160 +/- 0.003**.
- Register (c), w=3 tau=0.15, all exact: dynamic optimum (Karp) **0.0313**; best rule =
  rotation **0.0387**; composed anti-repeat (core) 0.0423; full-menu anti-repeat 0.0728;
  iid_eq **0.0967**; static_det 0.3835; trained SACRED (gen43 dynamic K=1)
  **0.0462 +/- 0.0008**; matched-budget window-Q 0.0472.
- **Yardstick guard (binding rule 8):** the harness JSON's `history_opt_c` field comes from
  the defective undamped-RVI `oracle_refs` and is NOT citable; the exact anchors above are
  the record for this cell.

**Pre-registered expectations (both directions reportable, per-model as always).** From the
two banked instances: (a) lands at or above loss_det; (b) misses the disjoint structure and
lands at or above the uniform-full-menu stack 0.2252, far above v* 0.1276, with a possible
model x instance split (the Gdansk qwen pattern); (c) discovers anti-repeat from feedback,
dropping below its own static play but staying above the exact optimum 0.0313 and short of
the trained policy 0.0462. The pre-registered scored question carries over: does the stated
mixture put mass on the disjoint core (distance to the inv-vuln stack vs to uniform), and
does the rationale mention route independence/shared edges? Note for the thesis wording: on
this instance the two-line stack is not merely strong but EXACTLY OPTIMAL at K=1, and
trained SACRED itself sits behind it (0.160) and behind rotation dynamically (0.0462 vs
0.0387, the gen43 thin-slack cell), so every comparative sentence names the ladder exactly.

### RESULT: the 71-33 cell (2026-08-12, batch 15:47-20:02 BST + a quiet-box recovery of two
### episodes; artefacts `models/runs/b2_llm/batch_7133/` + `batch_7133_scored.json`, scorer
### `scratch/b2_score_7133.py`; registration SHA `f129694`, results at this fold's commit)

**Process disclosures, before any verdict.** (i) llama completed all 25 conversations in
~17 min; qwen took ~4.2 h (essay-length turns throughout, two concurrent streams sharing the
box). (ii) Two qwen register-(c) episodes (seeds 0 and 4) failed both scripted attempts under
that load, one on the 900-s turn timeout and one on HTTP 400 (consistent with the 64K context
ceiling under essay-length turns; the surviving episodes carried up to ~24k tokens of replies).
Recovered by re-invoking the idempotent runner on the quiet box after the batch; both landed
on the next attempt and are disclosed as third-attempt samples. (iii) No other retries; zero
parse failures; final set 25/25 conversations per model. (iv) Endpoint MagicDNS as registered;
no `src/` or `scripts/` change anywhere in this cell (scratch + ledger only).

| register | anchor context | llama-3.3-70b | qwen3-27b (= Qwen3.6-27B) |
|---|---|---|---|
| (a) deterministic (worst-cased) | loss_det 0.4199 | 0.641 (route 4 on 9/10) | 0.572 (route 5 on 10/10) |
| (b) stated-strategy | **v* 0.1276 = inv-vuln stack (exactly optimal)** · uniform-disjoint 0.1666 · uniform-full 0.2252 · SACRED 0.160 | **0.619 +/- 0.000** (gate 1.2/3) | **0.254 +/- 0.076** (gate 2.1/3) |
| (c) sequential vs pattern-of-life (w=3, tau=0.15) | opt 0.0313 · rotation 0.0387 · SACRED 0.0462 · iid_eq 0.0967 | **0.069 +/- 0.024** (best 0.033) | **0.054 +/- 0.043** (best 0.000) |

Per-seed (b): llama 0.619 x10 (two distinct supports, identical value); qwen 0.298, 0.157,
0.173, 0.298, 0.375, 0.298, 0.128, 0.298, 0.216, 0.297. Per-episode (c): llama 0.0614,
0.0330, 0.0798, 0.0652, 0.1063 (repeat-in-window rate 0.00); qwen 0.0598, 0.0664, 0.1241,
0.0000, 0.0174 (repeat rate 0.21; the 0.0000 episode is a realisation of the sampled
adversary, not a stationary value).

**Readings (against the pre-registered expectations, all of which held; the qwen split was
pre-flagged and fired).**
1. **Register (a) as pre-registered:** both models sit above loss_det (0.641 / 0.572 vs
   0.4199) and each commits a single fixed route; neither computes the minimax-safe route.
2. **Register (b), llama: the calibration failure in its sharpest form yet.** Ten seeds
   produce only TWO distinct distributions (pure route 4; 0.2/0.8 over routes 4 and 10),
   and the two score identically (0.619) because routes 4 and 10 share their worst segment.
   4.9x the optimum, 2.7x uniform-full-menu stacking, core mass 0.28, and WORSE than the
   best deterministic route (0.4199): actively harmful randomisation, previously seen only
   on the Gdansk transfer cell, now on the home instrument.
3. **Register (b), qwen: the model x instance interaction, amplified.** Ten distinct
   distributions, core mass 0.55, best seeds 0.128 and 0.157 at or near the exact optimum
   (which the two-line stack attains), worst 0.375; mean 0.254 above uniform-full 0.2252
   (below it on 4/10 seeds, below uniform-disjoint on 2/10). Qwen's failure mode on this
   instance is RELIABILITY, not level.
4. **The pre-registered max-flow question, scored on the post-probe transcripts:** qwen
   names the EXACT maximal independent set {0,1,2,3,4,5} on 6/10 probes (pairwise-disjoint
   on 7/10); llama names the valid pair {1,3} on 6/10 and invalid larger sets otherwise
   (exact 0/10). The knowledge-application dissociation therefore holds in its purest form
   for qwen (names the exact core, still commits unreliable mixtures); for llama on this
   instance BOTH halves are weak. **The banked "both models name near-correct independent
   sets" sentence is NOT licensed on 71-33; per-model wording binding.**
5. **Register (c): both models land between the trained policy and static play** (llama
   0.069, a strict out-of-window cycler; qwen 0.054), above rotation 0.0387 and the optimum
   0.0313 on average, with SACRED 0.0462 ahead of both means. The relative gap to the
   trained policy is much smaller than on 35-159 (1.2-1.5x vs ~3.5x), and the mechanism is
   structural, not a capability jump: with six disjoint corridors the naive cycling the
   models discover in context is close to the best rule, the same thin-slack fact (rules
   leave only 1.24x over the optimum; gen43) that makes this cell the static concession
   region. Any quote of these numbers carries the instance with them.

**What the three-instance benchmark now banks for 71-33 (binding wording for the thesis's
4.5.1 rebase):** *on the consolidated Act-2 instrument, where the two-line
inverse-vulnerability stack is exactly optimal at K=1, neither pinned open-weight model
states a reliably calibrated mixture. Llama-3.3-70B commits near-deterministically to
overlapping padded routes and scores worse than the best single route (0.619 vs 0.4199).
Qwen3.6-27B reaches the optimal stack's level on its best draws (0.128 vs 0.1276) and spans
0.13-0.38 across seeds, with its mean above naive uniform stacking; it also names the exact
six-corridor independent set on demand, so its gap is application, not knowledge. Both
remain behind trained SACRED (0.160). Given sortie-by-sortie feedback both discover
anti-repeat cycling that lands between the trained policy and static play, and lands
comparatively close here because six disjoint corridors make naive cycling structurally
strong.* Per-model, per-instance, as always; no pooling with the 35-159 or Gdansk cells.

## LAUNCH: the THINKING-MODE rerun, all three cells, qwen only (PRE-REGISTERED 2026-08-13,
## BEFORE any batch call; Kilian's go: "B2 benchmark and gen39 step 2, nothing else, speed
## is priority 1")

**Question.** Does the deliberation mode change qwen's play on the B2 registers? The gen43
exam measured deliberation moving SCORES within the seed-reroll noise floor at the slot
register; the mixture-calibration and in-context registers have no thinking measurement.
On 71-33 qwen-off's register-(b) failure is RELIABILITY (0.254 +/- 0.076, best 0.128, worst
0.375), the sharpest open question.

**Design.** qwen3-27b ONLY (llama has no deliberation mode). Protocol identical to each
banked cell (same instances, registers, footprints a x10 / b x10 / c x5, seeds, temperature
0.7, endpoint) except two forced co-changes per the 5c precedent: `enable_thinking: true`
via `chat_template_kwargs`, and max-tokens 12000 -> 16000 (thinking traces brush lower
caps, leaving empty content). The harness gains an additive `--thinking` flag (absent =
byte-identical request bodies) and records decoding provenance per JSON. Smoke: one
register-(a) conversation at seed 99 (outside the real seed range, scratch path, not data):
plumbing end-to-end, gate 3/3, probe named the exact core. Runner
`scratch/b2_batch_think.sh`, three cells as three concurrent streams, ~215 calls/cell.
Outputs `models/runs/b2_llm/batch_{35159,gdansk,7133}_think/`.

**Corrected dynamic anchors, pinned (binding rule 8).** 71-33: as the 2026-08-12 cell
(opt 0.0313, rotation 0.0387, iid_eq 0.0967, SACRED 0.0462). 35-159: exact optimum 0.0413,
ATTAINED by rotation (gen40 sanity anchor; the banked cell's history_opt 0.049 was the
pre-repair defective RVI and is retired for comparisons), iid_eq 0.1468, SACRED 0.050,
static_det 0.613. Gdansk 249-95: iid_eq 0.223 (exact); the exact optimum is recomputed via
`scratch/dyn_exact.py` at scoring time and the banked 0.079 `oracle_refs` figure is not
citable. Register-(b) anchors unchanged per cell (banked).

**Expectations (both directions reportable, judged per cell, never pooled).** From the exam:
no score movement beyond the instrument's own spread would be unsurprising. The readable
positive is variance collapse toward the stack on 71-33 register (b) (deliberation fixes
calibration reliability); the readable negative is spread unchanged or worse. Noise floors
stated: (b) sd 0.076 at n=10, (c) sd 0.043 at n=5 (off-mode measurements). REPORTED validity
rows: empty-content turns and parse-fallback counts per cell (the thinking-overrun risk);
gate means; wall-clock. Comparisons are thinking-vs-off within each cell only. The gen39
step-2 thinking row is registered in its own ledger (aerial) and runs Mac-side in parallel.

### RESULT: the thinking-mode rerun (appended after the batch; nothing above changes)

**AMENDMENT (2026-08-13 14:4x, BEFORE any register-(b) thinking score was read; the 5c-class
forced co-change, applied once and disclosed).** At the registered 16,000-token cap the
thinking trace CENSORS register (b) specifically: over half of (b) attempts terminated with
null content (12 of ~21 resolved cells burned both scripted attempts; registers (a) and (c)
completed in full at the same cap, all 30 + 15 conversations, with only isolated single
retries). The mixture-commitment prompt is the one register that invites a full in-head
derivation of the game, and 16k truncates it on most draws. Consequence: register (b) reruns
at a UNIFORM 32,000-token cap (a-, c-registers stand as completed at 16k); the nine
16k-completed (b) cells are set aside as `*_16k.json` sidecars (preserved, reported in the
validity rows, never mixed into the scored table, since a 16k-completed sample is conditioned
on short-deliberation draws). The cap gates termination, not content, so it cannot bias a
completed conversation's score. Runner `scratch/b2_batch_think_b32k.sh` (b-only, six
workers). The censoring rate itself is banked as a validity finding: deliberation-mode output
length is register-dependent, and the strategy-commitment register is where it explodes.
