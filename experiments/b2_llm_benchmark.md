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
