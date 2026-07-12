# B2: the agentic-LLM exploitability benchmark (harness READY; live runs await API credentials)

- **status: PRE-REGISTERED 2026-07-12 (NEXT_STEPS_MASTER Block B item B2). Harness built and
  dry-run-validated (`scratch/b2_llm_benchmark.py`); LIVE runs are BLOCKED on Kilian providing
  API keys + a model list + a spend cap (external API spend is his resource decision; the
  standing launch authority covers local CPU only). One command per model once keys exist.**
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
