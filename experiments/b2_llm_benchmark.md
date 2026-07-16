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
