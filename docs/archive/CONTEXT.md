# CONTEXT.md — The Blueprint (project state & reference)

> **⚠️ STATUS BANNER (2026-07-06) — §0–§2 below are the historical record up to the Stage-2 hybrid
> build (2026-07-01); they are intentionally preserved, not current. THE EXPERIMENTAL CAMPAIGN IS
> COMPLETE — read `HANDOVER.md` first.** The project was **REFRAMED 2026-07-02** (`CRITIQUE.md`):
> headline = *robustness of adversarially-trained vs non-adversarially-trained SAC under held-out
> attacks*. The campaign (gen03→gen06, all pre-registered) then produced the definitive finding:
> **adversarial training does not confer — and measurably worsens — held-out robustness in this
> framework** (gen06, competence-gated: dD_targeted −881 ± 284, 0/3 pairings; robustness ranking
> greedy > vanilla > adversarially-trained), with a complete mechanism chain: the learned
> adversary can't learn to attack (gen03/04: below-random, entropy pinning), the protagonist
> can't learn decision-dense arenas (gen05: ceiling compression), and adversarial exposure
> degrades learning SNR even in the best case (gen06). Root cause: the zero-sum latency reward's
> signal-to-noise structure. **Next phase = thesis writing.** The living records are
> `HANDOVER.md` (state + outlook), `SACRED_PROGRESS.md` (per-run narrative), and the
> `experiments/genNN_*.md` ledgers (all numbers).
>
> **Read order for a new agent:** `HANDOVER.md` → `SACRED_PROGRESS.md` → `CRITIQUE.md` → the
> ledgers gen02–gen06 → this file (history) → `PROBLEM_REDESIGN.md` (historical design) →
> `SYSTEM.md` → `TASK.md`, and the literature review at
> `../../MT_Literature_Survey_Kilian_Schwarz_split.pdf` (two folders above the repo root).

---

## 0. TL;DR for whoever picks this up
SACRED is an adversarial (asymmetric zero-sum) deep-RL framework for the Stochastic & Dynamic VRP: a **protagonist dispatcher** vs an **antagonist** that injects congestion. The training *machinery* (SAC + ATLA + GNN) is now **stable and correct** after a long debugging effort. **But the current problem is too easy** — demand is spread across the map, so per-step routing decisions are near-inconsequential, the critic learns that all destinations are equivalent (the `Q_Spread` diagnostic collapses to ~0), and RL cannot beat a naive baseline (delivery rate sits flat at ~0.91 across every run). **We therefore pivoted to a problem redesign** (`PROBLEM_REDESIGN.md`): a *dynamic, multi-depot, latency-objective* VRP, reached via a curriculum, built ALONGSIDE the static baseline (kept for ZST). **Progress (2026-06-29):** Stage 0 (single-truck next-hop) DONE — *matches* greedy. Static 3b (multi-truck assignment) was built and briefly read as "first RL beats classical", but a **windowed re-read RETRACTED it** (mean `gap_atk` +18 = a loss; the −56/−188 were the noisy tail; the metric was mis-specified — single-episode vs the *co-evolving* antagonist). **Stage 1.5** (the assignment lever in the **Poisson dynamic** regime, measured with a fixed-adversary / best-checkpoint protocol) is **DONE (2026-06-30): RL does not beat greedy** — a near-wash like Stage 0 / static-3b (best-checkpoint `gap_atk` within noise; reliable ~6% static loss; antagonist runaway). **Our interpretation (recorded as our reasoning, open to challenge): the three near-washes (Stage 0, static-3b, Stage 1.5) all share destination-mode auto-routing, which we think starves the antagonist of leverage — so we hypothesised the missing lever is *next-hop routing*.** On that hypothesis we built **Stage 2 = hybrid (assignment + next-hop routing)**: H1–H6 done (env state-machine, chokepoint geometry, route-reach antagonist, greedy baseline, headroom check, eval; **75 tests green**), **not yet trained (H7)**. The pre-training headroom check is the most promising signal so far — the route-reach antagonist inflates greedy's latency **+79%** (vs ~3–8% in every prior rung) — but whether RL can *recover* that is untested. See the Stage-1.5 + Stage-2 records in §2, `TASK.md`, and `scratch/hybrid_geometry.png` (the current network).

---

## 1. What SACRED is (architecture)
**SACRED** (Soft Actor-Critic Robust Evolutionary Deep RL) models the SDVRP as an **asymmetric zero-sum Markov game**:
- **Protagonist (central dispatcher):** controls a fleet of trucks to fulfil demand; minimizes cost (delivery time / unserved demand).
- **Antagonist (adversary):** dynamically injects edge congestion under a budget to maximize disruption. `antagonist_reward = -protagonist_reward` (zero-sum).
- **Algorithm:** Soft Actor-Critic (max-entropy, off-policy, with an Experience Replay Buffer / ERB), trained via **ATLA** (Alternating Training with Learned Adversaries — one agent trains while the other is frozen, for stationarity).
- **Networks:** GATv2 graph encoder (PyTorch Geometric) + policy/value heads. State = a graph observation (nodes/trucks/edges) featurized into a PyG graph.
- **Thesis objectives:** (1) the zero-sum game formulation; (2) the sim environment; (3) SAC+ATLA + **ERB bootstrapping via population-based metaheuristics (ALNS)**; (4) **SBO** for facility-location/fleet composition; (5) evaluation vs SOTA metaheuristics; with **Zero-Shot Transfer (ZST)** as the crown-jewel claim.

### Module map (current code)
- `src/env/graph_env.py` — tick-by-tick physics/routing engine (trucks move along edges, deliver, reload). Demand is currently **static** (all placed at t=0). `reset()` restores `_initial_graph`.
- `src/env/smdp_wrapper.py` — event-driven SMDP wrapper; collapses ticks into protagonist/antagonist decision events; computes the reward (`_accumulate_step`); holds `SMDPTransition` (the replay record, with a lazily-built `feature_cache` excluded from pickling).
- `src/envs/osm_factory.py` — `make_osm_env(...)`: builds the Kaliningrad OSM env (290 nodes / 412 edges). **Demand is re-sampled randomly on every call** (`generate_stochastic_demand`); 4 trucks, 2 depots, 150 packages, `max_ticks=600`.
- `src/agents/sac.py` — `ProtagonistSAC`, `AntagonistSAC`, `ProtagonistQNet`, `AntagonistQNet`, `ReplayBuffer`, `_collate_graphs`, `_cached_featurize`. The SAC math lives in `.update()`.
- `src/agents/networks.py` — `GATv2Encoder`, `ProtagonistPolicyValueNet`, `AntagonistPolicyValueNet`, `featurize_state`. Each net is split into `encoder` + `head` (for batched updates).
- `src/agents/sacred_atla.py` — `ATLACoevolutionTrainer`: the ATLA loop + TensorBoard logging + checkpointing.
- `src/baselines/metaheuristic.py` — `AdaptiveLargeNeighborhoodSearchVRP` (ALNS), used to generate ERB demonstrations.
- `scripts/train_sacred.py` — training entrypoint (CLI). `scripts/generate_erb_osm.py` — parallel OSM ERB generator.

---

## 2. CURRENT STATUS — the pivot (most important section)
The training machinery works; **the problem does not exercise it.** Evidence:
- The `Value/Protagonist_Q_Spread` diagnostic (max−min Q across allowed destinations; added this session) **collapsed 5.3 → 0.46** over training — the critic correctly learns that destinations are near-equivalent in value.
- **Delivery rate is flat at ~0.91 across all three full runs**, regardless of fixes/reward changes. Protagonist policy entropy never falls (stays ~3.7, near max).
- **Root cause = problem structure, not a bug:** on the OSM map only **95/290 nodes (33%) carry demand, ~1–2 packages each**; from most positions several nearby destinations are equally good, so the per-step routing decision is near-inconsequential and ~0.91 is the time/capacity-constrained ceiling any coverage policy reaches. With demand everywhere the **antagonist is also toothless** (reroute to a different nearby node).

**Decision (with Kilian):** redesign the problem so decisions are consequential and the adversary matters — see **`PROBLEM_REDESIGN.md`**. Headline target = **Option 3**: a *dynamic multi-depot VRP* with **rate-based Poisson demand arrivals**, **hybrid assignment+routing** control, a **delivery-latency** zero-sum reward, reached via a **curriculum** from a single-truck routing-only validation rung. Three difficulty axes to sweep: **spatial K (hotspots) × temporal ρ=λ/μ (load) × adversarial budget**, plus ZST.

**Important nuance for judging future runs:** with this demand structure, *some* residual entropy is INHERENT (many decisions are genuinely soft). **Judge learning by delivery/latency improving and `Q_Spread > 0`, NOT by entropy collapsing.**

**Status of the redesign:** drafted and refined with Kilian; the §7 "supervisor agenda" in `PROBLEM_REDESIGN.md` is still open (Kilian will discuss with his supervisor). **Stage 0 (§7.1) is fully specified and does not depend on that agenda**, so it can be built — see `TASK.md`. The old `protag_reward_shaping` run was killed; the redesign is the active epic.

### Stage 0 — BUILT & VALIDATED (2026-06-28)
Stage 0 evolved from "single-truck routing-only" into a **next-hop route-choice** rung (the policy picks each edge, so the antagonist's congestion is a *learned, exploitable* decision — destination-choice + auto-A* gave the antagonist no leverage). Setup: OSM graph, **1 truck / 1 depot**, focused demand (single target), **latency reward**, **two-route corridor depot `14` → target `82`** (fast 6.4 / safe 7.4, disjoint).

**What made it learn (3 real fixes, in order found):**
1. **`reward_scale` 0.01→0.1** — at 0.01 the per-transition latency reward (~−0.15) was dwarfed by the SAC entropy bonus α·H (~0.5) → the agent maximised *entropy* not delivery (delivery collapsed to ~0, Q went *positive* +6). Next-hop fragments the episode into many short transitions, so the old scale was ~10× too small.
2. **Exploration fix** — pure next-hop on 290 nodes never reached the target from cold start (flat reward, 0% delivery). Fixed with a **forward α-corridor mask** (`routing_corridor_slack`, neighbours that keep the truck within α× the shortest distance to goal) + **decisions only at branches** (auto-resolve forced 1-option moves) + **no-U-turn** (anti-oscillation). Random policy then delivers 12/12.
3. **`routing_corridor_slack` 1.5→1.2** — killed mid-corridor dithering (also ~halved episode time to ~13 s/ep).

**Outcome (1000-ep maturity run, eval every 100):** the stack **learns a consequential, adversarial policy** — `Q_Spread` 0.85→1.58 (rising, strong), entropy 0.60→0.34 (commits), antagonist co-evolves (Q 22→62), delivers 12/12 — and **MATCHES greedy under attack (final gap +24, ~1.6%) but does NOT beat it.** The eval gap oscillated/cycled (post-antagonist-phase sampling is pessimistic; antagonist tended to win the late arms race). **Reason (as predicted): single-truck route-choice vs a *reactive* greedy is structurally near-a-wash** — greedy reroutes optimally, so there's little anticipation headroom. **The beat-the-baseline headroom lives in ASSIGNMENT (multi-truck), where greedy is provably suboptimal.**

**Verdict:** Stage 0 **validated under the agreed hybrid criterion (b) — "learns a consequential adversarial policy + matches greedy."** Machinery is sound; proceeding to the **assignment lever (Stage 2 / "3b")**.

**New Stage-0 artifacts:** `src/envs/stage0_factory.py::make_stage0_nexthop_env`; `smdp_wrapper.py` `routing_mode="next_hop"` + forward/branch mask + units-based latency reward; `src/baselines/greedy_dispatch.py` (next-hop greedy + `run_episode` + `eval_cells`); `train_sacred.py` `--problem stage0` + `--eval-every`; `scripts/evaluate_stage0.py`; `scripts/spar_visual_stage0.py` (live PyGame); `tests/test_stage0.py` (50 tests pass). Trainer gained optional `eval_fn`/`eval_every` periodic eval.

**Held / known issues (not yet actioned):** smdp-wrapper logic-critic findings ×3 (SMDP intra-option discounting is lumped/approximate; protagonist dispatch tick skips `_age_congestion`; corridor `inf<=inf` edge case) — all low-severity. graph_env critic findings: dispatch-dict determinism (sort by truck_id); within-tick arrival ordering by truck_id not arrival-time (moot single-truck). Perf: cheap safe wins = cache sorted neighbours (S2) + hoist `valid_levels` (S1); **reject** the `info`-dict-reuse perf fix (would alias retained transition infos). **FIXED since:** the A\* `_heuristic` inadmissibility — `/100` still left it up to ~140% suboptimal on this graph (rounded edge weights vs lat/lon coords), so routing + the greedy baseline ETAs now use **exact Dijkstra** (`_get_shortest_path`, `greedy_dispatch`); `evaluate_stage0.stage0_config` slack mismatch (now 1.2).

### Development roadmap (curriculum) — where we are
- **Stage 0 (DONE):** single-truck **next-hop route-choice**, latency, adversarial. Validated under criterion (b): learns a consequential adversarial policy, **matches** greedy (route-choice vs reactive greedy is structurally near-a-wash).
- **Stage 3b "assignment probe" (RETIRED 2026-06-29):** n=2, contested *static* demand, destination-mode assignment. Briefly read as the beat-greedy milestone; **retracted on a windowed re-read** (see the Stage-3b banner above). Code kept as baseline.
- **Stage 1.5 "dynamic assignment" (DONE 2026-06-30 — near-wash; record below):** the 3b assignment lever in the **Poisson dynamic** regime (multi-truck, contested hotspot, destination-mode, latency). Built end-to-end (Poisson env, queue/ETA observation, dynamic baseline, fixed-adversary/best-checkpoint eval) and ran 2 seeds × 800 ep (`gen02_dynassign`). **RL does not beat greedy:** best-checkpoint `gap_atk` ≈ −106 but within ±~1000 noise (not significant, selection-biased), reliable static loss (~+350 / ~6%), antagonist Q runaway (37→116). Same near-wash as Stage 0 / static-3b. (We skipped the planned 1-truck-Poisson rung — it drops the assignment lever.)
- **Stage 2 / hybrid (BUILT H1–H6, UNTRAINED):** next-hop routing added to the assignment lever — the policy chooses each edge, so the antagonist's congestion becomes a *learned, exploitable* decision (route-around). Static demand first. Chokepoint geometry (depots 110/135, demand `78,130,49,224,48,17,47,46` east of the node-0 hub); route-reach antagonist (blocks the gateway ahead on a truck's route). **Not yet trained.** Headroom check: route-reach = **+79%** attack (strong); a static gateway-avoiding heuristic *cannot* recover it (gateways are on the critical path) → any recovery must be a *dynamic* strategy RL might learn. See §2 Stage-2 record. *(This is the action model `PROBLEM_REDESIGN` §3.3 calls the headline — but note it rests on our "routing is the missing lever" interpretation, which is unproven.)*
- **Stage 2 / headline (LATER):** **hybrid assignment + next-hop routing** (combine the two existing levers; multi-headed action), Poisson, full sweeps over **K (spatial) × ρ (temporal) × adversary budget**, + **Zero-Shot Transfer**. Plus Obj-3 (ERB via dynamic dispatcher), Obj-4 (SBO fleet), Obj-5 (vs SOTA metaheuristics, e.g. rolling-ALNS).
- **Locked design decisions (with Kilian):** reward = delivery latency; termination = fixed horizon; headline action = hybrid; headline demand = Poisson; baselines staged greedy→rolling-ALNS; SBO deferred.

### Stage 3b — assignment probe (2026-06-28) — ⚠️ MILESTONE RETRACTED (2026-06-29)
> **RETRACTED:** the "first RL beats classical" claim does NOT survive a windowed read — `gap_atk`
> over the 20 eval points is mean **+18** (a loss; the −56 final / −188 best were the noisy *tail*),
> `gap_noatk` is a reliable **+8% static loss**, and the eval metric is **mis-specified** (a single
> deterministic episode of learned vs the *co-evolving* antagonist → ±100 swing is arms-race timing,
> not robustness; RARL needs a *frozen* protag vs a *fixed/held-out* antagonist over *multiple
> instances*). Per-phase checkpoints weren't saved (only final, overwritten) → no cheap re-salvage.
> **Static 3b is retired; pivoted to Stage 1.5 (dynamic).** Code kept as baseline. Record preserved.

**Setup:** `src/envs/assignment_factory.py::make_assignment_env` — depots **110 & 135** (graph diameter, ~44 apart), 8 **contested** demand nodes (each ~equidistant from both depots, |Δdist|≤1.8: 237,78,130,27,49,224,43,220), capacity-1, destination mode, latency reward. `train_sacred.py --problem assign`; `scripts/evaluate_assignment.py` (`eval_cells_assignment`, learned policy uses **state projection**); `greedy_insertion_policy` baseline; `scripts/plot_assignment_geometry.py` (geometry reviewed & approved). ~10.5 s/ep.

**First run (`assign_probe`, 1000 ep, eval-every-50) FAILED — learned LOSES to greedy everywhere** (gap +172 post-protag / +208 post-antag, 0/20 wins; loses even *statically* 960 vs 734). Diagnostic showed confident convergence to a *bad* optimum (Q_Spread 3.8→4.6, entropy 0.64→0.31) — not undertraining.

**ROOT-CAUSE BUG (found & fixed — multi-truck credit/coordination path, never exercised before Stage 0 was single-truck):** in a simultaneous multi-truck decision, the trainer/eval passed the **same event-time action mask to every truck**, so the policy could assign **two trucks to the same request** (confirmed by trace: both trucks → node 49 on the first decision → one wasted trip). Greedy-insertion forbids this via *sequential claiming*; the RL path didn't → unfair handicap, explains losing even statically. **Fix:** mask-level **sequential claiming** in both `sacred_atla.py` (training loop) and `evaluate_assignment.py` — once a truck claims a *demand* node it's removed from later trucks' masks (depots never claimed); the trainer now stores the per-truck reduced mask so SAC `action_idx` stays consistent. Verified: double-assignments 1→0; 50 tests pass.

**Retrain (`assign_probe_claimfix`, 1000 ep, eval-every-50) — MILESTONE: first time RL beats a classical baseline in this project.** vs the broken run (0/20 wins, gap +172/+208): now **6/20 wins, gap_atk post-protag/post-antag mean +12/+24, final checkpoint BEATS greedy-insertion under attack by 56 (6.5%), best −188 (−22%), clear upward trend in the back half (ep 750–1000).** `Q_Spread` 4.1→5.0, entropy 0.58→0.27, antagonist co-evolves (Q 19→37). **Matches the headroom probe exactly:** ~0 static headroom → learned slightly *loses* statically (gap_noatk ≈ +68, ~9%, flat) because greedy is near-optimal unattacked; 8–17% adversarial headroom → learned *captures 6.5%* under attack. **This is the adversarial-robustness story (RARL/ATLA): trades ~9% nominal for a 6.5–22% win under attack** — validates the core thesis claim (Obj 1 zero-sum game + Obj 5 beat-classical) and the whole curriculum direction. **Caveats:** still loses statically (over-defensive); noisy/co-evolution cycling (not every checkpoint wins); single geometry, greedy (not yet SOTA) baseline.

### Experiment-management infrastructure (built 2026-06-28) — USE THIS FOR ALL FUTURE RUNS
Single noisy runs can't separate signal from seed-luck (our results swing ±100+). So we built **seeded "generations"**:
- `train_sacred.py` gained `--seed` (reproducible/labelled), `--group` (nests run under `logs/tb_runs/<group>/` + `models/runs/<group>/` so TensorBoard groups it), `--threads` (torch CPU cap for parallelism). Run name in a group = `<group>/<tag>_seed<k>`.
- `scripts/run_generation.py` — launches a generation (config-recipes × seeds) in parallel, throttled (`--max-concurrent`, `--threads`), writes `experiments/<group>.md` **ledger** (git SHA + question + decision-metric-fixed-in-advance). Recipes: `assign_erb`, `assign_noerb`, `stage0`.
- `scripts/aggregate_generation.py` — the deliverable: **mean ± std across seeds** per config + headline `gap_atk` verdict + optional mean±band plot. **Read this, not raw curves.**
- **Thread benchmark (recorded):** 4 torch threads optimal on the M4 (=default), sublinear scaling (1.83× at 4 cores) → parallel low-thread runs are throughput-efficient. `scratch/thread_benchmark.py`.
- **Methodology dogma now:** fix the decision metric *before* looking; report mean±std over ≥3 seeds; always include a control; never compare across git states (ledger pins the SHA).

### ERB ablation (`gen01_erb_ablation`) — STARTED, PAUSED, INCONCLUSIVE
Ablation: `{assign_erb (1600 greedy no-attack demos via `generate_erb_assign.py` + `--erb-path data/erb_assign.pt`), assign_noerb}` × seeds {0,1,2}. Launched 6 runs in parallel, then **paused for heat/noise** — only **`assign_erb_seed0` ran to completion** (resumed losslessly from its ep-50 checkpoint; checkpoints carry the full replay buffer). seed1/seed2 paused at ep 50 (resumable); no-ERB seeds not started. See `experiments/gen01_erb_ablation.md`.

**`assign_erb_seed0` result (n=1 — DO NOT over-conclude):** ERB **did not achieve its goal** — `gap_noatk` stayed **~+50** (start +54), i.e. it did *not* fix the static partition. `gap_atk` noisy (8/20 wins, best −96 mid-run) and **collapsed late** (ep 1000 = +282; `Q_Spread` 7.1→1.9, entropy 0.5→0.7 de-committing, antagonist Q 19→56). **Two takeaways:** (1) ERB-as-built isn't earning its keep; (2) **late co-evolution instability — the antagonist runs away in late phases and wrecks the protagonist — is the bigger, config-agnostic problem and the likely source of all our noise.** Also: the **final-checkpoint metric is misleading under co-evolution** (penalises the protagonist for whichever phase training ends on; this run was capable at −96 mid-training).

### FORK RESOLVED (2026-06-29, with Kilian) — see TASK.md
The static-3b milestone was **retracted** (windowed read; mis-specified metric; per-phase checkpoints not saved → no cheap re-salvage). Direction confirmed (Option 3); **static 3b retired**, pivoted to **Stage 1.5 (multi-truck assignment + Poisson)** with the instability/measurement fixes (best-checkpoint, fixed-adversary multi-instance eval, reined antagonist, per-phase checkpointing) **baked into the new rung** so the first dynamic run is measured right. The old A/B/C fork is closed.

### Stage 1.5 — dynamic assignment result (`gen02_dynassign`, 2026-06-30)
**Setup:** Poisson arrivals (λ=0.06, ρ≈1) on the contested 8-node hotspot band, 2 depots/trucks, **destination-mode assignment** (env auto-routes via Dijkstra; routing deferred), capacity-1, latency reward. **Antagonist** (after a budget-redesign for compute, see below): **full-blockage only** (`congestion_levels=(1.0,)`), **one roadblock per decision event** (`max_antag_actions_per_event=1`, new `SMDPConfig` field), `congestion_duration=120`, `antagonist_interval=25` → ~30 sustained roadblocks/ep. 800 ep, switch-every 50, **2 seeds**. Pre-training headroom gate: the urgency dispatcher (oldest-first) **loses at every load**; heuristic-antagonist attack cost ~+8% at λ=0.06; the clairvoyant ceiling's gap is dominated by free-flow optimism, not adversarial headroom.

**Two latent bugs the full-blockage config exposed (both fixed + regression-tested):** the antagonist's congestion-level value was **hardcoded `[0.25,0.5,0.75,1.0]`** in `AntagonistSAC.select_action` (with `(1.0,)` it picked 0.25 → the action mask rejected it → **budget stayed 0 = no adversary**) and again in the antagonist `update` parse (wrong level index → **IndexError crash** at the antagonist phase). Both now read `self.congestion_levels`. The old 4-level configs matched the hardcoded list, so no prior run was affected. **Compute:** the first launch (budget 4000, 4 levels, no cap) was infeasible — the antagonist spent its budget via ~133 congestion sub-actions/ep, each an SAC update → **antagonist phase 295 s/ep vs protagonist 45** (~47 h). The full-blockage + per-event cap redesign cut both phases to **~18 s/ep**.

**Result — decision metric fixed in advance (best-checkpoint `gap_atk` vs the FIXED final antagonist, 5 held-out demand seeds, mean±std across the 2 training seeds):**
- seed0 best = **ep50 (essentially untrained)**, `gap_atk` −33 ± 1223; seed1 best = ep550, −179 ± 909 → **cross-seed mean −106**, but **±~1000 std (SEM ±450) → not significant**, and **selection-biased** (min over 15 noisy snapshots; seed0's "best" being the untrained ep50 is the tell; most of seed0's snapshots actually *lose*, gap_atk +200…+650).
- `gap_noatk` ≈ **+348 (~+6%)**, low-variance, both seeds → RL **reliably loses statically**.
- Diagnostics: `Q_Spread` healthy (~7, no collapse), entropy 0.56→0.47; **antagonist Q 37→116 (runaway)** — the co-evolution instability; `gap_atk` swings ±1000 with it. Delivery ~0.55, mean delivered latency ~145, final queue ~22 (all flat).

**Reading:** the *machinery* is sound (Q_Spread high, antagonist learned to attack hard), but **RL does not beat greedy** — it matches-to-loses under attack within large noise and loses ~6% statically: the **same near-wash as Stage 0 and static-3b**. With the env auto-routing, the antagonist only degrades service *rate*; the protagonist's assignment-only lever can't convert that into a robust win, and the adversary runs away. **Conclusion: destination-mode assignment is structurally too thin; the missing lever is next-hop routing.** A 3rd seed was judged not worthwhile (noise-dominated + structural, not seed-count). **Next (our choice at the time): Stage 2 hybrid.** Artifacts: `make_dynamic_assign_env`, `--problem dynassign`, `scripts/evaluate_dynamic_assign.py`, `scratch/dynassign_headroom.py`, `scripts/animate_dynassign.py`, ledger `experiments/gen02_dynassign.md`.

### Chokepoint analysis of the Kaliningrad graph (2026-06-30)
To design a rung where *routing* matters, we analysed the graph (`scratch/find_chokepoints.py`, `chokepoints.png`). **Facts:** (a) 74 bridges/cut-edges, but all peripheral dead-end spurs (max betweenness 0.034) → **no critical bridge** (so "assignment-via-cut-edge" is unavailable on this map); (b) the graph is dominated by the **node-0 hub** — high edge-betweenness routing chokepoints `('0','1')` (betw 0.23) and `('0','129')` (betw 0.19), whose *edge-level* detours are large (8.9×, 18.9×) but whose **route-level** detours are moderate (**1.06–1.40×**: short edges with nearby parallel paths). A geometry search (`scratch/design_hybrid_geometry.py`, `hybrid_geometry.png`) placed 8 demand nodes east of the hub so Depot A's routes funnel through `('0','1')` and Depot B's through different gateways; searching depot pairs gave only marginal/degenerate gains → depots kept at 110/135.

### Stage 2 — HYBRID (assignment + next-hop routing) — BUILT H1–H6, NOT TRAINED (2026-07-01)
**What it is:** the protagonist makes two interleaved decision types — **assignment** (pick a pending request when idle at depot → sets `assigned_target`) and **next-hop routing** (choose each edge toward the target; the *policy* owns the path, so congestion is exploitable). `routing_mode="hybrid"` in the wrapper; the GNN policy head is unchanged (it scores candidate nodes for both). Static demand, capacity-1, latency, the chokepoint geometry above.
**Antagonist redesign for this rung:** **full-blockage only** (`congestion_levels=(1.0,)`), **1 roadblock per decision event** (`max_antag_actions_per_event`), `antagonist_interval=25` / `congestion_duration=125` (=5× → a block expires *on* a decision event, ~5 concurrent max), and a new **`antag_reach="route"`** — the antagonist may block the edges on a truck's *static shortest path to its target* (pre-block the gateway *ahead* = anticipation), vs the legacy `"leashed"` 3-hop reach. (Two latent bugs surfaced + fixed while building this: the antagonist's congestion *level value* was hardcoded `[0.25,0.5,0.75,1.0]` in both `select_action` and the SAC `update` parse → with `(1.0,)` it picked 0.25/crashed; now reads `self.congestion_levels`. Old 4-level configs were unaffected.)
**H5 headroom check (`scratch/hybrid_headroom.py`, static/deterministic) — data:** greedy no-attack 902; greedy vs **leashed** antagonist 931 (**+3.2%**); greedy vs **route-reach** antagonist 1615 (**+79%**); a static "avoid-the-gateways" heuristic delivers **nothing** (gateways are on the only routes to the demand). With `('0','1')` permanently blocked, greedy still delivers all 8 (routes around; 0 ticks trapped) because the route-level detour is only ~1.3× (< corridor slack 2.0).
**Our reading (labelled — untested):** route-reach makes the adversary *much* stronger than any prior rung (+79% vs ~5%), and the recovery strategy (if any) must be *dynamic* (timing / unpredictable routing / assignment-balancing) — greedy and static heuristics can't do it, RL might. **Not yet trained, so unknown whether the headroom is recoverable.** Honest risk: if the adversary follows reroutes too well, no policy recovers it and it washes like the others despite the strong attack.
**Artifacts:** `make_hybrid_assign_env`, `--problem hybrid`, `routing_mode="hybrid"` + `antag_reach`, `TruckState.assigned_target`, `hybrid_greedy_policy`, `scripts/evaluate_hybrid.py` (`sac_hybrid_policy`, `eval_hybrid_cells`, `select_best_checkpoint`), `tests/test_hybrid.py`, `scratch/{find_chokepoints,design_hybrid_geometry,search_hybrid_depots,hybrid_headroom}.py`, `scratch/{chokepoints,hybrid_geometry}.png`.

**Tests: 75 passing** (adds `tests/test_hybrid.py` (7) + `tests/test_dynamic_assign.py` (11)).

_(Older count, kept for history:)_ **Tests: 62 passing** (`tests/test_stage0.py`, `tests/test_assignment.py` incl. an SAC-update format-drift guard, `tests/test_dynamic_assign.py` (7, Stage-1.5 Poisson demand incl. the latency-telescoping invariant)). **A\* heuristic replaced by exact Dijkstra** (was ~140% inadmissible on this graph) — recontextualises Stage-0 greedy (true ≈ 996, not 1128).

---

## 3. Project history / journey (how we got here — so you don't re-tread it)
This codebase was originally built with Google Gemini; the prior Claude SWE session audited and de-risked it. In chronological order:

1. **Hygiene:** removed leftover debug `print()`s from the SMDP hot-path, deleted `patch.py`/`patch2.py` (source-rewriting debug injectors), fixed a `congestion_heap` cross-episode leak.
2. **The big correctness fix — SAC divergence.** The 2000-ep run `...0614_170342` **diverged** (α ran to 69, critic loss to 232k, Q to 601, policy stayed uniform). Root cause: the **alpha (temperature) auto-tune loss had an inverted sign** (`-log_alpha*(entropy-target)` = positive feedback). Fixed to `log_alpha*(entropy-target)`. Also: `target_entropy=-1.0` (unreachable for discrete entropy) → `None` (discrete-SAC fallback `+0.45·lnN`); added `clip_grad_norm_=10.0` on critic+actor. Validated stable.
3. **Performance.** Batched `update()` so the GATv2 encoder runs once per minibatch (split nets into `encoder`+`head`; `_collate_graphs`; guarded by `tests/test_batched_equivalence.py`); added a per-transition featurization cache; made checkpoints run-specific (`models/runs/<run>/`) and excluded the cache from pickling. **Honest result: ~1.45× on the protagonist phase, not the 5–8× first projected** — on CPU, batching a GNN only removes Python overhead, not the irreducible forward/backward compute (profile: `update()` ≈ 95% of runtime, ~67% of that is GNN forward+backward). **MPS was re-tested and is ~2.4–4× SLOWER** (tiny graph → many small ops → dispatch/sync overhead dominates). **Training is CPU-locked.** Per-episode ≈ 36 s (protagonist) / 24 s (antagonist); a 2000-ep run ≈ 16 h.
4. **Two more reward iterations (both stable, neither made the protagonist learn):**
   - `protag_signal_rebalance` (`reward_scale` 0.001→0.01, antagonist `target_entropy` 0.98→0.5·lnN): fixed an antagonist α climb, but delivery stayed ~0.91.
   - `protag_reward_shaping` (`delivery_reward` 10→100, `remaining_demand_penalty` 0.5→0.05): made reward positive and Q in-range, antagonist healthy — but `Q_Spread` collapsed and delivery stayed ~0.91. **This is the run whose diagnostic produced the pivot.**
5. **ERB generator rebuilt.** The old `scripts/generate_erb.py` solved the *toy* graph with the *old* reward (16 transitions). New `scripts/generate_erb_osm.py` solves the *OSM* env in parallel (workers ≈ cores; ALNS converges by ~150 iters on this graph; MPS is irrelevant — ALNS is pure-Python). Produced **`data/erb_transitions_osm.pt` = 42,020 transitions (~1.3 GB)** under the (now-superseded) shaping reward.
6. **The pivot:** diagnosed the inconsequential-decision problem, designed the redesign with Kilian (`PROBLEM_REDESIGN.md`), killed the old run.

**Lessons that shaped how we work (now in `SYSTEM.md`):** diagnose with data not assertions (we read tfevents directly, profiled, benchmarked); don't trust early/noisy curves (we twice misread a delivery-rate peak as a trend — use windowed means, not TB smoothing); report failures and self-corrections plainly; verify before destructive ops (we accidentally clobbered a checkpoint once).

---

## 4. Codebase state — what is stable and carries over
All of the following are **correct and reusable** for the redesign (they are problem-agnostic):
- SAC math (`sac.py .update()`): correct alpha sign, gradient clipping, SMDP `γ^dt` discounting, batched encoder, `Q_Spread` logging.
- ATLA trainer (`sacred_atla.py`): alternating phases, freezing, run-specific full-state checkpoints, TensorBoard logging.
- GATv2 nets (`networks.py`): encoder/head split, equivalence-tested.
- Featurization cache + checkpoint hygiene.
- ERB pipeline (`generate_erb_osm.py`) — though for a *dynamic* problem the inner solver must change from static ALNS to a dynamic dispatcher (see `PROBLEM_REDESIGN.md` §5).
- Tests: `tests/` (40 passing), incl. `tests/test_batched_equivalence.py`.

**What is OLD-PROBLEM-specific and will be replaced/augmented by the redesign:** the static demand model in `graph_env.py`, the reward in `smdp_wrapper.py::_accumulate_step` (becomes latency-based), and the protagonist action = next-node-destination (gains an assignment component at multi-truck stages). **Per Kilian's decision, build the dynamic env ALONGSIDE the existing one — do not delete the static path; it is the documented baseline + ZST reference.**

---

## 5. Standard operations

### 5.0 `train_sacred.py` CLI flags (the "tags" appended to a run) — definitive list
Every training run is `PYTHONPATH=. python scripts/train_sacred.py <flags>`. The flags:

| Flag | Type / default | Meaning |
|---|---|---|
| `--problem` | `osm`\|`stage0`\|`assign` (default `osm`) | **Which problem.** `osm` = static-demand Kaliningrad baseline (4 trucks, destination mode, legacy reward). `stage0` = single-truck next-hop route-choice validation rung (latency reward, `routing_mode=next_hop`, slack 1.2, corridor 14→82). `assign` = 3b multi-truck assignment probe (n=2 depots 110/135, contested demand, destination mode, latency reward). Each branch sets its **own `SMDPConfig` + `reward_scale`** internally (stage0/assign use 0.1; osm 0.01) and auto-disables ERB preseed for stage0/assign. |
| `--episodes` | int (default 1) | Total training episodes. Typical full run: 1000. |
| `--switch-every` | int (default 5) | ATLA phase length: train one agent N episodes while the other is frozen, then swap. Project standard = 50 (→ 10 co-evolution cycles over 1000 ep). |
| `--batch-size` | int (default 32) | SAC minibatch size for `update()`. |
| `--hidden-dim` | int (default 64) | GATv2 hidden width. |
| `--device` | `cpu`\|`mps`\|`cuda` (default cpu) | **Always `cpu`** here — MPS is ~2.4–4× slower for this small-graph GNN (re-confirmed). |
| `--tag` | str (default `sacred_atla`) | Free-form run label. Run dir = `<tag>_<ep>ep_sw<switch>_b<batch>_<timestamp>` under `logs/tb_runs/` and `models/runs/`. Use a descriptive, unique tag per run (e.g. `assign_erb`). |
| `--eval-every` | int (default 100) | **stage0/assign only:** run the learned-vs-greedy 4-cell eval every N episodes, logged under `Eval/*` (`gap_atk`, `gap_noatk`, …). **Set to 50** for an unbiased read (eval-every-100 with switch-50 aliases to *only* post-antagonist phase-ends → pessimistic). 0 disables. |
| `--erb-path` | path (default None) | Seed the protagonist replay buffer from a `.pt` of pre-generated `SMDPTransition`s (e.g. `data/erb_assign.pt` from `generate_erb_assign.py`). Demos are already correctly formatted (no compat shim). Overrides `--preseed-buffer`. |
| `--preseed-buffer` | bool (default True) | **Legacy** path: auto-loads `data/erb_transitions.pt` (the stale 16-transition toy ERB) + runs `generate_erb.py` if missing. Auto-forced `False` for stage0/assign. Prefer `--erb-path` for new ERB. |
| `--resume-checkpoint` | dir (default None) | Resume from `models/runs/<run>/` (loads protagonist+antagonist `checkpoint.pt`, continues from saved episode). |
| `--log-dir` | path (default `logs/tb_runs`) | TensorBoard root. |

**Key TensorBoard scalar tags** (read via `event_accumulator`, windowed means): `Episode/Delivery_Rate`, `Episode/Total_Wait`, `Episode/Mean_Latency`, `Value/Protagonist_Q_Spread`, `Value/{Protagonist,Antagonist}_{Q,Entropy}`, `Params/*_Alpha`, `Loss/*_Critic`, and (stage0/assign) `Eval/{greedy_atk,learned_atk,gap_atk,gap_noatk,...}`.

- **Train (old problem, baseline):** `PYTHONPATH=. python scripts/train_sacred.py --device cpu --episodes 2000 --switch-every 50 --batch-size 32 --hidden-dim 64 --tag "<tag>" --preseed-buffer False 2>&1 | tee logs/<tag>.log` (single-line `&&`-chained with `cd` + venv for a fresh tab — see SYSTEM.md; Kilian's Mac never sleeps, so no `caffeinate`).
- **Read run metrics directly (preferred over screenshots):** load the tfevents with `tensorboard.backend.event_processing.event_accumulator`; use **windowed means** for trends. Key tags: `Episode/Delivery_Rate`, `Value/Protagonist_Q_Spread`, `Value/*_Entropy`, `Value/*_Q`, `Params/*_Alpha`, `Loss/*_Critic`.
- **Generate ERB:** `PYTHONPATH=. python scripts/generate_erb_osm.py --episodes 34 --iterations 150 --workers 9 --out data/erb_transitions_osm.pt` (workers = cpu_count−1; M4 = 10 cores, 4 P + 6 E).
- **Pause/resume training:** `Ctrl+Z` then `fg` (lossless if the machine stays on); or `Ctrl+C` (durable, loses ≤ one phase-switch's worth, ~50 episodes — checkpoints are written per phase switch to `models/runs/<run>/`). Resume: `--resume-checkpoint models/runs/<run_name>`.
- **Hardware:** M4 Mac, 10 cores (4 performance + 6 efficiency), 24 GB RAM. CPU-locked for training.

---

## 6. Known debt / quirks
- **`--erb-path` flag is missing (TODO).** `train_sacred.py` preseed hardcodes `data/erb_transitions.pt` (the stale 16-transition toy ERB); it cannot load `data/erb_transitions_osm.pt` until a flag (or file swap) is added.
- **ERB rewards are stale.** `data/erb_transitions_osm.pt` was generated under the shaping reward; the redesign's latency reward will require regenerating it (and swapping ALNS for a dynamic dispatcher).
- **MPS is ruled out** for this workload (re-confirmed ~2.4–4× slower post-batching). CPU only.
- **Topology/determinism dogmas** (precomputed components, `sorted(...)` over unordered sets, action-mask filters isolated nodes to avoid `nx.NetworkXNoPath`) — see SYSTEM.md.
- **Antagonist action space** is a flattened `Discrete(E·L + 1)` (edge × level + wait), not MultiDiscrete.
- **Old toy ERB** `data/erb_transitions.pt` (48 KB) is stale (toy graph, old reward) — ignore it.

---

## 7. Key files & artifacts (pointers)
- `PROBLEM_REDESIGN.md` — the forward design (the redesign + curriculum + §7.1 Stage-0 spec). **The most important doc for what to build next.**
- `TASK.md` — the concrete Stage-0 build plan.
- `SYSTEM.md` — how to operate (behavioral + coding dogmas + this session's lessons).
- `../../MT_Literature_Survey_Kilian_Schwarz_split.pdf` — the academic grounding (read for the theory: SDVRP, SAC, RARL/ATLA, ERB bootstrapping, SBO, ZST).
- `data/erb_transitions_osm.pt` — 42k ALNS demos (stale reward; for reference / future regen).
- `logs/tb_runs/` — run histories (the diverged `...0614`, `protag_signal_rebalance`, `protag_reward_shaping`).
- Memory: the prior agent kept notes under `~/.claude/projects/.../memory/` (divergence cause, perf ceiling, command style).
