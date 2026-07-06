# TASK.md — state record + plan

> **⚠️ 2026-07-06: THE ACTIVE PLAN IS NOW `ROADMAP.md`** (contested-resupply redirection; see
> `DIRECTION.md`). Everything below, including the 2026-07-06 "campaign complete" banner, is the
> historical record.

> **⚠️ REFRAMED 2026-07-02 (Kilian + Fable) — the plan below this banner is HISTORICAL.**
> The critique (`CRITIQUE.md`) was accepted: the headline is now the **robustness claim**
> (SACRED/ATLA vs an identical but non-adversarially-trained vanilla SAC, under a held-out attack
> portfolio; greedy demoted to reference line). Decisions: D1 reframe ✓ · D2 dynassign pilot
> first, headline matrix on the FIXED hybrid · D3 vanilla-SAC control only · D4 descoping deferred.
> **Active plan + pre-registered metric: `experiments/gen03_robustness_dynassign.md`** (Phase 1).
> Hybrid fixes landed 2026-07-02 (zombie-orbit bug, goal observability/13-dim features,
> info-parity ETAs, narrow-checkpoint slicing — commit `d2b065b`, 82 tests green); post-fix probes:
> greedy no-attack 902→847, episodes end ~tick 220 (was always 1500), 1-gateway blockade floor
> +10.4%, scripted route-reach attack +40…+184% over budget 250…4000 (Phase-3 training budget
> recommendation: ~1500). New infra: `--vanilla`, `--train-antagonist-only --protagonist-snapshot`,
> `src/baselines/attackers.py`, `scripts/evaluate_portfolio.py` (paired portfolio + validation/test
> attacker split + `--select-best`).
>
> H7 (training the hybrid as previously specced) is SUPERSEDED — it must never run without these
> fixes; the hybrid now enters as the Phase-3 robustness-matrix arena.
>
> **Update 2026-07-04:** gen03 pilot = pre-registered null (ATLA ≈ vanilla; learned adversary
> weaker than random); gen04 gate = FAIL with motion observability (entropy pinning diagnosed) →
> co-evolution parked. **Phase 3 = `experiments/gen05_hybrid_matrix.md`: {vanilla,
> scripted-adversarial} × portfolio on fixed hybrid, budget 1500.** Back pocket: ATLA rider arm,
> gen04b entropy re-gate. Full narrative: `SACRED_PROGRESS.md`.
>
> **FINAL UPDATE 2026-07-06 — CAMPAIGN COMPLETE; see `HANDOVER.md`.** gen05 = competence-void
> (neither arm learned the hybrid; ceiling compression); gen06 (dynassign, competence gate PASSED
> at +5.5–7.0% of greedy) = **primary significantly reversed: adversarial training worsens
> held-out robustness** (dD_targeted −881 ± 284, 0/3 pairings; ranking greedy > vanilla >
> adversarial). Open decisions with Kilian: freeze-and-write (recommended) vs option-(b) hybrid
> learnability stretch; supervisor conversation (D4 descope: ERB/SBO/ZST/ALNS); thesis-planner
> launch. Remaining chores: env/envs merge (post-freeze), visualiser dims touch-up if figures
> need old checkpoints.
>
> **Deferred cleanup TODOs (approved 2026-07-04, blocked on running jobs / campaign freeze):**
> ~~(1) AFTER gen05 training completes: remove the legacy `--preseed-buffer` path~~ **DONE
> 2026-07-04, commit `d6982bb`** (test_erb needed no trim — already file-independent). (2) AFTER the experimental
> campaign freezes (~Jul 16-18): merge `src/env/` vs `src/envs/` naming (mechanical refactor,
> one commit, suite-guarded). See commit `f92f88d` for the full cleanup record.

> **(Historical handoff note, 2026-07-01):** the Stage-2 H-steps below were the prior plan, resting
> on the (unproven) interpretation that "next-hop routing is the missing lever". H1–H6 built; H7
> deliberately not run. History: Stage-0 (near-wash) → static-3b (claimed milestone, RETRACTED) →
> Stage 1.5 dynamic assignment (2 seeds, near-wash) → Stage 2 hybrid. Full record in `CONTEXT.md` §2.

## Where we are (one paragraph)
**Three rungs have now near-washed** against reactive greedy — Stage 0 (routing), static-3b
(assignment), Stage 1.5 dynamic assignment (`gen02_dynassign`: best-checkpoint `gap_atk` within
±~1000 noise, reliable ~6% static loss, antagonist runaway). Diagnosis is **structural**: every
**destination-mode** rung auto-routes, so the antagonist only degrades service *rate* and the
protagonist can't beat greedy's reactivity. The missing lever is **next-hop routing** — the policy
commits to roads, so congestion becomes a *learned, exploitable* decision. **Stage 2 = hybrid
(assignment + routing).** Graph-theory chokepoint analysis (`scratch/find_chokepoints.py`,
`chokepoints.png`): no critical bridges; the map is dominated by the **node-0 hub** with strong
routing chokepoints `('0','1')` (betw 0.23, detour 8.9×) and `('0','129')` (detour 18.9×). Geometry
search (`design_hybrid_geometry.py`, `hybrid_geometry.png`) picked **depots 110/135 + demand
`(78,130,49,224,48,17,47,46)`** so Depot A's routes funnel through `('0','1')` and Depot B's through
different gateways → both levers live. Searching depots too gave only marginal/degenerate gains →
**keep 110/135.**

## STAGE 2 — HYBRID (assignment + next-hop routing) — the plan

**Locked geometry:** depots 110/135 · demand `(78,130,49,224,48,17,47,46)` · capacity-1 · latency.
**Decisions (resolved 2026-06-30):** **D1** antagonist reach = **route-reach** (keep `leashed` as a
comparison arm in H7) · **D2** = **static demand first** (Poisson later) · **D3** thesis framing =
**parked until after the run** · **D4** = **2 seeds**.
**Antagonist config (locked):** full-block `(1.0,)` · **1 block / decision event** ·
`antagonist_interval=25` · `congestion_duration=125` (= 5 intervals → a block expires *on* a decision
event = seamless renew/release; up to ~5 concurrent blocks = strong side, tunable down later).

- [x] **H1 — Hybrid env mechanic (`routing_mode="hybrid"`)** *[done 2026-07-01, no CPU, +2 tests,
      suite 71 green]*. The truck state machine:
  - `TruckState.assigned_target` (graph_env) — set by an assignment decision; env flips it to
    `home_depot` on serve and to `None` on reload (naturally no-op in other modes since it stays
    `None`). `_serve_demand` gated to serve **only the assigned target** (no opportunistic serving).
  - Wrapper `routing_mode="hybrid"`: `_hybrid_action_mask` branches per idle truck — no target +
    load → **assignment** (`_assignment_candidates` = pending requests); has target → **routing**
    (`_forward_mask` corridor toward `assigned_target`, extracted + shared with next-hop). `step_
    protagonist` splits actions (set target vs `dispatch_truck_edge`); `_auto_resolve_forced_moves`
    /`_current_decision_type` handle both types (assignment due at ≥1, routing at ≥2). Policy head
    unchanged.
  - Tests: full cycle (assign → route → serve → return → reload → reassign) delivers all demand;
    no opportunistic serving; destination/next_hop/dynamic modes still green.
- [x] **H2 — Hybrid factory + entry point** *[done 2026-07-01, no CPU + tests]*.
      `make_hybrid_assign_env` (locked geometry, static) + `--problem hybrid` in `train_sacred.py`
      (SMDPConfig: hybrid, slack 2.0, full-block, cap 1, interval 25/dur 125, `max_ticks=1500`).
      **Diagnostic finding:** the route-level detour around `('0','1')` is only **1.06–1.40×** (the
      edge-level 8.9× was misleading — short edge, parallel path), so slack 2.0 **lets trucks route
      around** — with `('0','1')` permanently blocked, greedy still delivers all 8, **0 ticks
      trapped**. Routing lever is live; the anticipation headroom is in the *timing* game (H3/H7).
- [x] **H3 — Antagonist: route-reach + aligned timing** *[done 2026-07-01, no CPU, +2 tests, suite
      74 green]*. New `SMDPConfig.antag_reach`: `"leashed"` (3-hop, default, kept for the H7
      comparison) vs `"route"` (= exactly the edges on each truck's **static shortest path to its
      target** → pre-block the gateway *ahead*; a truck with no target isn't committed and adds no
      edges). Static path = the truck's *intended* route, so a block forces a *detour* rather than
      the adversary chasing reroutes. Hybrid config uses `antag_reach="route"` + full-block, 1/event,
      interval 25 / duration 125. Tests: route-reach is exactly the route edges incl. `('0','1')` and
      surfaces in the mask; greedy vs route-reach antagonist runs, lands blocks (budget spent), still
      delivers all (routes around).
- [x] **H4 — Hybrid greedy baseline** *[done 2026-07-01, no CPU + tests]*. `hybrid_greedy_policy` in
      `greedy_dispatch.py`: assignment = nearest unclaimed request (sequential claiming); routing =
      forward next-hop on the congestion-aware shortest path to the assigned target. Reactive
      (reroutes around visible congestion, can't anticipate). Delivers all 8 on the geometry.
- [x] **H5 — Light headroom check** *[done 2026-07-01, light CPU]* (`scratch/hybrid_headroom.py`,
      static/deterministic). **Strong positive:** the **route-reach antagonist inflates greedy's
      latency +79%** (902→1615) vs only **+3.2% leashed** — the reach relaxation gives real leverage
      and ~79% recoverable headroom (dynassign's attack was only ~5%). **Nuance:** a static
      "avoid-the-gateways" heuristic *can't* recover it — the gateways are on the **critical path**
      (demand only reachable through them), so you can't statically dodge; the anticipation must be
      **dynamic** (timing / unpredictability / assignment-balancing) — exactly what RL might learn
      and greedy/static-heuristics can't. Genuine train-and-see, but far more promising than
      dynassign. (Risk: the strategy may not exist if the adversary follows too well → H7 tells.)
- [x] **H6 — Eval + checkpoint infra** *[done 2026-07-01, no CPU, +1 test]*. `scripts/evaluate_hybrid.py`:
      `sac_hybrid_policy` (drives both decision types, claiming for assignment only), `eval_hybrid_cells`
      (learned vs greedy × no-attack/fixed-antagonist, single deterministic episode — static, so
      **best-checkpoint is unbiased**, no max-over-noise), `select_best_checkpoint`. Per-phase
      snapshots already saved by the trainer. Eval_fn wired into `--problem hybrid`.
- [ ] **H7 — Train + diagnose** *[real CPU]*. Seeded generation, **2 seeds** (3rd only if promising),
      static demand. Headline = best-checkpoint `gap_atk` vs the fixed adversary (mean±std), plus the
      **route-reach vs leashed** comparison (does anticipation reach pay?), plus the post-hoc headroom
      diagnostic. If it holds, *then* add Poisson (D2) and re-run. Ledger pins SHA + decision metric.

## RETRACTION — static 3b "milestone" does not survive a windowed read (2026-06-29)
- `gap_atk` over the 20 eval points of `assign_probe_claimfix`: mean **+18** (a loss; ≈+2%),
  range **[-188, +170]**, 6/20 wins. The famous **−56 final / −188 best** are the favourable
  *tail* of a noisy series, not a regime. (`assign_probe` broken run mean +190 confirms the
  claim-fix helped — but "better than a buggy run" ≠ "beats greedy".)
- `gap_noatk` is the trustworthy signal: **≈+8% loss, every checkpoint, low variance** — RL
  reliably makes *worse* assignments than greedy unattacked.
- **The eval metric is mis-specified:** `eval_cells_assignment` is a **single deterministic
  episode** of learned vs the **current co-evolving** antagonist. The ±100 swing is *arms-race
  timing*, not robustness. The RARL-correct test is a **frozen** protagonist vs a **fixed/held-out**
  antagonist, averaged over **multiple instances**.
- **Checkpoints were not saved per-phase** (only final `checkpoint.pt` is kept, overwritten), so
  a best-checkpoint fixed-adversary re-eval of the existing run is **impossible** → retire on the
  evidence above; don't burn CPU re-running the weakest ("shrunk-until-RL-wins") static story.
- **Keep the static-3b code** (`make_assignment_env`, `--problem assign`) as baseline / ZST ref.

## THE PLAN — Stage 1.5: multi-truck assignment + Poisson (measured correctly)
Routing stays **deferred** (destination-mode, env auto-routes via Dijkstra); hybrid next-hop+assign
is the later true Stage 2. Antagonist leverage here = congestion drops service rate μ → with fixed
λ the backlog **compounds** → protagonist must counter via *assignment* (greedy-insertion is myopic
about the compounding → headroom hypothesis, tested by the Step-4 gate).

- [x] **Step 1 — Poisson demand in the env** *[done 2026-06-29, no CPU, +7 tests]*. Additive
      dynamic path in `graph_env.py` (static path untouched): seeded Poisson arrival schedule
      injected into the existing per-node `demand` machinery, FIFO `arrival_tick` queue per node,
      `is_done` terminates on `max_time` only (no early stop in lulls). `make_dynamic_assign_env`
      + `poisson_arrival_fn` (contested 8-node band as K=1 hotspot). Tests pass (incl. the
      latency-**telescoping invariant** Σ_t remaining == Σ_req(T−arrival+1)); full suite 62 green.
      **Two findings:** (a) the latency reward needs **no change** — the existing potential-based
      `−remaining_demand`/tick auto-anchors to arrival because a unit only enters `remaining_demand`
      when it arrives; (b) the **existing greedy-insertion policy drives the dynamic env unmodified**
      (smoke: 400-tick episode, ~20 arrivals, 15 delivered, mean latency ~43) → Step 3 shrinks.
- [x] **Step 2 — Queue observation + metrics** *[done 2026-06-29, no CPU, +2 tests, suite 64 green]*.
      `node_in_dim` 9→11; the policy now sees the two signals greedy is blind to (request age +
      congestion-aware ETA). End-to-end ATLA smoke (4 episodes, both phases, antagonist active)
      passes; dynamic delivery-rate now computed correctly (was a broken 0%).
  - [x] **2a. Observation exposes dynamic features.** `graph_env.observe()` (dynamic-gated) adds
        `node_waits` `{node: oldest_wait}` + `truck_etas` `{truck: {node: congestion-aware dist}}`
        (idle/at-node trucks → demand nodes + home depot). Versioned single-source-Dijkstra cache
        (`_congestion_version` bumped in `set_congestion`) keeps ETAs current and cheap.
  - [x] **2b. featurize adds 2 columns.** `featurize_state` appends `[oldest_wait/100, eta/50]`,
        reads `node_waits`/`truck_etas` (default 0 → static = zeros). `NODE_FEATURE_DIM = 11`.
  - [x] **2c. Bumped `node_in_dim` 9→11** at all defaults + call sites (networks/sac/train_sacred/
        evaluate_*/spar_visual*/run_sbo/tests). ⚠️ old 9-dim checkpoints (retired/done rungs) won't
        load — accepted; code paths retrainable.
  - [x] **2d. Episode metrics.** Trainer branches on `_dynamic_demand`: fixes Delivery_Rate for
        dynamic + logs `Episode/Mean_Delivered_Latency`, `Episode/Final_Queue`, `Episode/Num_Arrivals`.
  - [x] **2e. Tests + suite green.** `featurize` shape →(N,11), columns populated (dynamic) / zero
        (static); batched-equivalence re-passes; `test_networks` (9,11) updated. **64 passed.**
- [x] **Step 3 — Dynamic baseline + correct eval + checkpoint-keeping** *[done 2026-06-29, no CPU,
      +2 tests, suite 66 green]*. End-to-end smoke via the real `--problem dynassign` entry point
      (training + per-phase snapshots + multi-seed eval + select-best) passes.
  - [x] **3a. `--problem dynassign` entry point** in `train_sacred.py` (`make_dynamic_assign_env`,
        latency/destination, `--arrival-rate`, reproducible per-episode demand seed) + eval_fn wired.
  - [x] **3b. Fixed `run_episode` for dynamic** — counts actual arrivals (`delivered + remaining`),
        adds `mean_delivered_latency`. (Greedy-insertion already did sequential claiming.)
  - [x] **3c. Per-phase checkpoint snapshots** — trainer writes protag+antag actor snapshots to
        `models/runs/<run>/snapshots/{protagonist,antagonist}_ep<N>.pt` each phase switch.
  - [x] **3d. Multi-seed fixed-adversary eval** (`scripts/evaluate_dynamic_assign.py`):
        `eval_dynamic_cells` — learned & greedy vs the SAME fixed antagonist over N fixed Poisson
        seeds → mean±std gaps. Wired as the in-training eval_fn (seeds 0–2). **The metric the
        retraction demands.** Also fixed the trainer eval-print to accept mean/std keys.
  - [x] **3e. Best-checkpoint selector** — `select_best_checkpoint` / `--select-best`: scans
        `snapshots/`, evals each protag vs the fixed final antagonist over N seeds, reports best±std.
  - [x] **3f. Tests + suite green** — dynamic `run_episode` metrics + eval determinism/structure;
        **66 passed**.
- [x] **Step 4 — Headroom gate** *[done 2026-06-29, light CPU, +1 test, suite 67 green]*. Built
      urgency dispatcher + clairvoyant ceiling + λ/budget sweep. **Result is nuanced — DECISION
      POINT (with Kilian) before Step 5.**
  - [x] **4a. Urgency dispatcher** (oldest-first, congestion-aware) in `greedy_dispatch.py` + test.
  - [x] **4b. Clairvoyant ceiling** — perfect-foresight free-flow 2-truck scheduler, horizon-truncated.
  - [x] **4c/4d. Gate run** (`scratch/dynassign_headroom.py`), T=800, 8 seeds:
        | λ | drate | greedy_no→at (attack cost) | urgency | clair_gap (free-flow) |
        | 0.025 | 0.95 | 740→763 (+3%) | −0.5% (loses) | 67% |
        | 0.040 | 0.89 | 1815→1902 (+5%) | −5% (loses) | 38% |
        | 0.060 | 0.71 | 5953→6282 (+5.5%) | −7% (loses) | 10% |
        | 0.080 | 0.55 | 11187→11487 (+3%) | −5% (loses) | 4% |
        **Findings:** (1) urgency LOSES at every load → no simple-heuristic headroom; (2) attack is
        weak at the static-3b budget (400) — but **scales with budget**: at λ=0.06, attack cost
        +5.5%→+11%→+26% for budget 400→1500→4000 (the default budget is tuned for 8 static
        requests, too small for 40+ dynamic ones); (3) the clairvoyant gap is mostly free-flow
        optimism, not adversarial. **Net: real adversarial headroom only appears with a budget
        scaled to the dynamic load, and achievability via assignment-only is still unproven.**
  - **DECISION (Kilian): Step-5 fork** — A) train destination-mode at a scaled budget (~3–4k) as
    the definitive seeded test; B) pull next-hop routing forward (true Stage 2 hybrid) where the
    adversary's leverage is direct; C) strengthen the gate (congestion-aware rollout clairvoyant)
    to measure *achievable* headroom first. ρ≈1 operating point ≈ **λ=0.06** (drate 0.71, queue 14).
- **Antagonist redesign + 2 latent bug fixes (2026-06-29).** The first launch (budget 4000, 4
  levels, no cap) was **compute-infeasible**: the antagonist spent its budget via ~133 congestion
  sub-actions/episode, each an SAC update → **antagonist phase 295 s/ep vs protagonist 45** (~47 h
  projected). Fix = **full-blockage only** (`congestion_levels=(1.0,)`) + **cap 1 roadblock per
  decision event** (`max_antag_actions_per_event=1`, new SMDPConfig field) + **sustained**
  (`congestion_duration=120`, `antagonist_interval=25`) → ~32 strategic roadblocks/episode. Re-gate:
  leverage **preserved/stronger** (+8% heuristic at λ=0.06 vs +5.5%). **Found+fixed 2 latent bugs**
  the (1.0,) config exposed: the antagonist's level value was **hardcoded `[0.25,0.5,0.75,1.0]`** in
  `select_action` (→ picked 0.25, mask rejected it, **budget stayed 0 = no adversary**) and again in
  the **update parse** (wrong level-index → **IndexError crash** at the antagonist phase). Both now
  use `self.congestion_levels`; regression test added. **Re-timed (working adversary): both phases
  ~18 s/ep** (was 295). Tests 69 green.
- [ ] **Step 5 — First seeded dynamic generation** *[real CPU, responsible]*. Config locked:
      full-block antagonist (above), `--problem dynassign --arrival-rate 0.06`, per-phase snapshots,
      multi-seed fixed-adversary eval. **Measured ~18 s/ep single-process** → 2 seeds ‖ 1000 ep
      ≈ **~7.5 h** (800 ep ≈ ~6 h); 3 seeds ≈ ~10 h. Aggregate mean±std; ledger pins SHA + the
      decision metric (best-checkpoint `gap_atk` mean<0 vs fixed adversary). **Awaiting launch go.**

**Proposed design params (NOT locked — change any):** capacity 1 · fixed-horizon termination ·
destination-mode (routing deferred) · n=2 depots · small K (start concentrated) · λ set empirically
by the Step-4 gate to ρ≈1 · antagonist reined.

## Instability + measurement fixes baked in (were the "fork B")
Best-checkpoint selection (Step 3), fixed-adversary multi-instance eval (Step 3), reined antagonist
+ earlier stop (Step 5), per-phase checkpoint-keeping (Step 3). These travel with the new rung so the
first dynamic run is measured right — not retrofitted after a noisy result.

## How to run things (current infra — USE IT)
- **Single run:** `PYTHONPATH=. python scripts/train_sacred.py --problem <dynassign> --episodes 1000
  --switch-every 50 --batch-size 32 --hidden-dim 64 --device cpu --eval-every 50 --seed <k>
  --group <gen> --tag <tag>` (new `--problem` value added in Step 1–3).
- **A generation (parallel seeds):** `scripts/run_generation.py --group <gen> --configs <...>
  --seeds 0,1,2 --threads 3 --max-concurrent 3` (≤10 cores; writes `experiments/<gen>.md`).
- **Aggregate:** `scripts/aggregate_generation.py --group <gen> --plot` → mean±std + verdict.
- **Pause/resume losslessly:** `kill -STOP <pid>` / `kill -CONT <pid>` (Mac stays on); checkpoints
  carry the full replay buffer → `--resume-checkpoint` is lossless.
- **Read run metrics directly:** `event_accumulator` + **windowed means** (never the final point —
  that is exactly what burned the static-3b "milestone").
- **GATE BEFORE TRAINING:** always run the Step-4 headroom probe before a multi-hour run.

## Definition of done (for the rung)
- Any "RL beats greedy" claim is **seed-averaged (≥3), mean±std**, on the **fixed-adversary,
  multi-instance, best-checkpoint** metric, decision metric fixed in advance (ledger). Single runs
  and final-checkpoint numbers are anecdotes.
- `PYTHONPATH=. pytest tests/` green (currently 55) — paste raw output after touching agents/env.

## Explicitly later (curriculum / supervisor agenda — don't start without Kilian)
True Stage 2 hybrid assignment+routing (the multi-headed action head); K×ρ×budget sweeps; **ZST**;
Obj-3 dynamic-dispatch ERB / rolling-ALNS (also Obj-5 SOTA baseline); Obj-4 SBO. Held low-severity
bug/perf fixes: `CONTEXT.md` §2 "Held / known issues" + `docs/archive/MASTER_AUDIT.md`.

**Visualiser idea (Kilian, for later):** in the dynamic visualiser, colour each demand node by
*wait time* — light orange → dark orange the longer it has waited (a direct visual of the Step-2
request-age feature). Build on `scripts/spar_visual_stage0.py`. Not now.
