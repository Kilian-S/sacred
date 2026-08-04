# Problem Redesign — Draft for Review

> **Status:** DRAFT for Kilian's review (+ supervisor discussion). Not yet canonical.
> If approved, fold the agreed parts into `CONTEXT.md` (architecture/state) and `TASK.md` (actions).
>
> **IMPLEMENTATION STATUS (2026-06-29) — read `CONTEXT.md` §2 for the live record:**
> - **Stage 0 (§7.1) — DONE & VALIDATED.** Realised as a single-truck **next-hop route-choice** rung (not just destination-choice — the antagonist needs a *learned* routing decision to attack). Learns a consequential adversarial policy; **matches** greedy (route-choice vs reactive greedy is structurally near-a-wash → little headroom).
> - **Stage "3b" assignment probe — BUILT, MILESTONE.** Jumped early to the **assignment** lever (Option 3's core) because that's where beat-the-baseline headroom provably lives (~0% static / **8–17% adversarial**). n=2 trucks/depots, destination-mode assignment, latency reward, greedy-insertion baseline. **First time RL beats a classical baseline here** (−6.5% under attack). Static demand for now (Poisson deferred).
> - **Open problem:** late **co-evolution instability** (antagonist runaway); ERB-seeding (Obj 3 pilot) inconclusive. See `TASK.md` fork.
> - **Locked decisions:** latency reward; fixed-horizon termination; headline = hybrid assignment+routing; headline demand = Poisson; baselines staged greedy→rolling-ALNS; SBO deferred. §7 agenda otherwise still open.

---

## 1. Why redesign (the evidence)

Three full runs + the `Value/Protagonist_Q_Spread` diagnostic converge on one conclusion: **on the current problem the per-step routing decision is near-inconsequential**, so the protagonist cannot learn to beat a naive policy.

- `Q_Spread` (max−min Q across allowed destinations) **collapsed 5.3 → 0.46** over training: the critic correctly learns that destinations are near-equivalent in value.
- Delivery rate sits **flat at ~0.91** across all runs (reward shaping, alpha fixes, etc. did not move it).
- Root cause is **problem structure, not a hyperparameter**: demand is spread across **95/290 nodes (33%), ~1–2 packages each**, so from most positions several nearby destinations are equally good. With 4 trucks / 600 ticks, ~0.91 is likely the **time/capacity-constrained ceiling** any reasonable coverage policy reaches.
- Corollary: with demand everywhere, the **antagonist is also toothless** — congesting one edge just sends a truck to a different nearby demand node.

**Design imperative:** make decisions *consequential* (so Q discriminates and RL can beat a baseline) **and** make the antagonist *matter* (so the adversarial co-evolution — the actual SACRED thesis — has depth), **without** sacrificing the five research objectives (zero-sum game, sim env, SAC+ATLA+**ERB bootstrapping via metaheuristics**, **SBO for facility/fleet**, **eval vs SOTA metaheuristics**, and the crown jewel: **Zero-Shot Transfer**).

---

## 2. The three candidate formulations (and verdict)

| Option | Description | Makes routing consequential? | Antagonist matters? | Preserves Obj 3/4/5? | ZST potential | Verdict |
|---|---|---|---|---|---|---|
| **1** | 1 truck, 1 demand/episode, ends on delivery | ✅ (every step changes time-to-target) | ✅ | ❌ VRP collapses to Dijkstra → ALNS/SBO/metaheuristic-eval vacuous | ⚠️ weak (just navigation) | **Debug/curriculum rung, not final** |
| **2** | 1 truck, capacity 1, dynamic demand stream, reload each trip | ✅ | ✅ | ⚠️ thin VRP, ALNS near-trivial | ⚠️ moderate | **Intermediate rung** |
| **3** | n depots = n trucks, dynamic demand, protagonist **assigns truck→demand *and* routes** | ✅ assignment + routing both consequential | ✅ congestion changes which truck is fastest *and* is routed-around | ✅ genuine dynamic multi-depot VRP | ✅ a real transferable *strategy* exists | **Headline problem** |

**Recommendation: anchor on Option 3, reach it via a curriculum starting near Option 1.**
Option 3 is the literature's favoured **assignment-based MDP** (assign customers to vehicles, sidestepping the route-based curse of dimensionality), it is genuinely *Dynamic* (the "D" in SDVRP that the current static-demand setup lacks), and it keeps every objective alive.

**Is the extreme (1 demand) a bad idea?** Not as a *debug/validation rung* — it gives the cleanest possible signal to prove the SACRED stack can learn an adversarial policy at all. But as the *final* problem it is too reductive (kills Obj 3/4/5, risks shallow co-evolution, and — being less realistic than clustered dynamic demand — invites the reviewer critique "you shrank the problem until RL won").

---

## 3. Proposed Option-3 environment

### 3.1 Entities
- **Graph:** unchanged (Kaliningrad OSM, 290 nodes / 412 edges), or a held-out graph for ZST tests.
- **Depots & trucks:** `n` depots, one truck home-based per depot (start `n=2–4`). Tunable; later SBO-optimised (ties to Obj 4).
- **Demand:** a **dynamic stream** of requests `(location, size, arrival_tick)` instead of all-at-once at `t=0`.

### 3.2 Demand-arrival process (the key change)
Requests are described by **two independent knobs — temporal (when) and spatial (where):**

- **Temporal — rate-based, *not* event-based.** Requests arrive on a **Poisson process** (rate λ) on the simulation's own clock, **decoupled from what the trucks are doing** (idle, mid-route, departing/returning to a depot). A customer orders on their schedule, not when a truck finishes — this is the standard SDVRP model the literature survey cites. This is deliberately *not* the "deliver-one → spawn-one" event-based model (Option 2), which keeps the active load artificially constant and self-limiting. Rate-based arrivals let the backlog **fluctuate and overload**, which is both realistic and the source of the interesting dynamics: because arrivals keep coming independently, **the antagonist's congestion makes the queue *compound*** (slowed trucks fall behind a steady arrival stream), giving the adversary real leverage that event-based arrivals would deny it.
- **Spatial — K hotspots.** Locations drawn from the demand heatmap, parameterised by **K = the number of demand clusters** (concentration). K=1 concentrates demand near one area (which truck/route there matters a lot → consequential); large-K/uniform is the current diffuse spread (soft decisions). K is the **spatial difficulty knob** (see §6).
- **Size:** start at **1** (capacity-1 trips, per your Option 2/3) → can raise for batching richness.

**Implication — a temporal difficulty axis falls out for free:** with rate-based arrivals the system is a queue, so its hardness is governed by **traffic intensity ρ = λ / (fleet service rate μ)**. ρ ≪ 1 → trucks easily keep up, queues empty, decisions barely matter (the same trap as diffuse demand). ρ ≈ 1 or > 1 → a backlog forms, the agent **must prioritise**, and the antagonist can tip the system into overload. ρ is the **temporal difficulty knob** (see §6).

### 3.3 Action spaces — hybrid assignment + routing (target), staged by the curriculum

**Decision:** the protagonist controls **both assignment (which truck serves which pending request) and routing (the path it takes)**. Two clarifications make this the right call:

- **Routing control is not new.** The *current* system is already routing-based (protagonist picks the next destination node per idle truck). Its failure was **not** the action type — it was that decisions were inconsequential (demand everywhere). In the redesigned problem both become consequential.
- **Assignment-only is a simplification, not a necessity.** The literature favours assignment-based MDPs only because *joint routing of many vehicles* hits the curse of dimensionality. For a **small fleet (n = 2–4)** hybrid is tractable and richer. Crucially, **routing control is the part the antagonist directly attacks** — if routing were automated, the policy would never *learn* to route around congestion, weakening the adversarial-robustness story (the thesis core).

**Dimensionality cost is moderate, not explosive:** the *per-decision* action space stays small (assignment ≈ #pending requests; routing ≈ node degree ~3). The real cost is more decisions per episode (~2–5×), two skills to learn, and longer credit chains — a manageable increase in sample complexity with the dense latency reward.

**The curriculum stages it for you (see §4):** with 1 truck (Stage 0) assignment is moot → it's *pure routing*; assignment layers on naturally as trucks are added (Stage 2). So you reach hybrid incrementally, never bolting on a multi-headed action space in one step.

- **Antagonist = congestion** (largely unchanged): pick edge + level under a budget. Now congestion *delays deliveries* on routed paths → directly increases the protagonist's latency cost, and (with rate-based arrivals) lets the backlog compound.

### 3.4 Reward (redesigned — this is the crux)
Switch from the absolute-backlog integral to a **delivery-latency** objective:
- Per delivered request: cost = **wait time** (`delivery_tick − arrival_tick`), or reward = `−latency`.
- Equivalently, penalise the **cumulative waiting time of outstanding requests** per tick (potential-based; telescopes to total latency).
- `antagonist_reward = −protagonist_reward` (zero-sum preserved).

Why this fixes the signal: the per-decision impact (which truck, routed around which congestion) **directly changes that request's latency** → clean credit assignment → `Q_Spread` stays > 0. The antagonist's congestion directly inflates latency → it matters.

**Headline metric becomes:** average delivery latency (or **% served within deadline**) — *as a function of antagonist budget* — RL vs ALNS-baseline vs non-adversarial-RL.

### 3.5 Episode termination
Time-based (`max_ticks`), since demand is a continuous stream. (Alternative: until `N` requests served — decide in §7.)

---

## 4. Curriculum (de-risks algorithm-vs-problem ambiguity; also stages the action space)
- **Stage 0 — validation (≈ Option 1):** 1 truck, 1 depot, `K=1`, single request / short fixed set (no Poisson yet). Action = **pure routing** (assignment moot with one truck). *Goal: not a result — prove the stack learns.* Success = `Q_Spread` stays high, entropy falls, protagonist beats a naive shortest-path baseline, antagonist co-evolves. If it fails *here*, the bug is algorithmic, not the problem.
- **Stage 1:** 1 truck, **Poisson** dynamic stream, small `K`. Still routing-only; introduces the temporal/queue dynamics (ρ).
- **Stage 2 — headline (Option 3):** `n` trucks/depots, Poisson stream, **hybrid assignment + routing**, full sweeps over the three difficulty axes (§6).

The curriculum itself is a defensible methodological contribution, and it introduces the action-space dimensionality (routing → + assignment) and the arrival complexity (fixed → Poisson) one rung at a time.

---

## 5. How ALNS / ERB (Obj 3) and baselines adapt
- Static ALNS no longer fits a *dynamic* problem. The expert/demonstrator becomes a **dynamic-dispatch heuristic**: greedy-insertion ("assign each new request to the truck that serves it soonest given current congestion"), or a **rolling/re-optimising ALNS / rollout dispatcher (RRD)** as in the surveyed literature.
- ERB bootstrap then seeds the protagonist with good *assignment* decisions — Obj 3 stays intact, just with a dynamic baseline. The existing `generate_erb_osm.py` machinery (parallel, demand-coupled) carries over; the inner solver swaps.
- Obj 5 ("beat SOTA metaheuristics") = compare trained RL vs this dispatch heuristic under attack.

---

## 6. Difficulty axes → turn the finding into *curves*, not a point
Rather than pick one "RL wins" setting (which a reviewer distrusts), sweep **three principled difficulty axes** and report measured RL-vs-baseline curves:
1. **Spatial — K hotspots** (concentrated → diffuse): *where* adaptive RL starts to matter.
2. **Temporal — traffic intensity ρ = λ/μ** (under-loaded → overloaded): *how hard the load must be* before prioritisation pays off.
3. **Adversarial — antagonist budget** (weak → strong): the robustness curve, RL vs non-adversarial-RL vs heuristic dispatcher.

Plus the headline **evaluation protocol — Zero-Shot Transfer (ZST):** train on one graph / demand distribution, evaluate zero-shot on a held-out one.

These axes *are* the experimental contribution; this project's flat-delivery result becomes the motivating evidence for studying *where* adaptive, adversarially-trained RL beats classical dispatch.

---

## 7. Open questions — supervisor agenda (NOT blockers for Stage 0)
These shape the *headline* (Stage 2) design; Stage 0 needs only the tiny subset in §7.1, so we can build and validate while these are discussed.

1. ~~Assignment-only vs + routing?~~ **Decided: hybrid assignment + routing, staged by the curriculum (§3.3, §4).**
2. **Capacity:** 1 (clean) vs larger (batching/VRP richness)?
3. **Reward:** raw latency vs %-within-deadline vs throughput? (Affects what "robust" means.) *Latency form approved as the basis — §3.4.*
4. **Arrival model:** Poisson rate λ and the target ρ regime; hotspot count/spread K; stationary vs time-varying λ?
5. **Termination:** fixed horizon vs N-requests-served?
6. **Baseline dispatcher:** greedy-insertion vs rolling-ALNS vs RRD?
7. **Depots/fleet:** fixed `n`, or SBO-optimised (Obj 4 hook)?
8. **Realism guard:** keep the design justifiable as *more realistic last-mile* (dynamic, clustered, meaningful disruption) — not "shrunk until RL wins."

### 7.1 Minimal decisions to *start* Stage 0 (already resolved)
Latency reward (§3.4, approved) · pure routing (1 truck → assignment moot) · 1 truck / 1 depot, `K=1` · single request or short fixed set (no Poisson yet) · time-based termination · shortest-path baseline. Everything else defers to the agenda above.

---

## 8. Implementation impact (high level — for a later TASK.md)
- **`graph_env.py`:** add a dynamic demand queue + Poisson arrivals (currently all demand at `t=0`); congestion-aware shortest-path routing already exists.
- **`smdp_wrapper.py`:** **reward = latency-based** (arrival→delivery wait); new decision-event triggers (request arrival / truck free); protagonist action = routing (kept — Stage 0) → + assignment (truck ↔ pending request) at Stage 2.
- **`networks.py` / `sac.py`:** GNN observation must encode pending requests + per-truck congestion-aware ETAs; the action head gains an assignment component at Stage 2 (the routing head already exists). Antagonist mostly unchanged.
- **`generate_erb_osm.py`:** swap static ALNS for the dynamic-dispatch heuristic.
- **Metrics/logging:** add latency, %-within-deadline; keep `Q_Spread`.

These are non-trivial (the env + protagonist action space are the big ones) but localised; the SAC/ATLA training machinery, the stability fixes, and the batched `update()` all carry over unchanged.

---

## 9. One-line recommendation
**Adopt Option 3 (dynamic multi-depot VRP, Poisson arrivals, hybrid assignment + routing, latency-based zero-sum reward) as the headline problem, reached via a curriculum from a single-truck routing-only validation rung, and report RL-vs-baseline across three difficulty axes — spatial K × temporal ρ × adversarial budget — plus zero-shot transfer.**
