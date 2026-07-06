# HANDOVER.md: master state & onboarding for the incoming agent (2026-07-06)

> **★ START HERE (new agent, 2026-07-06 end-of-session). READ ORDER for exact parity with the
> previous instance:**
> 1. **`REDESIGN_INTERDICTION.md`**: the north star: why the pivot was necessary (§0.5 full
>    evidence chain) + the equilibrium proof (§1).
> 2. **`ROADMAP.md` Phase I**: the build plan; I0/I1/I1b/I2 are DONE (see below), I3 is next and
>    detailed there.
> 3. **`experiments/gen08_interdiction.md`**: the pre-registration + the **G1 and G2 gate results
>    (both PASSED)**; the citable interdiction numbers.
> 4. **`THESIS_STORYLINE.md`**: the 4-act positive arc (the written-thesis spine).
> 5. Then history/bridge as needed: `DIRECTION.md` (why worst-case is the right register),
>    `experiments/gen07_contested_matrix.md` (the flat-landscape finding that forced the pivot),
>    `SACRED_PROGRESS.md` entries 11-12, `SYSTEM.md` (dogmas), and §1-5 of this file (the campaign).
>
> **BUILD STATE (branch `gen08-interdiction`, off `main`; suite 122+ green; the interdiction code
> is the live work):** DONE: I0 equilibrium oracle (`src/baselines/interdiction_oracle.py`);
> I1 env core + G1 gate (`src/envs/interdiction.py`, reproduces the oracle); I1b SAC-trainable env
> (make_interdiction_env + masks); **I2 feasibility slice PASSED (`scripts/train_interdiction.py`):
> the PROJECT'S FIRST POSITIVE RESULT: adversarial training cut interception 100% (shortest-path)
> -> 23% (SACRED), approaching the computed equilibrium 16.7%.** NEXT = I3: asymmetric instances
> (non-uniform equilibria) to separate SACRED from vanilla cleanly (the symmetric slice gave a thin
> sacred-vs-vanilla gap), + K/connectivity sweeps + seeds + learned-antagonist co-evolution.
> Key gotchas already paid for: SAC `reward_scale` default 0.001 is far too small (use ~1.0 with
> interception_loss ~10); best-respond to the defender's EMPIRICAL AVERAGE play (fictitious play),
> not the instantaneous policy, or it oscillates/chases. Kilian's decisions: Kaliningrad graph,
> single convoy first. Plan-first rule; Kilian owns CPU/launches.
>
> **⚠️⚠️ CURRENT DIRECTION (2026-07-06, latest): THE INTERDICTION-GAME REDESIGN.**
> Read **`REDESIGN_INTERDICTION.md` FIRST**: it is the north star. Short version: the campaign
> (gen03-06) and the exploitability follow-up (gen07) established that adversarial RL cannot win
> with a *congestion* adversary, because congestion is observable/reroutable/reversible, giving a
> reactive-dominated, FLAT attack landscape (proven: the corrected best-response gate lands at
> 0.35× random; every block is equally damaging). The fix is to change the ADVERSARY, not tune the
> old one: model Application 1's real threat, **interdiction/ambush** (hidden, irreversible,
> pre-committed), which is a **Stackelberg security game** where a deterministic router is
> maximally exploitable and the minimax **mixed strategy** (which SAC's entropy produces) provably
> cuts interception. **Proven at the equilibrium level on the real Kaliningrad graph: deterministic
> routing 100% intercepted → mixed 17-33%** (`scratch/interdiction_game_probe.py`). Decisions
> (Kilian 2026-07-06): Kaliningrad graph, single convoy first. **Read order now:**
> `REDESIGN_INTERDICTION.md` → `ROADMAP.md` (build plan) → `THESIS_STORYLINE.md` (4-act arc) →
> `experiments/gen08_interdiction.md` (forward pre-reg) → then the history below +
> `DIRECTION.md`/`experiments/gen07_contested_matrix.md` (why the exploitability path was
> necessary and where it hit the wall).
>
> **⚠️ (Superseded) REDIRECTION banner (2026-07-06, evening):** the exploitability reframe in a
> contested-resupply framing. Right instinct (minimax → worst-case robustness), but its
> destination-arena / learned-BR *realization* hit the flat-landscape wall; it has crystallised
> into the interdiction game above. `DIRECTION.md` records the reasoning bridge; `ROADMAP.md` is
> the active plan (interdiction phases). The campaign record below stands unchanged.

You are **Kilian Schwarz's SWE/planner agent on the SACRED MSc thesis** (Imperial College London,
supervisor Dr. Panagiotis Angeloudis). A prior Fable instance ran the project from the 2026-07-01
handover through the complete experimental campaign. **The campaign is finished.** Your job is to
support what comes next: thesis writing, the supervisor conversation, small follow-up
experiments only if Kilian asks: while preserving the standards that made the results
trustworthy. Everything below is verifiable in the repo; never cite numbers from anywhere but the
`experiments/` ledgers.

---

## 1. The finding (what the campaign established)

**Adversarial co-training, as naturally formulated for the stochastic-dynamic VRP, does not
confer robustness: and measurably worsens it.** The evidence chain, all pre-registered:

1. **gen03** (`experiments/gen03_robustness_dynassign.md`): ATLA co-evolution vs an identical
   vanilla-SAC control: no robustness delta (dD ≈ −250…−290 ± ~300–500, n.s.). Mechanism found:
   the **learned adversary attacks worse than uniform-random blocking** (D ≈ 0.6–1.9k vs random
   ≈ 1.7–2.1k), while a 40-line scripted heuristic hits 3–6× harder (≈ 4.9–5.9k).
2. **gen04** (`gen04_antag_gate.md`): after giving the adversary full motion observability (edge
   occupancy features), a retrained best-response attacker is *still* ≈ random (ratio 0.84 vs the
   pre-registered 1.25). Diagnosis: **entropy pinning** (max-entropy SAC over ~120 flat options
   with drowned advantages is mandated to play near-uniform) + reward SNR + γ-myopia.
3. **gen05** (`gen05_hybrid_matrix.md`): the matrix in the rich hybrid arena was
   **competence-void**: neither arm learned the task (clean W ≈ 5.6× greedy), and degradation
   near the saturation ceiling is compressed (**weak policies fake robustness**: an identified
   evaluation pitfall). One nugget: against a *competent* victim (greedy), the seeing learned
   attacker became the strongest attacker in the portfolio (+1667 > scripted's +1154/+714) -
   learned adversaries work where the reach mask aims for them and the victim is predictable.
4. **gen06** (`gen06_dynassign_matrix.md`): **the definitive result.** Competence gate PASSED
   (all six arms within +5.5–7.0% of greedy clean; gen03's band replicated). Primary
   **significantly reversed**: pooled `dD_targeted = −881 ± 284` (n=90, 0/3 pairings positive);
   the adversarially-trained arm is worse even under **its own training attacker**
   (dD_pathrand = −775 ± 244); dead even under random attack; zero clean cost. Robustness
   ranking: **greedy (4921) > vanilla (5196–5882) > adversarially-trained (6361–6575)**: the
   reactive classical dispatcher is the most robust policy measured (consistent with Ritzinger
   et al. 2015, cited in the literature review).

**Unifying mechanism:** the zero-sum latency reward (−queue/tick) buries each agent's
controllable contribution under a large uncontrollable shared baseline. The attacker's critic
can't resolve which blocks worked (→ pinned at ~uniform); the defender trained under attack gets
several-fold worse return SNR for the same sample budget (→ learns *less*, and the deficit
surfaces exactly where queue compounding amplifies policy quality: aimed attacks).

**Constructive contributions:** (a) four named preconditions for adversarial training to work in
this domain: a real coping channel in the action space, attacks with learnable structure, a
competence-first curriculum, variance-reduced (counterfactual-baselined) rewards; (b) the
evaluation methodology: pre-registration, competence gates, held-out attack portfolios,
per-policy best responses, paired instances, stochastic evaluation of max-entropy policies -
each earned by a specific documented failure (the static-3b retraction, gen05's ceiling trap).

## 2. Read in this order

1. **This file.**
2. **`SACRED_PROGRESS.md`**: the chronological run chronicle (10 entries, entry template at the
   top; Kilian wants every significant future run family appended there).
3. **`CRITIQUE.md`**: the 2026-07-02 critique that reframed the thesis; §5's skeptical-examiner
   questions still shape the writing.
4. **Ledgers** `experiments/gen02…gen06*.md`: pre-registered metrics + all citable numbers
   (+ portfolio JSONs beside them).
5. **`CONTEXT.md`** (banner explains what's historical), **`PROBLEM_REDESIGN.md`** (the pivot's
   design rationale), **`SYSTEM.md`** (operating dogmas: read fully before running anything),
   **`TASK.md`** (banner = plan state; body historical), **`docs/archive/`** (retired docs).
6. PDFs (extract with `.venv/bin/python` + pypdf):
   `../../MT_Literature_Survey_Kilian_Schwarz_split.pdf` (the assessed lit review; §2 =
   the five research objectives) and `../../MSc Transport - Research Project Guidance
   2025-2026.pdf` (deadlines/rubric: the thesis planner's first read).
7. Figures: `scratch/hybrid_geometry.png`, `scratch/chokepoints.png`, `assets/kaliningrad_*.png`,
   `scratch/dynassign_demo.gif`; probe scripts in `scratch/` are the reproducibility record.
8. Code (in this order): `src/env/smdp_wrapper.py` + `src/env/graph_env.py` (physics, SMDP
   events, antagonist reach/budget, hybrid state machine), `src/agents/networks.py`
   (featurization: 13 node / 4 edge dims incl. goal + motion columns; width-slicing back-compat),
   `src/agents/sac.py` (SAC math; `infer_node_in_dim`/`infer_edge_in_dim`; `_clip_x`/`_clip_ea`),
   `src/agents/sacred_atla.py` (trainer modes: atla · vanilla · antagonist_only ·
   scripted_adversary; `--update-every`), `src/baselines/greedy_dispatch.py` + `attackers.py`
   (random/targeted/pathrand/mask-first "gateway"), `scripts/train_sacred.py` (all flags),
   `scripts/evaluate_portfolio.py` (the robustness harness: paired W/D/dD, `--select-best
   --select-attacker`, held-out seed bases 10_000_019/20_000_019), `scripts/run_generation.py`
   (recipes + ledger discipline).

## 3. Evaluation vocabulary (fixed: use exactly this)

`W(arm, attack)` = mean total delivery latency (total_wait), lower better. `D(arm, a) = W(a) −
W(none)`, paired per instance. `dD = D(vanilla, a) − D(other, a)`; positive = the other arm is
more robust. Decision metrics are fixed in the ledger BEFORE looking; ≥3 seeds; paired instances;
CIs always; competence gate before interpreting any robustness comparison.

## 4. State of the machinery

- **Suite:** 83 tests green (`PYTHONPATH=. pytest tests/`: run after touching agents/env, paste
  raw output). All five problem rungs runnable: `--problem {osm,stage0,assign,dynassign,hybrid}`.
- **Selected checkpoints (gen06):** vanilla ep750/ep100/ep100, scripted ep450/ep200/ep600 under
  `models/runs/gen06_dynassign_matrix/*/snapshots/`; BR actors under `.../br_*_s0_seed0/`.
  gen05's analogues under `models/runs/gen05_hybrid_matrix/`.
- **Back-compat:** checkpoints of any feature-width era stay evaluable: agents slice features to
  their trained width; loaders infer widths from the checkpoint (`infer_*_in_dim`).
- **Hardware/ops:** M4 CPU-only (MPS 2.4–4× slower, settled), 4 torch threads solo / 3×3
  parallel; ~18–55 s/ep depending on rung; eval is cheap (~0.2–0.6 s/ep: don't over-estimate).
  Long jobs: `nohup … & disown` in their own session (harness-managed background tasks got reaped
  once and killed the children: documented in the gen05 ledger recovery note). Pause/resume:
  `pkill -STOP/-CONT -f train_sacred.py`. Kilian sometimes forbids scheduled wakeups: ask/obey.
- **Never** train without a ledger (pre-registered metric + pinned SHA), never compare across git
  states, never argmax-eval a max-entropy policy, gate multi-hour runs on cheap probes.

## 5. Open decisions & outlook (Kilian owns all of these)

1. **Freeze-and-write vs the option-(b) stretch.** Recommendation on record: freeze on gen06
   (defensible, complete); option (b): make the hybrid learnable (tighter corridor slack vs the
   slack-1.4× detour tension, γ↑, ERB warm-start from greedy demos = Obj-3 material): is now a
   test of the four preconditions and belongs in future work unless Kilian wants one more swing
   before ~Jul 16–18 (the campaign freeze target).
2. **Supervisor conversation (the old "D4").** Agenda: the gen06 finding + framing ("when and why
   adversarial VRP training fails, with conditions for success"), and descoping of ERB
   (inconclusive n=1), SBO (untouched), rolling-ALNS baseline, and ZST (options: a small transfer
   test *of the diagnosis* on a held-out geometry, or descope).
3. **Thesis writing.** A separate planner instance is briefed at
   `/Users/kilian/Kilian/ICL/Thesis/thesis/THESIS_PLANNER_HANDOFF.md` (launch = open Fable in
   `thesis/`, say "read THESIS_PLANNER_HANDOFF.md and begin"). That repo (Overleaf-synced) is
   where all report writing happens; this code repo is its read-only evidence base. Figure/
   experiment requests flow back here as a precise list.
4. **Back-pocket register** (recorded options, not scheduled): ATLA rider arm in a
   competent-protagonist matrix (motivated by gen05's +1667 finding); gen04b antagonist
   entropy-target re-gate (~2 h, tests the pinning hypothesis directly); option-(b) as above.
5. **Deferred chores:** `src/env/` vs `src/envs/` merge (mechanical, post-freeze, suite-guarded -
   TASK.md banner TODO 2); visualiser dims touch-up (`spar_visual*.py` builds 11-dim agents) if
   thesis figures need old-checkpoint rendering.

## 6. How to work with Kilian

Single `&&`-chained shell commands; his Mac never sleeps; he pauses runs for heat/noise; he
decides CPU spend and design changes: always present options with a recommendation and wait;
report honestly including self-corrections (this project retracted a headline claim once and is
stronger for it); ask questions when direction is genuinely his to choose, otherwise act.
Persistent memory for this repo path exists (`~/.claude/projects/-Users-kilian-Kilian-ICL-Thesis-
code-sacred/memory/`): read `MEMORY.md` there at session start; keep it and
`SACRED_PROGRESS.md` current as work proceeds.
