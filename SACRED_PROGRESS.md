# SACRED_PROGRESS.md — the run chronicle

> **Purpose.** One entry per *significant* run or run family (no smoke tests, no micro-benchmarks),
> in chronological order, so the development of the SACRED project can be followed coherently over
> time — what each experiment set out to show, what it actually showed, and how it moved the
> thesis. Detailed protocols/numbers live in `experiments/<gen>.md` ledgers and `CONTEXT.md`; this
> document is the narrative spine.
>
> **Entry template** (append new entries at the bottom, keep them this shape):
> ```
> ## <N>. <run / run family name>  (dates · code state · ledger)
> - **Goal (prospective):** what the run was launched to demonstrate, written as it was framed then.
> - **Headline results:** the few figures that matter, with uncertainty.
> - **What we learned:** the honest reading, including surprises and retractions.
> - **Thesis progression:** what capability/knowledge the project gained.
> - **What it means for the thesis:** implications for the headline claim(s).
> - **Thesis fit:** which research objectives (Obj 1–5, ZST) it serves and how it will appear
>   in the written thesis.
> ```
> Research objectives shorthand (from the literature review §2.2): **Obj 1** zero-sum game
> formulation · **Obj 2** simulation environment · **Obj 3** SAC+ATLA+ERB bootstrapping ·
> **Obj 4** SBO facility/fleet · **Obj 5** evaluation vs metaheuristics *and vs a
> non-adversarially-trained SAC baseline* under disruption · **ZST** zero-shot transfer.

---

## 1. OSM static baseline era (`...0614_170342` diverged run · `protag_signal_rebalance` · `protag_reward_shaping`)  (2026-06-14 → 06-18 · pre-`90e759c` · no ledger, see CONTEXT §3)

- **Goal (prospective):** train the original SACRED formulation end-to-end on the Kaliningrad OSM
  graph (290 nodes, 4 trucks, 150 packages, static demand spread over 95 nodes) and show the
  protagonist learns to dispatch under a co-evolving congestion adversary.
- **Headline results:** first 2000-ep run **diverged** (SAC temperature α 1→69, critic Q 7→601 —
  root cause: inverted alpha-loss sign, later fixed). After the fix, two reward-variant runs were
  *stable but flat*: delivery rate pinned at **~0.91** across every variant; the critic's
  `Q_Spread` diagnostic **collapsed 5.3→0.46**; policy entropy never fell.
- **What we learned:** the machinery had a real bug (alpha sign) *and* the problem had a real flaw:
  with demand everywhere, per-step decisions are near-inconsequential — ~0.91 is the ceiling any
  coverage policy reaches, and the adversary is toothless (trucks just serve a different nearby
  node). "Stable training" and "learnable problem" are different properties.
- **Thesis progression:** produced the trusted SAC core (correct alpha loss, grad clipping, batched
  GNN updates ~1.45×, exact-Dijkstra routing) and the diagnostic toolkit (`Q_Spread`, windowed
  tfevents reads) still in use.
- **What it means for the thesis:** motivated the pivot to a *dynamic, multi-depot,
  latency-objective* VRP (PROBLEM_REDESIGN.md) — decisions must be consequential and the adversary
  must matter for any adversarial-RL claim to be testable.
- **Thesis fit:** Obj 2 (the environment) and Obj 3 (working SAC+ATLA machinery); appears in the
  thesis as the motivation section for the problem redesign ("why the naive formulation is
  unlearnable"), plus a methods lesson on divergence diagnosis.

## 2. Stage 0 — single-truck next-hop route choice  (2026-06-27/28 · pre-`90e759c` · CONTEXT §2)

- **Goal (prospective):** validation rung: prove the SACRED stack can learn a *consequential
  adversarial* policy at all, on the cleanest possible signal (1 truck, two-route corridor, the
  policy picks each edge so congestion is an exploitable decision).
- **Headline results (1000 ep):** it learns — `Q_Spread` 0.85→1.58, entropy 0.60→0.34 (commits),
  antagonist co-evolves (Q 22→62), 12/12 delivered — but **matches reactive greedy under attack**
  (final gap +24, ~1.6%) and never beats it. Three enabling fixes: reward_scale 0.01→0.1 (task
  signal vs entropy bonus), forward-corridor action mask, corridor slack 1.2.
- **What we learned:** the stack trains; single-truck route choice vs a reactive
  congestion-aware greedy is structurally near-a-wash (greedy re-plans optimally each edge).
- **Thesis progression:** first working curriculum rung; next-hop routing machinery, greedy
  baselines, periodic eval infrastructure.
- **What it means for the thesis:** first of the three "near-washes" that would later force the
  question of whether "beat greedy" was ever the right success criterion.
- **Thesis fit:** Obj 2/3 validation; appears as the curriculum's first rung and as evidence in
  the "reactive baselines are strong" argument (consistent with Ritzinger et al. 2015).

## 3. Static-3b assignment probe (`assign_probe`, `assign_probe_claimfix`)  (2026-06-28, **retracted 06-29** · pre-`90e759c` · CONTEXT §2)

- **Goal (prospective):** move the lever to multi-truck *assignment* (2 depots, 8 contested static
  requests, destination mode) where greedy insertion is provably suboptimal, and show RL beats it.
- **Headline results:** first run lost everywhere (a double-assignment bug); after the claim-fix,
  the run was briefly celebrated as "first RL beats classical" (final gap −56, best −188) — then
  **RETRACTED on a windowed re-read**: mean `gap_atk` over all 20 eval points **+18** (a loss),
  reliable static loss ~+8%, and the metric itself was mis-specified (single deterministic episode
  vs the *co-evolving* antagonist = arms-race timing, not robustness).
- **What we learned:** the project's most important *methodological* lesson: never headline a
  final/best point; fix the decision metric in advance; evaluate against fixed/held-out
  adversaries over multiple instances; save per-phase checkpoints. Also a real mechanical fix
  (sequential claiming) that survives in every later rung.
- **Thesis progression:** triggered the experiment-management infrastructure (seeded generations,
  ledgers pinning SHA + pre-registered metrics, aggregate-over-seeds reporting).
- **What it means for the thesis:** the retraction is *presentable* — it demonstrates
  self-correcting methodology, and it seeded the evaluation standards every later claim rests on.
- **Thesis fit:** methods chapter (evaluation methodology under co-evolution); a cautionary
  subsection the examiners will read as rigor, not failure.

## 4. `gen01_erb_ablation` — ERB bootstrapping pilot  (2026-06-28/29, paused/inconclusive · `experiments/gen01_erb_ablation.md`)

- **Goal (prospective):** does seeding the replay buffer with greedy demonstrations (Obj 3's ERB
  bootstrapping) fix the static assignment gap and accelerate learning? {erb, noerb} × 3 seeds.
- **Headline results (n=1 completed — seed0 only, paused for heat):** ERB did **not** move the
  static gap (`gap_noatk` ~+50 throughout); late co-evolution instability wrecked the run's tail
  (ep-1000 gap +282, `Q_Spread` 7.1→1.9).
- **What we learned:** (a) ERB-as-built wasn't earning its keep; (b) the bigger, config-agnostic
  problem was late antagonist runaway + the final-checkpoint metric — findings that shaped the
  best-checkpoint / fixed-adversary protocol.
- **Thesis progression:** first use of the generation infrastructure; identified the
  measurement fixes that gen02 baked in.
- **What it means for the thesis:** Obj 3's ERB claim is currently unsupported; it stays descoped
  until a rung exists where learning speed is the bottleneck (revisit-or-descope decision with
  supervisor).
- **Thesis fit:** Obj 3 (ERB) — likely a short "attempted, inconclusive at n=1, deprioritised"
  subsection unless revisited on the final rung.

## 5. `gen02_dynassign` — Stage 1.5 dynamic (Poisson) assignment  (2026-06-29/30 · SHA `dd96228`+WT · `experiments/gen02_dynassign.md`)

- **Goal (prospective):** the assignment lever in the dynamic regime (Poisson λ=0.06 ≈ ρ=1,
  2 depots, destination mode, latency reward, full-blockage antagonist), measured with the fixed
  static-3b lessons: does RL beat greedy-insertion under attack, beyond seed luck? 2 seeds × 800 ep.
- **Headline results:** best-checkpoint `gap_atk` ≈ **−106 ± ~1000** (not significant,
  selection-biased — seed0's "best" was the untrained ep50); reliable static loss `gap_noatk` ≈
  **+348 (~+6%)**; antagonist Q 37→116 ("runaway"). Two latent antagonist bugs found+fixed
  (hardcoded congestion-level values) and the antagonist-phase compute blow-up solved
  (budget-capped full-blockage redesign, 295→18 s/ep).
- **What we learned (as read at the time):** third near-wash; interpreted as "destination-mode
  auto-routing starves the antagonist → the missing lever is next-hop routing." *(Later revision:
  gen03 showed the antagonist itself couldn't attack, and the ρ≈1 operating point made the metric
  variance ±1000 — the rung was likely unresolvable as posed.)*
- **Thesis progression:** the full dynamic environment (Poisson arrivals, queue/ETA observations,
  per-phase snapshots, fixed-adversary multi-instance eval) — the SDVRP machinery the thesis
  promised; its snapshots became the SACRED arm of the gen03 pilot for free.
- **What it means for the thesis:** the "D" in SDVRP is built and validated; the beat-greedy
  framing hit its third wash, setting up the external critique.
- **Thesis fit:** Obj 2 (dynamic env), Obj 5 (evaluation protocol evolution); appears as the final
  motivating null before the reframe.

## 6. Approach critique + hybrid fixes (no training)  (2026-07-01/02 · `90e759c`→`d2b065b` · `CRITIQUE.md`)

- **Goal (prospective):** external fresh-eyes interrogation of the whole approach after three
  near-washes (Kilian's handoff instruction), before spending CPU on the built-but-untrained
  hybrid rung (H7).
- **Headline results (probes, not training):** (a) the hybrid rung had a **mechanical bug firing
  every episode** — cross-event double assignment stranded a truck orbiting a served node,
  episodes always ran to the 1500-tick horizon (fixed: episodes end ~tick 220, greedy's own
  baseline improved 902→847); (b) the H5 "+79% recoverable headroom" decomposed: post-fix, a
  permanent single-gateway blockade costs only **+10.4%** (floor), while the scripted route-reach
  attack costs **+40…+184%** over budget 250–4000 — the damage is chase dynamics, i.e. genuinely
  contestable; (c) observability gaps documented (policy couldn't see its own goal; 2-hop GNN vs
  global-Dijkstra greedy; antagonist blind to commitments/motion); (d) γ-myopia and eval-protocol
  biases (off-target fixed adversary, selection-on-test, argmax-ing a max-entropy policy).
- **What we learned:** the three near-washes were **not valid evidence** for their structural
  conclusions (biased + underpowered protocol, handicapped learner), and the thesis's own Obj-5
  control — a *non-adversarially trained* SAC — had never been run.
- **Thesis progression:** the **reframe** (accepted as D1): headline = *robustness from
  adversarial training* (SACRED vs vanilla SAC under held-out attacks), greedy demoted to
  reference line; hybrid rung repaired (13-dim observability, info-parity ETAs); portfolio
  evaluation protocol designed (paired instances, per-policy best responses, validation/test
  attacker split).
- **What it means for the thesis:** the claim became achievable-as-posed and aligned with the
  literature review's actual research objectives; every subsequent experiment is pre-registered.
- **Thesis fit:** methods + framing chapters; the critique's structure (framing drift, protocol
  biases, learnability handicaps) maps directly onto the thesis's "limitations of naive
  adversarial evaluation" section.

## 7. `gen03_robustness_dynassign` — Phase-1 robustness pilot  (2026-07-02/03 · `de5ff7d`→`c7ff687` · `experiments/gen03_robustness_dynassign.md`)

- **Goal (prospective):** first test of the reframed headline on the dynassign rung, reusing the
  sunk gen02 SACRED runs: *does ATLA co-evolution buy robustness to held-out attacks vs an
  identical vanilla-SAC control?* Pre-registered primary: dD = D(vanilla, its own best-response) −
  D(sacred, its own best-response) > 0, paired 95% CI excluding 0, both sacred seeds.
- **Headline results:** **primary NULL, wrong sign**: dD = −291 ± 500 (pair 0), −255 ± 295
  (pair 1). No clean-performance cost of ATLA either (both arms ≈ +7% behind greedy statically).
  **The diagnostic that matters** — attacker hierarchy on identical paired instances:
  scripted `targeted` heuristic D ≈ **+4.9–5.8k (~+79% on greedy)** ≫ `random` ≈ +1.7–2.1k ≫
  **learned best-response ≈ +0.6–1.9k — weaker than random**, despite 300 dedicated training
  episodes each; during BR training the antagonist's true reward *fell* (~9.0k→8.4k) while its Q
  tripled (35→115) — the "Q runaway" was critic over-estimation all along.
- **What we learned:** **the binding constraint on the whole framework is adversary competence,
  not protagonist learning.** ATLA trained SACRED against a near-random sparring partner, so there
  was no robustness to learn; the co-evolved adversary's apparent strength in earlier rungs was an
  artifact. The large scripted-attack surface proves robustness *is* measurable here.
- **Thesis progression:** the complete robustness-evaluation pipeline ran end-to-end (vanilla
  control, per-arm checkpoint selection on a validation attacker, per-policy best-response
  attackers, 1,140-episode paired portfolio with CIs ±300–500); root cause isolated to missing
  motion observability → the N1 fix (directed edge-occupancy features) + the gen04 gate.
- **What it means for the thesis:** a pre-registered, correctly-powered, mechanistically-explained
  null — publishable-shape in its own right. The narrative sharpens to: *the central practical
  obstacle to RARL in vehicle routing is training a competent environment-adversary; we
  demonstrate it, diagnose it, fix it, and measure what adversarial training then buys.* Decision
  recorded 2026-07-04: the scripted-adversarial training arm stays **in the back pocket** (D3
  unchanged) — it becomes the Phase-3 fallback if pure SACRED fails the gen04 gate.
- **Thesis fit:** Obj 5's named control (finally run) + Obj 1/3 stress-test; this is the pivotal
  results chapter that motivates the final experiment. The methodology (portfolio, per-policy
  best responses, pre-registration) is itself a contribution.

## 8. `gen04_antag_gate` — can the adversary see now?  (2026-07-04 · ledger `experiments/gen04_antag_gate.md` · IN PROGRESS)

- **Goal (prospective):** after the N1 fix (edge features 2→4: directed truck occupancy +
  progress — the exact state the scripted attacker exploits), retrain one best-response attacker
  against the *same* frozen defender as gen03 and gate Phase-3's co-evolution: **PASS** =
  D(br_fixed) ≥ 1.25 × D(random) on 16 validation instances; **STRONG PASS** = additionally
  ≥ 0.5 × D(targeted); **FAIL** = co-evolution parked, scripted-adversarial arm promoted for
  Phase 3.
- **Headline results:** **GATE FAIL.** With full motion observability, the retrained
  best-response attacker degrades the defender by **1663 ± 517** — still below random blocking
  (1984 ± 447; ratio 0.84 vs required 1.25) and 0.28× the scripted heuristic (5868 ± 647).
  Training signature identical to gen03: true reward fell (8710→8120), Q inflated 3× (37→113),
  critic loss never converged, **α pinned at 1.0 / entropy ~2.1**.
- **What we learned:** observability was necessary but not sufficient. The failure is in the
  antagonist's *learning problem*: (a) the max-entropy objective with a 0.5·ln(N) target over a
  ~120-option flat action space **requires a near-uniform policy** at these advantage magnitudes
  — near-uniform over the mask ≈ the random attacker, which is exactly what both gates measured;
  (b) reward SNR (~1–2k controllable vs ~8k queue baseline) and γ=0.99/tick myopia starve the
  critic. "Learned congestion adversaries underperform an informed 40-line heuristic even with
  full observability" is now a two-datapoint finding.
- **Thesis progression:** the gate did its job for ~2 h of CPU — Phase 3 will not spend its
  budget on a broken co-evolution loop. Per the pre-registered consequence, the
  **scripted-adversarial arm is promoted** for Phase 3; an ATLA arm may still ride along because
  the hybrid arena's route-reach mask aims attacks structurally (decision pending Kilian).
- **What it means for the thesis:** the honest narrative sharpens further: adversarial *pressure*
  is easy to supply (scripted) and hard to *learn* (max-entropy SAC over a flat edge space);
  SACRED's robustness claim is tested with the strong scripted adversary, while the co-evolution
  component becomes a diagnosed limitation with a concrete mechanism (entropy pinning + SNR),
  not a vague failure.
- **Thesis fit:** Obj 1/3 (the limits of the zero-sum co-evolution as instantiated) and the
  methods chapter (gating expensive training on cheap pre-registered probes).
- **Decisions (2026-07-04, Kilian):** scripted-adversarial arm **promoted** into Phase 3
  (`gen05_hybrid_matrix`); the ATLA co-evolution rider arm and the lowered-entropy-target re-gate
  (gen04b) both go to the **back pocket** — recorded options, not scheduled work.

## 9. `gen05_hybrid_matrix` — Phase 3, the headline robustness matrix  (2026-07-04 · `cd11f14`/`324a644` · `experiments/gen05_hybrid_matrix.md`)

- **Goal (prospective):** the reframed thesis headline on the repaired hybrid arena: does
  adversarial training against the strong scripted `targeted` attacker buy robustness to the
  held-out `gateway` attack, vs an identical non-adversarial control? 2 arms × 3 seeds × 400 ep
  (budget 1500, horizon 800, `--update-every 8`); pre-registered primary = pooled dD_gateway > 0,
  CI excl. 0, ≥2/3 pairings.
- **Headline results:** **primary NOT MET — sign reversed** (pooled dD_gateway = −192 ± 181,
  0/3 pairings positive). The dominating observation: **neither arm learned the task** —
  W(none) ≈ 4.6–4.8k vs greedy's 847 (~5.6× worse), and in-distribution dD_targeted = −20 ± 135:
  training against the attacker taught no measurable coping *even against that same attacker*.
  With W(none) near the 6.4k saturation ceiling, degradation is ceiling-compressed (both learned
  arms show smaller D than greedy — weakness masquerading as robustness).
- **What we learned:** the binding constraint moved. gen03/gen04 diagnosed the *adversary* as
  unable to learn; gen05 shows that on the hybrid rung the *protagonist* can't learn either at
  this budget/structure (hundreds of edge-level micro-decisions, thin credit, γ-myopia; Q_Spread
  ≈ 0.1 = the critic never discriminated). A robustness comparison between two incompetent
  policies is uninterpretable — competence is a precondition for the robustness question.
- **Thesis progression:** the full matrix pipeline (dual-arm training, held-out attack design,
  paired stochastic portfolio) ran end-to-end and produced a clean pre-registered readout in one
  day — the machinery is no longer ever the bottleneck. The interruption/resume during training
  also proved the checkpointing discipline (lossless mid-run recovery).
- **What it means for the thesis:** the honest empirical arc is now: adversarial VRP training
  fails two ways — the adversary can't learn to attack (gen03/04), and on decision-dense arenas
  the protagonist can't learn to act (gen05). Options recorded in the ledger: extend training /
  make the hybrid learnable (fewer decisions, denser credit) / move the matrix to dynassign where
  policies demonstrably reach within ~7% of greedy / freeze and write the diagnostic arc.
  Kilian to decide with the freeze (~Jul 16–18) in view.
- **Thesis fit:** Obj 5 (the robustness evaluation methodology is itself the contribution:
  pre-registration, held-out attacks, paired instances, ceiling-compression as an identified
  robustness-evaluation pitfall) + the results chapter's second act; strengthens the methods
  narrative regardless of what the final matrix shows.

## 10. `gen06_dynassign_matrix` — the competence-valid robustness matrix  (2026-07-05 · `cfabc90`/`0bc6ec3` · `experiments/gen06_dynassign_matrix.md`)

- **Goal (prospective):** the Phase-3 retake in the arena where policies demonstrably learn:
  {vanilla, scripted-adversarial (trained vs the new stochastic `pathrand` attacker)} × 3 fresh
  seeds × 800 ep on dynassign; primary = pooled dD under the fully held-out `targeted` attack,
  behind a pre-registered **competence gate** (the gen05 lesson).
- **Headline results:** **gate PASSED everywhere** (all six arms within +5.5…+7.0% of greedy
  clean — gen03's band, replicated); **primary NOT MET and significantly reversed**: pooled
  dD_targeted = **−881 ± 284** (0/3 pairings), dD_pathrand = **−775 ± 244** (worse even under its
  own training attacker), dD_random = −45 ± 221 (even), clean premium ≈ 0. Robustness ranking:
  **greedy (D 4921) > vanilla (5196–5882) > adversarially-trained (6361–6575)**.
- **What we learned:** with competence finally established, adversarial training against a
  strong scripted attacker **worsens** robustness to route-aimed attacks — in- and
  out-of-distribution — at zero clean cost. Leading mechanism, consistent with the campaign-wide
  SNR theme: constant attack floods the latency reward with unavoidable damage, diluting the
  learnable signal; the deficit surfaces precisely where queue compounding amplifies policy
  quality. And the most robust policy in the matrix is the reactive classical dispatcher —
  Ritzinger's reactive-dominance, measured in our own framework.
- **Thesis progression:** the campaign now closes a complete, coherent chain of evidence:
  the learned adversary cannot learn to attack (gen03/04) → the protagonist cannot learn
  decision-dense arenas (gen05) → and even the best-case configuration (strong fixed adversary,
  competent protagonist, clean evaluation) yields *negative* robustness transfer (gen06).
- **What it means for the thesis:** the definitive experimental finding, and a genuinely
  publishable-shaped one: *adversarial co-training as formulated does not confer robustness in
  stochastic-dynamic VRP, and the paper explains why* — reward SNR under zero-sum latency,
  entropy pinning, decision density, and reactive-baseline dominance. The methodology
  (pre-registration, competence gates, held-out attack portfolios, paired instances) is the
  constructive contribution. Freeze can proceed on this result.
- **Thesis fit:** Objectives 1/3/5 all get evidenced answers (negative but rigorous); the
  discussion chapter's "when does adaptive RL help" question gets its honest empirical answer
  for this problem class.

## 11. Contested-resupply redirection: analysis + direction, no training  (2026-07-06 · docs `DIRECTION.md`/`ROADMAP.md`/`THESIS_STORYLINE.md`)

- **Goal (prospective):** answer Kilian's carte-blanche brief after gen06 closed the campaign:
  can adversarial training be made to show a real benefit if anything may change except SAC, the
  protagonist/antagonist dynamic, and RL? Context: the supervisor wants alternative applications
  (deck `../../Weekly Presentations/06.07..pptx`: contested drone resupply, humanitarian
  logistics, asset escort, LEO/Kessler) and pointed at Panopticon AI and AAMAS.
- **Headline results (analysis only, no runs):** (a) a comparative session read of the gen06
  tfevents found systematic arm differences beyond the ledgered SNR story: the
  adversarially-trained arms trained in a collapse regime (delivery 0.18-0.27 vs 0.66; queue
  ~2x) with temperature never annealing (final alpha 0.62-0.86 vs 0.13; entropy 0.47-0.52 vs
  0.37-0.39), while protagonist Q_Spread was HIGHER under attack (13-15 vs 2.6-3.8), so the
  gen03/04 "critic cannot discriminate" wording does not transfer to the gen06 protagonist;
  three mechanism candidates recorded (M1 reward SNR, M2 entropy-target mis-scaling with
  backlog, M3 collapse-regime state distribution); reproduced same night by the committed A3
  probes (`scratch/gen06_telemetry_probe.py` and siblings), results in the gen06 ledger's
  post-hoc appendix. (b) Direction converged with Kilian: move the
  headline to **exploitability** (worst case against a strategic, adaptive attacker; portfolio-
  max measurement) in a **contested-resupply** skin of the existing stack; five fixes map
  one-to-one to the campaign's diagnosed pathologies (exposure curriculum, counterfactual twin
  rewards, entropy repair, factored attacker + adversary population, credit horizon).
- **What we learned:** minimax training's native claim is worst-case, not average-case; the
  campaign's negative is register-valid, and the in-house existence proofs for the reframe are
  already ledgered (gen05's +1667 learned-attacker-vs-greedy nugget; gen06 BR rows).
  Deterministic dispatch's predictability becomes the measurable weakness; SAC's entropy becomes
  the mechanism, not the nuisance.
- **Thesis progression:** `DIRECTION.md` (new view + direction), `ROADMAP.md` (phased plan with
  gates), `THESIS_STORYLINE.md` (objective-by-objective argument) opened; gen07 pre-registration
  drafting is ROADMAP A4. Supervisor sign-off pending; nothing built; no CPU spent.
- **What it means for the thesis:** the narrative becomes three acts: diagnosis (gen03-06) →
  the right question (exploitability) → conditions under which adversarial training works
  (gen07, pre-registered either way).
- **Thesis fit:** Obj 1/3/5 gain a positive path (game evaluated against its own solution
  concept; SAC-as-mixed-strategy; controls that make Obj 5 causal); Obj 4 gains a reduced-form
  surrogate demonstrator option; ZST scoped as one held-out-geometry transfer test.
