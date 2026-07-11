# SACRED_PROGRESS.md: the run chronicle

> **Purpose.** One entry per *significant* run or run family (no smoke tests, no micro-benchmarks),
> in chronological order, so the development of the SACRED project can be followed coherently over
> time: what each experiment set out to show, what it actually showed, and how it moved the
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
- **Headline results:** first 2000-ep run **diverged** (SAC temperature α 1→69, critic Q 7→601 -
  root cause: inverted alpha-loss sign, later fixed). After the fix, two reward-variant runs were
  *stable but flat*: delivery rate pinned at **~0.91** across every variant; the critic's
  `Q_Spread` diagnostic **collapsed 5.3→0.46**; policy entropy never fell.
- **What we learned:** the machinery had a real bug (alpha sign) *and* the problem had a real flaw:
  with demand everywhere, per-step decisions are near-inconsequential: ~0.91 is the ceiling any
  coverage policy reaches, and the adversary is toothless (trucks just serve a different nearby
  node). "Stable training" and "learnable problem" are different properties.
- **Thesis progression:** produced the trusted SAC core (correct alpha loss, grad clipping, batched
  GNN updates ~1.45×, exact-Dijkstra routing) and the diagnostic toolkit (`Q_Spread`, windowed
  tfevents reads) still in use.
- **What it means for the thesis:** motivated the pivot to a *dynamic, multi-depot,
  latency-objective* VRP (PROBLEM_REDESIGN.md): decisions must be consequential and the adversary
  must matter for any adversarial-RL claim to be testable.
- **Thesis fit:** Obj 2 (the environment) and Obj 3 (working SAC+ATLA machinery); appears in the
  thesis as the motivation section for the problem redesign ("why the naive formulation is
  unlearnable"), plus a methods lesson on divergence diagnosis.

## 2. Stage 0: single-truck next-hop route choice  (2026-06-27/28 · pre-`90e759c` · CONTEXT §2)

- **Goal (prospective):** validation rung: prove the SACRED stack can learn a *consequential
  adversarial* policy at all, on the cleanest possible signal (1 truck, two-route corridor, the
  policy picks each edge so congestion is an exploitable decision).
- **Headline results (1000 ep):** it learns: `Q_Spread` 0.85→1.58, entropy 0.60→0.34 (commits),
  antagonist co-evolves (Q 22→62), 12/12 delivered: but **matches reactive greedy under attack**
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
  the run was briefly celebrated as "first RL beats classical" (final gap −56, best −188): then
  **RETRACTED on a windowed re-read**: mean `gap_atk` over all 20 eval points **+18** (a loss),
  reliable static loss ~+8%, and the metric itself was mis-specified (single deterministic episode
  vs the *co-evolving* antagonist = arms-race timing, not robustness).
- **What we learned:** the project's most important *methodological* lesson: never headline a
  final/best point; fix the decision metric in advance; evaluate against fixed/held-out
  adversaries over multiple instances; save per-phase checkpoints. Also a real mechanical fix
  (sequential claiming) that survives in every later rung.
- **Thesis progression:** triggered the experiment-management infrastructure (seeded generations,
  ledgers pinning SHA + pre-registered metrics, aggregate-over-seeds reporting).
- **What it means for the thesis:** the retraction is *presentable*: it demonstrates
  self-correcting methodology, and it seeded the evaluation standards every later claim rests on.
- **Thesis fit:** methods chapter (evaluation methodology under co-evolution); a cautionary
  subsection the examiners will read as rigor, not failure.

## 4. `gen01_erb_ablation`: ERB bootstrapping pilot  (2026-06-28/29, paused/inconclusive · `experiments/gen01_erb_ablation.md`)

- **Goal (prospective):** does seeding the replay buffer with greedy demonstrations (Obj 3's ERB
  bootstrapping) fix the static assignment gap and accelerate learning? {erb, noerb} × 3 seeds.
- **Headline results (n=1 completed: seed0 only, paused for heat):** ERB did **not** move the
  static gap (`gap_noatk` ~+50 throughout); late co-evolution instability wrecked the run's tail
  (ep-1000 gap +282, `Q_Spread` 7.1→1.9).
- **What we learned:** (a) ERB-as-built wasn't earning its keep; (b) the bigger, config-agnostic
  problem was late antagonist runaway + the final-checkpoint metric: findings that shaped the
  best-checkpoint / fixed-adversary protocol.
- **Thesis progression:** first use of the generation infrastructure; identified the
  measurement fixes that gen02 baked in.
- **What it means for the thesis:** Obj 3's ERB claim is currently unsupported; it stays descoped
  until a rung exists where learning speed is the bottleneck (revisit-or-descope decision with
  supervisor).
- **Thesis fit:** Obj 3 (ERB): likely a short "attempted, inconclusive at n=1, deprioritised"
  subsection unless revisited on the final rung.

## 5. `gen02_dynassign`: Stage 1.5 dynamic (Poisson) assignment  (2026-06-29/30 · SHA `dd96228`+WT · `experiments/gen02_dynassign.md`)

- **Goal (prospective):** the assignment lever in the dynamic regime (Poisson λ=0.06 ≈ ρ=1,
  2 depots, destination mode, latency reward, full-blockage antagonist), measured with the fixed
  static-3b lessons: does RL beat greedy-insertion under attack, beyond seed luck? 2 seeds × 800 ep.
- **Headline results:** best-checkpoint `gap_atk` ≈ **−106 ± ~1000** (not significant,
  selection-biased: seed0's "best" was the untrained ep50); reliable static loss `gap_noatk` ≈
  **+348 (~+6%)**; antagonist Q 37→116 ("runaway"). Two latent antagonist bugs found+fixed
  (hardcoded congestion-level values) and the antagonist-phase compute blow-up solved
  (budget-capped full-blockage redesign, 295→18 s/ep).
- **What we learned (as read at the time):** third near-wash; interpreted as "destination-mode
  auto-routing starves the antagonist → the missing lever is next-hop routing." *(Later revision:
  gen03 showed the antagonist itself couldn't attack, and the ρ≈1 operating point made the metric
  variance ±1000: the rung was likely unresolvable as posed.)*
- **Thesis progression:** the full dynamic environment (Poisson arrivals, queue/ETA observations,
  per-phase snapshots, fixed-adversary multi-instance eval): the SDVRP machinery the thesis
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
  every episode**: cross-event double assignment stranded a truck orbiting a served node,
  episodes always ran to the 1500-tick horizon (fixed: episodes end ~tick 220, greedy's own
  baseline improved 902→847); (b) the H5 "+79% recoverable headroom" decomposed: post-fix, a
  permanent single-gateway blockade costs only **+10.4%** (floor), while the scripted route-reach
  attack costs **+40…+184%** over budget 250–4000: the damage is chase dynamics, i.e. genuinely
  contestable; (c) observability gaps documented (policy couldn't see its own goal; 2-hop GNN vs
  global-Dijkstra greedy; antagonist blind to commitments/motion); (d) γ-myopia and eval-protocol
  biases (off-target fixed adversary, selection-on-test, argmax-ing a max-entropy policy).
- **What we learned:** the three near-washes were **not valid evidence** for their structural
  conclusions (biased + underpowered protocol, handicapped learner), and the thesis's own Obj-5
  control: a *non-adversarially trained* SAC: had never been run.
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

## 7. `gen03_robustness_dynassign`: Phase-1 robustness pilot  (2026-07-02/03 · `de5ff7d`→`c7ff687` · `experiments/gen03_robustness_dynassign.md`)

- **Goal (prospective):** first test of the reframed headline on the dynassign rung, reusing the
  sunk gen02 SACRED runs: *does ATLA co-evolution buy robustness to held-out attacks vs an
  identical vanilla-SAC control?* Pre-registered primary: dD = D(vanilla, its own best-response) −
  D(sacred, its own best-response) > 0, paired 95% CI excluding 0, both sacred seeds.
- **Headline results:** **primary NULL, wrong sign**: dD = −291 ± 500 (pair 0), −255 ± 295
  (pair 1). No clean-performance cost of ATLA either (both arms ≈ +7% behind greedy statically).
  **The diagnostic that matters**: attacker hierarchy on identical paired instances:
  scripted `targeted` heuristic D ≈ **+4.9–5.8k (~+79% on greedy)** ≫ `random` ≈ +1.7–2.1k ≫
  **learned best-response ≈ +0.6–1.9k: weaker than random**, despite 300 dedicated training
  episodes each; during BR training the antagonist's true reward *fell* (~9.0k→8.4k) while its Q
  tripled (35→115): the "Q runaway" was critic over-estimation all along.
- **What we learned:** **the binding constraint on the whole framework is adversary competence,
  not protagonist learning.** ATLA trained SACRED against a near-random sparring partner, so there
  was no robustness to learn; the co-evolved adversary's apparent strength in earlier rungs was an
  artifact. The large scripted-attack surface proves robustness *is* measurable here.
- **Thesis progression:** the complete robustness-evaluation pipeline ran end-to-end (vanilla
  control, per-arm checkpoint selection on a validation attacker, per-policy best-response
  attackers, 1,140-episode paired portfolio with CIs ±300–500); root cause isolated to missing
  motion observability → the N1 fix (directed edge-occupancy features) + the gen04 gate.
- **What it means for the thesis:** a pre-registered, correctly-powered, mechanistically-explained
  null: publishable-shape in its own right. The narrative sharpens to: *the central practical
  obstacle to RARL in vehicle routing is training a competent environment-adversary; we
  demonstrate it, diagnose it, fix it, and measure what adversarial training then buys.* Decision
  recorded 2026-07-04: the scripted-adversarial training arm stays **in the back pocket** (D3
  unchanged): it becomes the Phase-3 fallback if pure SACRED fails the gen04 gate.
- **Thesis fit:** Obj 5's named control (finally run) + Obj 1/3 stress-test; this is the pivotal
  results chapter that motivates the final experiment. The methodology (portfolio, per-policy
  best responses, pre-registration) is itself a contribution.

## 8. `gen04_antag_gate`: can the adversary see now?  (2026-07-04 · ledger `experiments/gen04_antag_gate.md` · IN PROGRESS)

- **Goal (prospective):** after the N1 fix (edge features 2→4: directed truck occupancy +
  progress: the exact state the scripted attacker exploits), retrain one best-response attacker
  against the *same* frozen defender as gen03 and gate Phase-3's co-evolution: **PASS** =
  D(br_fixed) ≥ 1.25 × D(random) on 16 validation instances; **STRONG PASS** = additionally
  ≥ 0.5 × D(targeted); **FAIL** = co-evolution parked, scripted-adversarial arm promoted for
  Phase 3.
- **Headline results:** **GATE FAIL.** With full motion observability, the retrained
  best-response attacker degrades the defender by **1663 ± 517**: still below random blocking
  (1984 ± 447; ratio 0.84 vs required 1.25) and 0.28× the scripted heuristic (5868 ± 647).
  Training signature identical to gen03: true reward fell (8710→8120), Q inflated 3× (37→113),
  critic loss never converged, **α pinned at 1.0 / entropy ~2.1**.
- **What we learned:** observability was necessary but not sufficient. The failure is in the
  antagonist's *learning problem*: (a) the max-entropy objective with a 0.5·ln(N) target over a
  ~120-option flat action space **requires a near-uniform policy** at these advantage magnitudes
 : near-uniform over the mask ≈ the random attacker, which is exactly what both gates measured;
  (b) reward SNR (~1–2k controllable vs ~8k queue baseline) and γ=0.99/tick myopia starve the
  critic. "Learned congestion adversaries underperform an informed 40-line heuristic even with
  full observability" is now a two-datapoint finding.
- **Thesis progression:** the gate did its job for ~2 h of CPU: Phase 3 will not spend its
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
  (gen04b) both go to the **back pocket**: recorded options, not scheduled work.

## 9. `gen05_hybrid_matrix`: Phase 3, the headline robustness matrix  (2026-07-04 · `cd11f14`/`324a644` · `experiments/gen05_hybrid_matrix.md`)

- **Goal (prospective):** the reframed thesis headline on the repaired hybrid arena: does
  adversarial training against the strong scripted `targeted` attacker buy robustness to the
  held-out `gateway` attack, vs an identical non-adversarial control? 2 arms × 3 seeds × 400 ep
  (budget 1500, horizon 800, `--update-every 8`); pre-registered primary = pooled dD_gateway > 0,
  CI excl. 0, ≥2/3 pairings.
- **Headline results:** **primary NOT MET: sign reversed** (pooled dD_gateway = −192 ± 181,
  0/3 pairings positive). The dominating observation: **neither arm learned the task** -
  W(none) ≈ 4.6–4.8k vs greedy's 847 (~5.6× worse), and in-distribution dD_targeted = −20 ± 135:
  training against the attacker taught no measurable coping *even against that same attacker*.
  With W(none) near the 6.4k saturation ceiling, degradation is ceiling-compressed (both learned
  arms show smaller D than greedy: weakness masquerading as robustness).
- **What we learned:** the binding constraint moved. gen03/gen04 diagnosed the *adversary* as
  unable to learn; gen05 shows that on the hybrid rung the *protagonist* can't learn either at
  this budget/structure (hundreds of edge-level micro-decisions, thin credit, γ-myopia; Q_Spread
  ≈ 0.1 = the critic never discriminated). A robustness comparison between two incompetent
  policies is uninterpretable: competence is a precondition for the robustness question.
- **Thesis progression:** the full matrix pipeline (dual-arm training, held-out attack design,
  paired stochastic portfolio) ran end-to-end and produced a clean pre-registered readout in one
  day: the machinery is no longer ever the bottleneck. The interruption/resume during training
  also proved the checkpointing discipline (lossless mid-run recovery).
- **What it means for the thesis:** the honest empirical arc is now: adversarial VRP training
  fails two ways: the adversary can't learn to attack (gen03/04), and on decision-dense arenas
  the protagonist can't learn to act (gen05). Options recorded in the ledger: extend training /
  make the hybrid learnable (fewer decisions, denser credit) / move the matrix to dynassign where
  policies demonstrably reach within ~7% of greedy / freeze and write the diagnostic arc.
  Kilian to decide with the freeze (~Jul 16–18) in view.
- **Thesis fit:** Obj 5 (the robustness evaluation methodology is itself the contribution:
  pre-registration, held-out attacks, paired instances, ceiling-compression as an identified
  robustness-evaluation pitfall) + the results chapter's second act; strengthens the methods
  narrative regardless of what the final matrix shows.

## 10. `gen06_dynassign_matrix`: the competence-valid robustness matrix  (2026-07-05 · `cfabc90`/`0bc6ec3` · `experiments/gen06_dynassign_matrix.md`)

- **Goal (prospective):** the Phase-3 retake in the arena where policies demonstrably learn:
  {vanilla, scripted-adversarial (trained vs the new stochastic `pathrand` attacker)} × 3 fresh
  seeds × 800 ep on dynassign; primary = pooled dD under the fully held-out `targeted` attack,
  behind a pre-registered **competence gate** (the gen05 lesson).
- **Headline results:** **gate PASSED everywhere** (all six arms within +5.5…+7.0% of greedy
  clean: gen03's band, replicated); **primary NOT MET and significantly reversed**: pooled
  dD_targeted = **−881 ± 284** (0/3 pairings), dD_pathrand = **−775 ± 244** (worse even under its
  own training attacker), dD_random = −45 ± 221 (even), clean premium ≈ 0. Robustness ranking:
  **greedy (D 4921) > vanilla (5196–5882) > adversarially-trained (6361–6575)**.
- **What we learned:** with competence finally established, adversarial training against a
  strong scripted attacker **worsens** robustness to route-aimed attacks: in- and
  out-of-distribution: at zero clean cost. Leading mechanism, consistent with the campaign-wide
  SNR theme: constant attack floods the latency reward with unavoidable damage, diluting the
  learnable signal; the deficit surfaces precisely where queue compounding amplifies policy
  quality. And the most robust policy in the matrix is the reactive classical dispatcher -
  Ritzinger's reactive-dominance, measured in our own framework.
- **Thesis progression:** the campaign now closes a complete, coherent chain of evidence:
  the learned adversary cannot learn to attack (gen03/04) → the protagonist cannot learn
  decision-dense arenas (gen05) → and even the best-case configuration (strong fixed adversary,
  competent protagonist, clean evaluation) yields *negative* robustness transfer (gen06).
- **What it means for the thesis:** the definitive experimental finding, and a genuinely
  publishable-shaped one: *adversarial co-training as formulated does not confer robustness in
  stochastic-dynamic VRP, and the paper explains why*: reward SNR under zero-sum latency,
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

## 12. gen07 exploitability build + probes + BR gate → the INTERDICTION REDESIGN  (2026-07-06 · branch `gen07-contested`; docs `REDESIGN_INTERDICTION.md`, `experiments/gen07_contested_matrix.md`, `gen08_interdiction.md`)

- **Goal (prospective):** execute the exploitability reframe (entry 11): build the five learnability
  fixes, gate on cheap probes, then run the contested matrix to show adversarial training buys
  worst-case robustness (lower exploitability) even if not average-case.
- **What was built (all flag-gated, suite 109 green, historical modes preserved):** B6 contested
  arena (`--problem contested` = dynassign + route reach); B1 counterfactual twin reward
  (`--reward-baseline twin`, zero-sum-preserving difference reward, invariant verified); B2 entropy
  repair (absolute per-agent targets); B3 exposure/strength curriculum; B4-lite scripted-attacker
  population; B5 γ flag; B7 contested ERB generator; plus the BR-gate machinery (train a
  best-response attacker vs a frozen greedy victim). No matrix training launched.
- **Headline results (probes + gate, greedy rollouts / one gate training each):**
  (a) **capacity probe**: raising truck capacity DESTROYS the exploitability lever (it de-stresses
  the system: clean W 5908→1036, attack bite 4768→697, lever 217→−88); the lever is a STRESS
  phenomenon → keep capacity 1. (b) **powered stress sweep**: the λ=0.08 "sweet spot" was NOISE
  (85 ± 253, n.s.); the crude-unpredictability lever is thin (~2-7% of D) at every load.
  (c) **hybrid routing probe**: crude random routing makes damage WORSE (the reactive attacker
  re-aims). (d) **the decisive BR gate**: unfixed BR vs greedy = 0.84× random (gen04 replica); the
  CORRECTED gate (all fixes: counterfactual reward + entropy repair + γ) plateaus at **0.35×
  random**, entropy pinned ~2.2 while α collapsed to 0.08 → **near-zero Q-spread**.
- **What we learned (the unifying, deeper-than-entropy-pinning finding):** on a stressed queueing
  network every route-reach block causes similar cascading damage, so the attack landscape is
  **FLAT**: random is already near-optimal (4733 ≈ 96% of scripted 4920) and no learned adversary
  has an edge. **Flat-where-large, thin-where-differentiated** = a structural property of the
  congestion adversary (observable/reroutable/reversible), not a fixable optimisation issue. This
  explains gen03/04/06 mechanistically. **Self-correction on record:** the first BR gate used the
  UNFIXED attacker (a gen04 replica); the corrected gate is the fair test, and it also fails.
- **The pivot (the constructive move):** change the ADVERSARY, not tune the old one. Application 1's
  real threat is **interdiction/ambush** (hidden, irreversible, pre-committed) = a **Stackelberg
  security game**, where a deterministic router is maximally exploitable and the minimax
  MIXED-strategy router provably robust, and **SAC's entropy IS the mechanism** that produces it.
  Proven at the equilibrium level, before any training, on the real Kaliningrad graph:
  **deterministic routing 100% intercepted → mixed 17-33%** (`scratch/interdiction_game_probe.py`;
  gap 0.67-0.83 across OD pairs, tunable in K). loss_mixed is a *computable ground truth*.
- **Thesis progression:** the storyline becomes a 4-act POSITIVE arc (learnability wall → the
  pre-registered negative + why → the flat-landscape diagnosis → the interdiction redesign that
  provably works). All five objectives get positive evidence. gen07 ledger CLOSED (records the
  attempt + the finding); gen08_interdiction ledger opened (forward pre-registration); the
  interdiction build is ROADMAP Phase I. Decisions (Kilian 2026-07-06): Kaliningrad graph, single
  convoy first.
- **What it means for the thesis:** the negative campaign is not the end but the *motivation*: it
  shows mechanistically that congestion is the wrong adversary, and the redesign chooses the
  problem (interdiction) where the SACRED mechanism is the actual solution. Positive, defensible,
  literature-grounded (deployed security games), validated against a computable optimum.
- **Thesis fit:** Obj 1 (security game with computable equilibrium), Obj 3 (SAC entropy = mixed
  strategy; ATLA = fictitious play), Obj 5 (headline exploitability gap vs shortest-path + vanilla,
  validated against the equilibrium), Obj 4 (interdiction-aware base placement), ZST (transferable
  mixed strategy). The methodology (pre-registration, gates, held-out, the equilibrium oracle)
  carries over intact.
- **BUILD + FIRST POSITIVE RESULT (2026-07-06 late; branch `gen08-interdiction`; ledger G1/G2):**
  built the equilibrium oracle (I0), the interdiction env with the G1 fidelity gate passing (I1),
  the SAC-trainable env (I1b), and ran the I2 feasibility slice: **shortest_path 1.000 > vanilla
  0.275 > SACRED 0.235 (equilibrium 0.167); adversarial training cut interception 100%->23%,
  converging toward the computed equilibrium.** The project's first positive result: a deep-RL
  router learns a mixed strategy approaching the security-game equilibrium, ~4x less exploitable
  than deterministic classical routing. Honest caveat (I3 work): the symmetric instance gives a
  thin SACRED-vs-vanilla gap; asymmetric instances (non-uniform equilibria) are next.

## 13. gen08 interdiction: I2 slice + the I3 asymmetric-instance arc, ending in the programme's first sacred-vs-vanilla PASS  (2026-07-06/07 · branch `gen08-interdiction`, SHAs pinned per run in the ledger · `experiments/gen08_interdiction.md`)

- **Goal (prospective):** realise the interdiction redesign: show a SAC dispatcher trained
  adversarially (SACRED) learns a MIXED-STRATEGY route policy less exploitable to a committed,
  hidden interdictor than shortest-path routing AND than a non-adversarially trained SAC,
  approaching the computable minimax equilibrium (the gen08 pre-registered question).
- **Headline results:** I2 symmetric slice (33->71, 6 disjoint routes, K=1): **shortest_path
  1.000 > vanilla 0.275 > sacred 0.235**, equilibrium 0.167 (interception 100% -> 23%; the
  project's first positive result). I3 wave 1 (length-band vulnerability): primary FAILED
  (window reading, sacred ~ vanilla) while sacred << shortest replicated 3/3. B2 shared-edge
  instances (11 routes incl. k-shortest near-duplicates, walk mode): B2-P primary FAILED
  (fictitious-play cycling; average play 0.242-0.261 vs vanilla 0.429-0.445, 3/3); B2-P2
  (all-history BR mixture) failed WORSE (stale-mixture parking, telemetry-confirmed); **B2-P3
  (smooth fictitious play) PASSED the pre-registered primary on every clause: TAP ladder
  shortest_path 1.000 > vanilla 0.477 > uniform 0.455 > sacred 0.362 >> equilibrium 0.167
  (3/3 seeds + pooled)**; strong form (within 0.05 of equilibrium) NOT met (distance
  0.163-0.239).
- **What we learned:** (i) instance structure decides whether the non-adversarial control can
  IMITATE the equilibrium: length-band vulnerability correlates with travel cost and lets it
  (wave 1); shared-edge overlap forbids it provably (oracle: no cost-driven mixture below 0.467
  vs equilibrium 0.167), and there vanilla lands ABOVE uniform noise on 3/3 seeds
  (cost-calibrated mixing is predictability with extra steps). (ii) The fictitious-play
  discipline bracket: best-responding to the latest PURE commitment over-disciplines
  (large-amplitude last-iterate cycling); a uniform ALL-HISTORY mixture under-disciplines (goes
  stale; the travel-cost gradient parks the policy on one route, entropy collapsing 2.0 -> 0.7);
  SMOOTH fictitious play (softmax best response to the trailing-250 play, tau = 0.05 pinned by
  an oracle probe) is the stable middle and passed. (iii) Exploitability estimators for FP
  learners: realised-play windows and even policy snapshots sit mid-cycle; the trailing-averaged
  policy distribution (TAP) is the deployable estimator that stabilised. (iv) Short smokes
  validate plumbing, not slow-timescale dynamics (a 300-sortie smoke missed parking that begins
  ~sortie 1000; the 1000-sortie smoke with a pre-registered drift signature caught it).
- **Thesis progression:** equilibrium oracle (LP + best response + frontier
  `cost_constrained_value`), soft-interception heterogeneous instances, the route-walk trie with
  EXACT branch-product policy mixtures, the TAP metric, smooth-FP attacker modes, alpha/entropy
  telemetry; suite 131 green; three pre-registered result records with pinned SHAs and
  pre-committed exit criteria (a first for the project, and it worked: the dynamics chase was
  bounded to three iterations by design).
- **What it means for the thesis:** the headline positive claim now exists in its pre-registered
  POLICY form: adversarial training makes a deep-RL router ~2.8x less exploitable than the
  deterministic operational default, ~1.32x less than non-adversarial SAC, and less exploitable
  than uniform noise, at a quantified clean-cost premium, on the real Kaliningrad graph, scored
  against a computable equilibrium: plus a measured dynamics study explaining WHEN adversarial
  training converges (smooth two-sided FP) and when it does not.
- **Thesis fit:** Obj 1 (equilibrium + the FP dynamics study), Obj 2 (interdiction layer + walk
  trie), Obj 3 (SAC entropy as the mixing mechanism; ATLA as fictitious play, with the
  convergent realisation identified), Obj 5 (the exploitability ladder + cost-security frontier;
  the named non-adversarial control finally beaten in a pre-registered primary). Appears as the
  thesis's final Results act.

## 14. F1 single-convoy sweeps -> the MULTI-CONVOY PIVOT  (2026-07-07 · branch `gen08-interdiction` · `scratch/multiconvoy_*.py`, `experiments/gen08_interdiction.md`)

- **Goal (prospective):** broaden the banked B2-P3 single-convoy headline with the pre-registered
  wave A/C sweeps (F1), then, on hitting walls, find a direction where SACRED demonstrably works AND
  all five research objectives are met.
- **Headline results:** F1 (single-convoy symmetric K-sweep) LAUNCHED then KILLED: on the symmetric
  instance (uniform == equilibrium) adversarial training is a LIABILITY and sacred DESTABILISES (A-K1
  sacred TAP 0.38/1.00/0.40 vs vanilla ~0.31; seed 1 full collapse + alpha runaway). Multi-convoy
  ORACLE (no training): SOFT interception + a LOSS-AVERSE (mission-failure) objective gives SACRED a
  large, general win (N=2 gap median 0.48 across 20 OD pairs, growing with fleet size), a real ALNS
  to beat (cost-vs-risk coordination, SACRED dominates the frontier), and meets ALL FIVE objectives;
  a risk-neutral objective is the trap (dilutes to ~0).
- **What we learned:** (i) on a symmetric/flat game adversarial training destabilises (pick instances
  where vanilla provably cannot imitate the equilibrium); (ii) single-convoy cannot meet Obj-5's
  metaheuristic clause (ALNS = shortest-path); (iii) multi-convoy + soft + loss-averse resolves both,
  and the OBJECTIVE FUNCTION is load-bearing (loss-averse required and realistic); (iv) the win
  generalises across the graph and scales with fleet size.
- **Thesis progression:** the multi-convoy interdiction oracle (`scratch/multiconvoy_*.py`) as a new
  computable ground truth; the objective-spectrum + cost-frontier analyses; the direction re-pointed
  from single-convoy to multi-convoy to satisfy the full objective set.
- **What it means for the thesis:** the positive claim broadens from "single-convoy shared-edge"
  (banked, B2-P3) to "multi-convoy contested resupply where adversarially-trained randomised routing
  beats a coordinating classical metaheuristic under a realistic mission-failure objective", meeting
  all five objectives. Oracle-level proof done; the build + training is the next act.
- **Thesis fit:** Obj 1-5 all met (Obj-5 metaheuristic fixed; Obj-4 fleet composition added); the
  multi-convoy game is the thesis's final Results act. Single-convoy stays the proven headline.
- **Phase M build progress (2026-07-08):** M0 oracle proof + M1 env/oracle (G-M1 gate) + M2 ALNS
  baseline (reaches loss_det exactly) DONE and committed (suite 146 green); M3 trainer
  (`scripts/train_multiconvoy.py`: N-step sortie episode, oracle-BR-to-occupancy FP interdictor,
  vanilla control) built + smoked. Smoke (110->135 N=3, 1000 sorties): SACRED beats the optimal
  classical planner (0.645 < ALNS 0.904), stable (no collapse), BUT sacred ~ vanilla and far from
  the equilibrium 0.328 because the policy routes convoys INDEPENDENTLY, not the correlated
  stack-and-randomise. Next: an explicit "convoys-so-far per route" observation feature to make
  correlation learnable, re-smoke, then the full 3-seed launch (~50 min at 3-parallel).

## 15. Multi-convoy Phase M: Fork A instance + fleet-route headline + the learned-follower arc  (2026-07-08/09 · branch `gen08-interdiction` · `scripts/train_multiconvoy.py`, `scratch/multiconvoy_instance_screen.py`, `experiments/gen08_interdiction.md`)

> **UPDATE (see entry 16):** the multi-convoy fleet-route headline number in this entry (the transient
> single-seed **0.257**) is SUPERSEDED by the LOCKED 3-seed best-checkpoint **0.283 +/- 0.021**
> (gen09-HEADLINE, SHA `ad70a9c`, `experiments/gen09_multiconvoy.md`). Entry preserved as the Phase M
> record; entry 16 has the stabilisation arc + the locked headline.

- **Goal (prospective):** realise the multi-convoy oracle finding as trained SACRED: an adversarially
  trained dispatcher whose randomised joint routing (stack-and-randomise) beats a coordinating ALNS
  metaheuristic under the loss-averse mission-failure objective, meeting all five objectives.
- **Headline results:** (i) the disjoint N=3 instances DESTABILISE the leader (33->71 cycled / alpha
  runaway, landed ~ALNS) - and an oracle SCREEN of 72 disjoint OD pairs proved this is STRUCTURAL:
  disjoint routes give a near-uniform leader equilibrium (H/lnR >= 0.97) = flat fictitious-play
  landscape. Asymmetry (a non-uniform leader, an FP gradient) REQUIRES shared edges. (ii) Fork A: the
  screen picked **62-97 k_extra=8** (shared-edge, 12-route menu-select; asymmetry H/lnR 0.63, margin
  ALNS/eq 3.2x, stack mass 0.97). There the LEADER is stable and near-equilibrium: **fleet-route
  (leader-mix + structural fleet stacking) TAP 0.257 (1.19x eq 0.216) << ALNS 0.699 << vanilla ~0.945**
  - BANKED as the multi-convoy headline. (iii) The learned-follower bootstrap (make followers LEARN to
  copy, not copy structurally) hit a chicken-and-egg (the critic never experiences the rare stack
  reward); a six-attempt fix chain made the critic-side learned correlation weight `follow_w` CLIMB
  monotonically (the milestone: the critic can be made to value emergent coordination), tail-average
  0.482 beats ALNS + vanilla, but coordination SATURATED weak (stack ~0.18) and 0.482 loses to the
  structural 0.257. Fallback banked; learned coordination is the secondary Obj-3 result.
- **What we learned:** (i) disjoint = structurally uniform leader (flat FP); pick shared-edge asymmetric
  instances for a learnable leader. (ii) A joint/correlated objective needs the coordination signal
  EXPLICIT and UNDILUTED at the scoring head, AND the CRITIC must value coordination - the actor cannot
  follow what the critic won't rank (follow_w climbing is the diagnostic). (iii) To learn a rare joint
  behaviour, the critic must EXPERIENCE it: demonstration bootstrapping (forced-copy warmup vs a FROZEN
  mixing leader) + prioritised replay of the rare stacked transitions (ERB / Obj-3). (iv) Zero-sum FP
  cycles by construction; judge on the stationary-tail time-average, not per-eval stage play.
- **Thesis progression:** the multi-convoy env + oracle + ALNS baseline (Phase M1/M2), the route-index
  menu-select head (scales to shared-edge, no walk trie), the two-role-alpha temperature split, the
  learned undiluted route-correlation term on actor + critic, the Fork-A instance screen. Suite 146
  green; all additive/flag-gated (campaign byte-identical).
- **What it means for the thesis:** the multi-convoy headline is a positive Obj-5 result on a computable
  equilibrium (SACRED's randomised routing beats the SOTA metaheuristic AND non-adversarial SAC), with
  an honest caveat (structural stacking) and a mechanistically-rich secondary result (learned emergent
  coordination proven possible, `follow_w` climbing, but saturating below the structural version). Two
  banked headlines now: single-convoy B2-P3 (0.362) and multi-convoy fleet-route (0.257 << ALNS 0.699).
- **Thesis fit:** Obj 1 (multi-convoy asymmetric zero-sum game + computable equilibrium + the FP
  time-average framing), Obj 2 (multi-convoy env + menu-select), Obj 3 (SAC + ATLA-as-FP + ERB/demo
  bootstrapping, now load-bearing in the learned-follower arc), Obj 4 (fleet composition, N a lever),
  Obj 5 (the fleet-route ladder shortest 0.973 > vanilla 0.945 > ALNS 0.699 >> SACRED 0.257 -> eq
  0.216). The final multi-convoy Results act; single-convoy stays the proven core.

## 18. The overnight programme: gen11 decomposition, F3 SBO, gen12 sweeps, ZST step 0  (2026-07-10 night · branch `gen08-interdiction` · ledgers gen11_menuhead, f3_sbo_demonstrator, gen12_sweeps, zst_step0; night report `NIGHT_REPORT_2026-07-10.md`)

- **Goal (prospective):** execute CRITIQUE_PREFREEZE §8 autonomously overnight (Kilian's launch
  authority + decision rules: gen11 pass = new headline; no design chasing beyond gen11), in his
  modified order: gen11 first, then F3 SBO, eval rows, M5 sweeps, ZST step 0, docs hygiene.
- **Headline results:** (i) **gen11 (4 arms x 3 seeds): NO PASS; the decomposition is the
  product.** Features arm 0.443 / identity arm 0.476 = the plateau, and both arms' added head
  parameters silently stayed ~0 (param groups inherited the base lr: an optimisation-scale no-op,
  so the head-term CONCEPT is untested); leader-only-push arm 0.980 flat = the follower-push
  hypothesis FALSIFIED spectacularly (single-state replay -> softmax saturation, alpha -> 295;
  follower pushes are load-bearing state diversity). (ii) **F3 SBO demonstrator (Obj-4, reduced
  form): MET** - 450-design placement x fleet space, SurrogateMLP on cheap features incl. the
  closed-form harmonic-vulnerability aggregate: held-out Spearman 0.894, argmin regret 0.0000.
  (iii) **gen12 sweeps (Obj-5 varied disruption): SACRED < ALNS in 10/10 cells** (K {1,2,3} x
  N {2,3,5} x two ODs) with the margin GROWING with fleet size; **the held-out OD 35-159 reaches
  1.09-1.69x its equilibrium POST-FIX (N3K1 best-ckpt 0.261 = 1.27x eq)** -> the 0.447 plateau is
  INSTANCE-SPECIFIC to 62-97, not architectural. (iv) **ZST step 0: pre-registered scoping
  negative** - the home-trained policy beats shortest/uniform on held-out 110-135 (0.699) but
  loses to a random-init net (0.584): with no observable threat map there is no transfer
  mechanism; ZST needs map-conditioned multi-instance training (step 1). Also: fleet-cost column
  (SACRED's premium = the equilibrium's own), vanilla 3-seeded (0.855 +/- 0.003) + its
  best-checkpoint row, docs number-hygiene (ledgers = sole number source).
- **What we learned:** (i) instance asymmetry substitutes for head discriminability: where the
  equilibrium has a sharp FP gradient (H/lnR 0.44), honest embeddings suffice; where it is
  flatter (0.63), the pre-fix identity hash had been doing the work. (ii) Added head parameters
  need their own lr scale or they are silent no-ops. (iii) Replay-state diversity is load-bearing
  for a shared menu actor (single-state SAC = saturating bandit). (iv) The vectorised
  mission-objective matrix (one matmul) moved the naive-oracle wall from K=3 to ~K=4-5 (RAM-bound):
  the scaling figure needs restating.
- **What it means for the thesis:** all five objectives now have at least a demonstrated form
  (Obj-4 was the last); Obj-5 has its disruption curves + a held-out replication STRONGER than the
  headline instance; the morning decision (recommended): 3-seed ho_N3K1 (35-159) as THE post-fix
  multi-convoy headline candidate, which would retire the pre-fix/post-fix asymmetry entirely.
- **Thesis fit:** Obj-4 (F3), Obj-5 (gen12 curves + held-out), Obj-1/3 (gen11's two mechanism
  findings), ZST (honestly scoped with a measured boundary + the designed step 1).

## 17. The 2026-07-09 audit: the node-ordering bug, the fix, and the gen10 post-fix re-runs  (2026-07-09 · branch `gen08-interdiction`, fix SHA `e9acb56` · `CRITIQUE_INTERDICTION.md`, `experiments/gen10_postfix.md`)

*(Chronologically AFTER entry 16; placed above it so the two locked-headline entries stay adjacent
to their supersession notes.)*

- **Goal (prospective):** Kilian requested an examiner-grade critique of the whole interdiction
  programme plus a codebase audit; then approved fixing what was found and re-running the banked
  headline configs (gen10, pre-registered before launch).
- **Headline results:** (i) **A project-wide representation bug found and fixed**: `featurize_state`
  sorts node ids, every consumer indexed by dict insertion order, so every network ever trained
  read a fixed permutation of the wrong nodes' embeddings (demonstrated: convoy at node 62 reads
  node 167's row). Fix = `node_index_map` single source of truth + 3 regression tests; suite 149
  green. (ii) **Exact re-evaluation of the gen09 headline** (the saved checkpoints, exact occupancy
  distributions): best-checkpoint TAP **0.295 +/- 0.024**, not the MC 0.283 (min-selection on
  sampling noise). (iii) **gen10-SC (single-convoy B2-P3 re-run, post-fix): PASSED every clause,
  pooled sacred TAP 0.276 vs vanilla 0.480** (banked: 0.362 vs 0.477): ~44% of the residual
  equilibrium gap was the bug. (iv) **gen10-MC (multi-convoy re-run): REGRESSED to 0.447 +/- 0.029**
  (prediction violated, reported as measured; one seed showed a 900-sortie softmax-saturation park
  with alpha runaway): the Obj-5 ordering still holds post-fix (0.447 << ALNS 0.699 << vanilla
  0.859) but the equilibrium margin worsened; confounds = menu-head discriminability under correct
  embeddings (the old permutation acted as an accidental route-identity hash), the role-alpha
  target fix, and a config tuned under the bug.
- **What we learned:** (i) representation-indexing consistency needs an explicit contract test, not
  convention; (ii) a bug can flatter learning (the permutation made 12-way route memorisation easy)
  so "suite green + result improved" never certifies representations; (iii) the single- vs
  multi-convoy split cleanly isolates WHERE the fix helps (walk-mode next-hop) vs where the
  architecture now binds (mean-pooled menu head on overlapping routes).
- **What it means for the thesis:** single-convoy headline strengthens to 0.276 (pending Kilian's
  confirmation it supersedes 0.362); the multi-convoy citable number stays the pre-fix banked
  best-checkpoint (exact 0.295 +/- 0.024 at `ad70a9c`) with the caveat disclosed, until the
  proposed gen10-MC2 diagnostic (2400 sorties, role-alpha fix flagged off; needs Kilian's go)
  resolves the regression. Full critique (objective-fit scoring, triviality analysis, SBO/ZST/
  scaling outlook, ranked pre-freeze programme): `CRITIQUE_INTERDICTION.md`.
- **Thesis fit:** Methods (the audit + contract-test lesson), all Results chapters (numbers move),
  Obj-3 (the role-alpha target correction), the honesty/self-correction narrative (a second
  retraction-grade correction handled by pre-registration).

## 16. gen09 multi-convoy leader-stabilisation arc -> the LOCKED best-checkpoint headline  (2026-07-09 · branch `gen08-interdiction` · `scripts/train_multiconvoy.py`, `src/baselines/fp_dynamics.py`, `experiments/gen09_multiconvoy.md`)

- **Goal (prospective):** the entry-15 fleet-route headline was a single unsaved seed-0 run (0.257);
  Kilian flagged that across seeds it varied (0.257/0.433/0.517/0.382). Turn that into a tight, saved,
  reproducible 3-seed headline by stabilising the leader (kill the across-seed variance), pre-registering
  and committing every attempt before running.
- **Headline results:** three "hold-the-tail" stabilisation attempts FAILED, and the failure IS the
  finding. STAB-1 (diffuse attacker tau 0.15): leader never concentrates (uniform). STAB-2 (sharp tau
  0.05): leader concentrates to the equilibrium hedge EARLY and TIGHT (best-ckpt 0.277 +/- 0.007) then
  DRIFTS to uniform. STAB-3 (ported the exact B2-P3 smooth-FP discipline into a shared
  `src/baselines/fp_dynamics.py`, used by both trainers): SAME drift (best-ckpt 0.293 +/- 0.029). So the
  leader's low exploitability is a REPRODUCIBLE TRANSIENT, not a stable fixed point (uniform is a
  competing FP attractor; the last iterate over-trains toward it - inherent last-iterate fictitious-play
  cycling, the single-convoy B2-P failure mode). **Decision (Kilian): stop the knob-tuning chase; resolve
  it the standard single-convoy way = BEST-CHECKPOINT selection.** The **LOCKED** definitive run
  (gen09-HEADLINE, SHA `ad70a9c`, 3 seeds, 1200 sorties, full saving incl. per-eval actor checkpoints):
  **fleet-route best-checkpoint TAP 0.283 +/- 0.021**. Ladder: shortest 0.973 > vanilla ~0.945 >
  ALNS-forced-stack 0.912 > ALNS 0.699 >> **SACRED 0.283** > equilibrium 0.216 (2.5x ALNS, 3.3x vanilla,
  1.31x eq). Fairness row: ALNS is FREE to stack but SPREADS by choice (0.699 < forced-stack 0.912), so
  SACRED's win is the RANDOMISATION, not a stacking privilege.
- **What we learned:** (i) the sharp adversary REPRODUCIBLY produces the equilibrium hedge (validated,
  tight across seeds) - the mechanism works; (ii) but it is a best-checkpoint transient, and
  best-checkpoint selection by exploitability (the deployable object; the last iterate is misleading
  under minimax) is the honest, standard resolution, with the drift SAVED and DISCLOSED, not hidden;
  (iii) the earlier seed spread was the drift caught at different training lengths, not irreducible
  variance; (iv) a false diagnosis is worth recording (the block-held/all-history attacker was NOT the
  cause - porting the true-smooth B2-P3 discipline changed nothing).
- **Thesis progression:** a shared, proven smooth-FP helper (`fp_dynamics.py`, one implementation for
  single- and multi-convoy); the leader-alpha floor; per-eval checkpoint saving (best-checkpoint is a
  re-evaluable ARTEFACT); the ALNS-forced-stack fairness metric. Every attempt pre-registered + committed
  before running (11 commits, `92e2d8a` and prior). Suite 146 green.
- **What it means for the thesis:** this is a real, chapter-worthy result and a rigorous story: the
  multi-convoy Obj-5 headline (SACRED's randomised routing beats the SOTA metaheuristic AND non-adversarial
  SAC, approaching a computable equilibrium) is LOCKED at 0.283 +/- 0.021, with an honest, disclosed
  last-iterate-instability caveat resolved by best-checkpoint selection - the same discipline single-convoy
  used. The transient/best-checkpoint finding is itself a contribution (WHERE and WHY minimax routing
  converges vs over-trains). No more leader experimentation (Kilian); write-up next; scaling tier only if runway.
- **Thesis fit:** Obj 5 (the LOCKED ladder + the fairness row), Obj 1 (the FP last-iterate-cycling / transient
  characterisation), Obj 3 (SAC-entropy-as-mixed-strategy under adversarial pressure; best-checkpoint
  discipline). `experiments/gen09_multiconvoy.md` is the authoritative locked record. The learned-follower
  bootstrap (entry 15) stays the banked Obj-3 SECONDARY.

## 19. The node-ordering fix, gen10-13 re-runs, and the morning steps  (2026-07-09/10 · fix SHA `e9acb56` · `experiments/gen10_postfix.md`, `gen13_lock.md`, `CRITIQUE_INTERDICTION.md`)
- **Goal (prospective):** an examiner-grade audit of the interdiction programme found a project-wide
  representation bug (`featurize_state` sorts node ids; every consumer indexed by dict insertion order),
  so every trained net had read a fixed permutation of the wrong nodes' embeddings. Fix it and re-run.
- **Headline results:** fix landed (`node_index_map` single source of truth + regression tests, suite
  149->161 over the arc) with a role-alpha Bellman-target fix and exact fleet-route evaluation. **gen10-SC
  (single-convoy re-run): PASSED, pooled sacred TAP 0.276 vs vanilla 0.480** (banked 0.362/0.477; ~44% of
  the residual equilibrium gap was the bug). **gen10-MC / gen10-MC2 (multi-convoy): REGRESSED to a
  reproducible 0.447 plateau** (the pre-fix identity-hash had been supplying route discrimination the
  mean-pooled menu head lacks on the flat 62-97 equilibrium). **gen13-lock: the multi-convoy headline
  moved POST-FIX to 35-159 (best-ckpt TAP 0.274 +/- 0.025)** - a held-out-screened instance whose
  asymmetry the honest embeddings suffice for; the two-headline pre-fix/post-fix asymmetry retired.
  gen11b: the head-term learning-rate fix works - identity capacity confirmed (E' 0.295 = pre-fix) but
  transferable features recover less (B' 0.408) where the equilibrium is flat. Scaling re-measured with
  the vectorised objective matrix: N3K3 solves in 23 s, so the wall-clock scaling claim was retired.
- **What we learned:** representation-indexing consistency needs a contract test not convention; a bug
  can flatter learning ("suite green + result improved" certifies nothing about representations); never
  mix pre-/post-fix ladders; added head params need their own lr scale; instance asymmetry substitutes
  for head discriminability.
- **Thesis fit:** both headlines now on corrected code (Obj-5); the bug arc is a first-class methods
  contribution; Obj-1 gains the FP-transient characterisation.

## 20. The expansion programme: evidence completion + the SBO stack + ZST step 1  (2026-07-10/11 · `DIRECTION_EXPANSION.md` · ledgers gen14_evidence, gen15_generalist, d1/a3/d3/a2/a4/b4/d2, f3)
- **Goal (prospective):** Kilian "disregard thesis writing"; expand computationally so all five
  objectives have trained evidence, ZST is realised at scale, and SBO becomes a holistic stack.
- **Headline results:** **C3 (gen14):** both headlines n=10 CIs (MC 35-159 0.256 [0.246,0.266]; SC
  paired dD 0.175 [0.137,0.213] excl 0, 10/10 - the statistical weak point closed) + native 35-159
  vanilla/forced-stack/fleet-cost rows. **A1 (gen15): the first TRAINED zero-shot transfer** - a
  map-conditioned generalist (edge-vulnerability observation + transferable per-route cost/vuln head
  features) routes held-out ODs at **1.59x their own equilibria** (beats loss_det 17/18). **The SBO
  stack (Obj-4, now the most complete objective):** F3 regression (Spearman 0.894, argmin regret 0),
  D1 acquisition loop (median 33 evals to the optimum vs random never), D2 hardening tier (equilibrium
  L1-shift 0.29 = tier coupling), **D3 composite** (surrogate over the TRAINED policy Spearman 0.959;
  policy-vs-oracle design-target corr 0.768 = designing against the deployed policy differs from the
  equilibrium). A3 amortisation (honest: LP faster+exact at K=1, so the ZST case is deployment
  structure + D3, not wall-clock). A4-core (matrix-free submodular greedy BR, verified, reaches K=5).
  B4 (correlated interception: independence is conservative). B0 (obs-staleness fix). **A2 = the
  direction finding: a single-source-graph generalist transfers across OD PAIRS but NOT across GRAPHS
  (ties random on a different graph)** -> multi-graph training is the route to cross-city ZST.
- **What we learned:** the generalist conditions on the map (not memorises); the GNN overfits to one
  graph without variety; the SBO loop prices the deployed policy where no LP can.
- **Thesis fit:** all five objectives now have trained/demonstrated evidence; ZST + the SBO stack are
  the payoff act; the ZST-vs-LP framing (deployment/scale/D3, never wall-clock) is the sharpest defence.

## 21. ZST at CITY scale + the boundary gates (gen16-19)  (2026-07-10/11 · ledgers gen16_multicity, gen17_lastiterate, gen18_learnedfollower, gen19_b1lite1)
- **Goal (prospective):** realise cross-CITY zero-shot transfer (the A2-predicted multi-graph fix),
  and spend the pre-committed boundary gates (last-iterate convergence; learned coordination; restore
  the D).
- **Headline results:** **gen16: the first CROSS-CITY zero-shot transfer** - one policy trained on
  Kaliningrad + East London + Istanbul routes fleets in never-seen **Gdansk at 1.677 +/- 0.072x its
  equilibria** (beats loss_det 17/18), and the A2-rescue row confirms the mechanism (1.90 vs random 2.43
  on the graph where single-source tied random). Transfer ladder: same-graph OD 1.59 -> held-out city
  1.68 -> single-source cross-graph ~random. **gen17/C4 FAILED** the hold-the-tail bar (annealed
  smoothing delays but does not prevent the drift; four failed attempts across two instances/eras make
  the equilibrium-transient finding INHERENT, best-checkpoint discipline final). **gen18/C2 FAILED**
  (follow_w trained to 2.93 - the lr fix worked - yet followers still collapse to fixed routes: the
  structural-stacking caveat is a real boundary, future work is exploration-side). **gen19/B1-lite-1
  PASSED = the D restored + solved:** the first SACRED game with within-episode dynamism (pattern-of-life
  interdictor); SACRED history-aware **0.050 ~ history_opt 0.049** (3/3), no-window causal control 0.148
  = iid_eq, worst-case row 0.219 ~ equilibrium 0.206 (no fragility).
- **What we learned:** graph variety is what makes cross-graph transfer work; the equilibrium transient
  and structural stacking are inherent boundaries (measured, not artefacts); against a pattern-of-life
  adversary a history-aware defender reaches the dynamic optimum, causally attributed to the window.
- **Thesis fit:** ZST becomes a boundary-mapped result (OD -> city -> construction-family); gen17/18 are
  measured boundaries (Obj-1 dynamics, Obj-3 coordination); gen19 returns the D of SDVRP in the
  security-game register (Obj-1/2). Next: F2 (Obj-1 learned agent), the ZST hardening rows, C1.
