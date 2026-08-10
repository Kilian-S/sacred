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

## 22. The disjoint-baseline finding, Block R repair, and gen26: the K-to-min-cut act  (2026-07-16 · branch `gen08-interdiction` · `CRITIQUE_16-07-26.md`, ledgers gen26_kboundary + seven R0a appendices)

- **Goal (prospective):** re-establish and verify the lost 2026-07-15 finding (a memory index
  entry pointed at critique/memory files that were never committed); repair what it invalidates;
  and relocate the thesis's positive static claim to a regime that survives it (Kilian's Block R
  decision, full autonomous launch authority, overarching goal = a positive, scientifically
  valid claim).
- **Headline results:** (i) **The finding is real (oracle-exact, `scratch/disjoint_baseline_probe.py`):
  uniform-stack over the edge-disjoint routes (the menu's own max-flow prefix) achieves the exact
  SC equilibrium (0.167 vs trained 0.276-0.362), matches the MC headline (0.250 vs 0.256
  [0.246, 0.266]), and transfers zero-shot at 1.13x eq (vs generalist 1.73) with no training —
  the ladders' "uniform" anchors were padded-menu strawmen.** Rows + fleet-cost columns + binding
  wording rules folded into all seven affected ledgers (R0a). (ii) **Surviving K=1 edges:** the
  policy's fleet cost tracks the equilibrium's (90.4 vs 91.0; heuristic 99.5) and it allocates
  near-equilibrium mass to the disjoint core without being told the structure (0.62 vs eq 0.703
  vs uniform 0.333; zero-shot 0.54-0.89 tracking per-instance eq) — R0b. (iii) **gen26 (the
  rescued static act): at K = m-1 SACRED beats BOTH max-flow heuristic variants on the exact
  yardstick (K=3 n=3: 0.664 +/- 0.018 vs 0.737/0.738; eq 0.604; STRONG bar met), and past the
  exact-LP wall (71-33, m=6, greedy yardstick, fidelity <= 1.8% at K <= 3) it beats
  uniform-disjoint at K=5 (0.667 +/- 0.016 vs 0.705, 3/3) and BOTH variants at K=6 (0.718 vs
  0.766/0.800, single seed)** — the pre-registered saturation expectation was wrong in SACRED's
  favour. Boundary-map figure: `assets/k_boundary_map.png`.
- **What we learned:** (i) baseline-completeness must be pre-registered like metrics (the
  strongest practitioner heuristic was never in any comparison set for 25 generations); (ii) the
  screens' det/eq criterion measured where DETERMINISM fails, not where naive randomisation
  fails; (iii) learning pays in a measurable band: nothing below K = m-1, both-variant wins at
  K = m-1 (exact), uniform-variant wins at K = m-1 past the wall, sole survivor at K = m;
  (iv) the last-iterate drift that plagues every K=1 result nearly vanishes at high K (finals ~
  bests) — plausibly the uniform attractor weakens when coverage pressure is high everywhere;
  (v) a claim of record-loss magnitude must be committed in the session that produces it.
- **Thesis progression:** the greedy-BR (matrix-free) trainer mode (flag-gated, +6 regression
  tests, suite 167; K=5 exact matrix would be 2.2 GB, K=6 14 GB — labels genuinely absent); the
  R0 probes as reproducible artefacts; the boundary map as the Act-IV product.
- **What it means for the thesis:** the static headline claim moves from "beats naive
  randomisation at K=1" (false) to **"trained where neither exact solvers nor naive heuristics
  can follow"** (measured, pre-registered, same-yardstick), with the K << m regime conceded to
  the heuristic in one honest sentence and the concession itself made a contribution (the
  boundary map).
- **Thesis fit:** Obj-5 (the rescued comparative act + the K-curve as "varied disruption" done
  right); Obj-1/3 (the drift-vanishes-at-high-K dynamics observation); Methods (baseline
  completeness as a named discipline beside pre-registration).

## 23. gen27: the dynamic generalist — zero-shot dynamic hedging on a never-seen city  (2026-07-16 evening · `experiments/gen27_dynamic_generalist.md` · `scripts/train_dyn_generalist.py`)

- **Goal (prospective):** compose gen19 (pattern-of-life exploitation, single instance) with gen16
  (multi-city transfer) into the rescued ZST act: ONE history-aware policy, trained on three
  cities, evaluated zero-shot on held-out Gdansk against each instance's computable dynamic
  yardsticks — the register where every static method is provably capped.
- **Headline results:** **PRIMARY + STRONG PASS, 3/3 seeds: pooled held-out ratio-to-cap 0.639
  +/- 0.025** (per-seed 0.605/0.644/0.666, beats the cap on 6/6, 5/6, 5/6 ODs; 1.74x the exact
  dynamic optimum). Two integrity amendments were ledgered BEFORE results: per-seed refs (an
  LP-degeneracy wobble of ~1-2% in the cap across processes, discovered in the pool logs) and
  MEASURED static baselines (the local-search static optimum improves the cap by only 2-5%, so
  the pass beats every static object as measurement, not construction). Naive-dynamic
  reconciliation: full-menu anti-repeat FAILS (1.37x: shared segments defeat naive avoidance);
  the composed disjoint+anti-repeat rule (0.50-0.61x) bounds the act from below; binding wording
  recorded in the ledger. Causal no-window control + worst-case row in flight.
- **What we learned:** (i) dynamic hedging TRANSFERS: the anti-repeat weight trains to -20
  (route-frequency avoidance) and works on maps never seen; (ii) the composed heuristic needs
  BOTH insights told to it — the policy discovered both, label-free; (iii) the mild-drift
  pattern persists (seed 1 final 0.816) and select-on-train handles it; (iv) my own experiment
  needed the same baseline-completeness treatment the static acts got — applied pre-results this
  time.
- **Thesis fit:** the aim's ZST sentence ("policies that standard algorithms cannot achieve") is
  now LITERALLY true in the dynamic register with trained, multi-seed, zero-shot evidence
  (Obj-1's D + aim-level ZST); the flagship positive act of the storyline.

## 24. B2 goes live: the LLM benchmark's first real conversation  (2026-07-16 night · `experiments/b2_llm_benchmark.md` · in progress)

- Prof. Angeloudis's green light + direct gateway access (port 8080 now reachable; no tunnel).
  Design finalised with Kilian in-conversation: llama-3.3-70b + qwen3-27b, UNHINTED only, three
  instances, ~75 conversations/model, on-box tmux runner (the prof's robustness tip), fine-tuning
  suggestion recorded as spin-out future work. **First live conversation (llama, register (b),
  35-159): the model committed uniform-0.125 over routes 4-11 — the MOST-overlapping cluster —
  scoring 0.663 (3.2x eq; worse than uniform-menu-stack 0.442), while the post-probe shows it can
  NAME a near-correct independent set when asked. Knowledge present, strategic application
  absent: the dissociation the benchmark exists to measure.** Transcript:
  `scratch/b2_livetest_llama_transcript.txt`; comprehension gate 1/3 (reported). Qwen live test +
  the overnight batch follow Kilian's review.

## 25. B2 COMPLETE: the LLM benchmark banked on two instances  (2026-07-17/18 · `experiments/b2_llm_benchmark.md`, `experiments/regime_decision_table.md`)

- **Headline results:** on 35-159, NEITHER model calibrates a mixed strategy (register (b)
  0.52-0.60, worse than the uniform-menu stack 0.442, far above the disjoint heuristic 0.250 and
  SACRED 0.256) despite passing comprehension probes and NAMING near-correct independent route
  sets on demand — the knowledge/application dissociation is the finding. Register (c): partial
  recovery via emergent anti-repeat (qwen best 0.059 vs SACRED 0.050; llama 0.177). The Gdansk
  zero-shot cell REPLICATES the calibration failure on a never-seen city, with a model x
  instance interaction (qwen approaches the heuristic there); neither model approaches the
  gen27 trained policy in the dynamic register. Binding two-instance wording recorded in the
  ledger; `regime_decision_table.md` is the practical synthesis (which policy for which
  adversary type x budget).
- **Thesis fit:** the differentiator act; independent support for the central mechanism
  (calibrated randomisation is exactly what language agents lack unaided); one subsection +
  one ladder column, workshop spin-out recorded.

## 26. gen28: the aerial act — three measured negatives, a retired positive, and the theatre exhibit  (2026-07-16 -> 18 · branch `gen28-aerial`, worktree `../sacred-aerial` · `experiments/gen28_aerial.md` + its 2026-07-19 appendix)

- **Goal (prospective):** the free-flight act (continuous coverage axis; the map-conditioning
  de-confounder; Kilian's then-standing "trained aerial result = MUST-HAVE").
- **What happened (five game revisions, all pre-registered, all disclosed):** the oracle screen
  arc (proximity exposure, standoff zones fixing the terminal-funnel degeneracy, complete lane
  families, the structure-not-firepower finding, grid-convergence certificates) delivered a
  genuinely non-degenerate game family. The trained acts then failed in sequence, each with a
  mechanism: v2.2 menu-N=1 = the saturating-bandit cell, measured; v2.3 walker = credit
  starvation; v4.0-dyn = the corridor-collapse information-structure finding (without the
  doctrine model the only expressible window behaviour is anti-repeat, which is provably bad on
  aerial layouts; with it, a two-line dodge is near-optimal — and the road gen19/27 successes
  are partly architecture-game coincidences, a caveat the flagship's wording must carry). The
  v3.0-3.2 fleet line learned and replicated a thin Tier-1 pass, RETIRED 2026-07-19 by the
  baseline-completeness appendix (the act's own tabular-FP row 0.555 + a best-5-route stack
  0.600 beat SACRED 0.734-0.746: a tie with the naive frontier, the gen26 K=6 pattern).
- **What survives:** zero-shot frontier-MATCHING (vs-naive 1.01-1.05: one policy re-derives the
  best rule's performance on sight, the amortisation claim in a third domain); the boundary/
  structure screen findings; the vector-theatre real-map render (status revised 2026-08-09:
  imperial-sacred is the visual centrepiece); and the extension of the preconditions chain to
  the air.
- **Thesis fit:** the aerial cells of the boundary map; Obj-2's "visual, interactive" clause
  finally earned; NO aerial trained-positive sentence is licensed.

## 27. gen29: the closing experiment — the coordination moat is real, and self-play cannot cross it  (2026-07-18 · branch `gen29-multiod`, worktree `../sacred-gen29` · `experiments/gen29_multiod.md`; probes `738ddd1`/`e6c29e2`; brief `GEN29_MULTIOD_HANDOFF.md`)

- **Goal (prospective):** Kilian's confirmed CLOSING EXPERIMENT: three supply streams sharing
  corridors (K=1, mission objective), the one register where the naive ceiling provably lifts.
- **Headline results:** (i) **The oracle half is the only gap in the whole project that
  survives a complete hostile baseline family** — screened over 55 non-degenerate cells, the
  joint equilibrium sits a median 31% below even the oracle-fitted best-m-pairing cap (37-55%
  on the screened best; coordinated napkin rules are WORSE than independence on 15/15 probe
  triples; K=2 compresses the gap, so structure, never firepower). (ii) **The trained half
  FAILED both pre-registered tiers on every seed**, and the blinded causal control came out
  EQUAL to the sighted policy (2.07 vs 2.02): the coordination channel carried nothing — the
  gen18 boundary replicating in the final register despite an architecture built specifically
  to avoid it. (iii) Single-instance diagnostic: self-play beats the oracle-fitted cap on one
  instance (1.41x eq) but floors ~1.4x and oscillates (FP cycling); the generalist was
  density-starved (~375 sorties/instance vs ~7000 to floor). (iv) OPEN (Kilian's call, the
  last computational decision): the pre-committed dense-credit re-aim and/or the distillation
  control as a "locate the wall" pair, or close.
- **What it means for the thesis (the boundary-map closing sentence, Kilian's framing):**
  *below a measurable boundary, contested routing needs no learning (two-line rules are
  near-optimal, proven across three domains); in the one register where no hand-built rule,
  however oracle-assisted, can express the optimal defence, model-free adversarial self-play
  at thesis scale also fails to capture it, with a causal control confirming the channel does
  not carry.* The map closes with a measured edge; gen27 remains the flagship positive.

## 28. gen30: security-aware facility location — Obj-4 realised at the placement tier, oracle-only  (2026-07-19 · branch `gen08-interdiction` · `experiments/gen30_secure_flp.md`; scripts `scratch/gen30_{secure_flp,analysis,surrogate}.py`; figures `assets/gen30_*.png`)

- **Goal (prospective):** the supervisor-direction act (2026-07-19): price depot designs
  against the operational security game, entirely oracle/eval-only (no training anywhere),
  pre-registered before any analysis code; the machinery anchor (147 -> 212,188,195)
  reproduces the committed gen29 headline exactly before any new number was read.
- **Headline results:** (i) **Component A:** the (cost, security) frontier is computable
  exactly at ~0.3 s/design; the cost-optimal depot's security premium is demand-dependent
  (0-206% across six seeded draws, 12-49% typical; fail branch fired and reported on 2/6),
  the knee recovers most of it for 3-7% extra cost, site spread x2.5-5.3 everywhere, and the
  design RANKING is deployment-robust (Spearman v_joint-vs-cap 0.865-0.923). (ii) **Component
  B (the headline):** dual-servability redundancy, which classical nearest-assignment FLP
  prunes by construction, is worth a median 25% (max 61%, >=5% on 35/40 payoff-blind pairs)
  at the coordinated joint optimum on the primary instance; the mechanism is corridor
  DIVERSITY (value ~ -0.56 vs depot-corridor Jaccard); **redundancy and coordination are
  measured COMPLEMENTS** (under m<=4 napkin deployment the median value is -10% on the
  deep-moat instance, 13/40 positive, while on the shallow-moat Gdansk draw it is freely
  harvestable) - the gen29 correlation moat reproduced and WIDENED (46% median gap-vs-cap on
  redundant designs) at the strategic tier; operating premium +49%/+72% at matched openings
  (security is paid in operations, not construction). (iii) **Demand-side floor:** a Gdansk
  target whose routes share 7 mandatory edges pins EVERY design at >= 0.330 (verified
  mechanically): a single-approach target caps security before any facility decision.
  (iv) **Component C:** the SurrogateMLP prices designs from 12 cheap features at held-out
  Spearman 0.870 (bar 0.8 MET; argmin rank 4/129); cross-city calibration does not transfer
  (0.46, disclosed boundary).
- **What we learned:** the strategic tier inherits the operational boundary map: where the
  coordination moat is deep, design value is locked behind coordination capability; where it
  is shallow, napkin deployment harvests it. Buying the second depot without the C2 to use it
  is usually worse than deploying one depot well.
- **Thesis fit:** Obj-4's "facility location ... holistic, simultaneous evaluation" sentence
  in its honest oracle-priced form (metamodel included); the D2/B1 tier-coupling arc completed
  at the placement tier; two committed figure pairs (frontier + overlap value, Kaliningrad +
  held-out Gdansk); fleet composition recorded as the one-line future rung.

## 29. gen31: the gen27 conversion lands — the aerial trained positive, confirmed blind  (2026-07-19/20 · branch `gen28-aerial`, worktree `../sacred-aerial` · `experiments/gen31_aerial_dyn.md`; trainer `scripts/train_aerial_dyn31.py`; corridor hunt `scratch/gen31_corridor_hunt.py`)

- **Goal (prospective):** Kilian's 2026-07-19 mandate: analyse what made gen27 work and
  recreate it in the air (iterate-until-positive, unlimited budget, full enemy-design
  freedom, M4 only). The literal transplant (v4.0-dyn) had already failed with the
  corridor-collapse mechanism; gen31 redesigned the ENEMY and the INFORMATION CHANNEL until
  gen27's preconditions held, oracle-verified before any training.
- **Headline results:** (i) **Phase 0 (corridor hunt, 48 oracle-exact cells):** the
  anticipatory mixed doctrine (70% punish the recent pattern, 30% pre-aim at the obvious
  escape route, tau 0.10) opens a corridor no rule class closes: static cap 3.8-4.9x the
  exact optimum, payoff-blind avoidance rules 2.7x+ (on structured layouts the doctrine
  punishes naive avoidance BELOW static play), fitted doctrine rules 1.45-1.8x; honest
  surprise: the corridor existed even under the v4.0 doctrine at tau 0.10 - v4.0 rejected
  the temperature and its policy lacked the doctrine channel to reach it. (ii) **Attempt 1
  (3 seeds x 16k sorties):** passes iteration diagnostics 3/3 (dev-test beats-cap and
  beats-blind 3/3 at ~0.51x the cap; rw[doctrine] ~ -15 dominant; no drift).
  (iii) **CONFIRMATION (fresh seeds 10/11/12 + blinded control, six PRISTINE gated
  layouts never touched by any probe): every bar passes - 6/6 beats-cap on 3/3 seeds
  (18/18 cells), pooled 0.515x the static cap, 2.06x the exact dynamic optimum (STRONG),
  blinded control 1.21x the cap at 0/6 (causal), blind dynamic family beaten 17/18,
  worst-case-vs-committing premium 1.22x (gen27's was 1.57x). ONE attempt, no re-rolls,
  no bar movement.** Fitted doctrine rules stay ~1.4x ahead (disclosed, the gen27
  composed-rule pattern).
- **What we learned:** the v4.0 failure was an information-channel failure, not a game
  impossibility - give the policy the per-route threat-given-window column and the win is
  expressible AND trainable; a corridor hunt with representability + complete-family gates
  aims a training act so well that iteration was not needed; the untouched-gated-set +
  blind-confirmation protocol makes an iterate-until-done mandate scientifically safe.
- **Thesis fit:** the aerial branch gains its trained positive in gen27's exact claim
  shape (zero-shot dynamic hedging, causal control, complete family), retiring the "no
  aerial trained-positive" caveat; the boundary map gains a second domain where learning
  pays in the adaptive register; the regime table's dynamic row now has an aerial cell.

## 30. gen32: the aerial positive on the REAL Kaliningrad map, rendered as the operations map  (2026-07-20/21 · branch `gen28-aerial`, worktree `../sacred-aerial` · `experiments/gen32_theatre_dyn.md`; trainer `scripts/train_aerial_dyn32.py`; env `src/envs/aerial_theatre_env.py`; render `scratch/gen32_ops_map.html`)

- **Goal (prospective):** Kilian's 2026-07-20 mandate: reproduce the gen31 synthetic-lattice
  positive on the REAL Kaliningrad->Gvardeysk vec-theatre (the substrate behind the committed
  ops-map artefact), answering the examiner question "did the abstract grid do the work?";
  full enemy-design freedom, M4, iterate-until-done; deliverable = the operations-map render.
- **Headline results:** (i) **Phase 0 (corridor hunt on real terrain, oracle-only):** the literal
  gen31 doctrine (rep+flee, w=2) COLLAPSES on real fields with a small safe support (blind
  rotation attains the optimum); the FIX (enemy-design freedom, never map edits) = a third
  doctrine component (anti-repeat anticipation) + w=3 (the gen27 value that breaks small-support
  rotation). Pinned q=(0.6,0.2,0.3) tau=0.10 w=3: across 12 fields G1 (static cap) min 2.67,
  G2 (blind beatable) >=1.25 on 11/12, G3 (fitted) ~1.08. (ii) **Attempt 1 (3 seeds x 16k):**
  PASS 3/3 on dev-test (beats-cap + beats-blind 2/2, ~1.30x the exact optimum). (iii)
  **CONFIRMATION (fresh seeds 10/11/12 + blinded control, six PRISTINE gated fields): every bar
  passes - 6/6 beats-cap on 3/3 seeds (18/18 cells), pooled 0.451x the static cap, 1.30x the
  exact dynamic optimum (STRONG), blinded control 1.28x cap 0/6 (causal, recency+doctrine
  columns pinned 0), beats the blind dynamic family 15/18 (the un-beaten field = the
  pre-disclosed marginal), worst-case premium 1.36x. ONE attempt, no re-rolls.** The abstract
  lattice did NOT do the work: the gen31 positive reproduces on real OSM terrain and TIGHTER to
  the optimum (0.45x cap / 1.30x opt vs gen31's 0.52x / 1.74x), because the real threat field is
  a cleaner signal for the doctrine channel.
- **What we learned:** the real-map corridor has higher field variance than the synthetic pinch,
  so the doctrine needed the anti-repeat-anticipation component + w=3 to be robust; a corridor
  hunt with the completed baseline family aims a real-terrain training act precisely; the new
  theatre env adapter (`src/envs/aerial_theatre_env.py`) makes any vec-theatre SAC-trainable.
- **Artefact:** a dynamic operations-map render (`scratch/gen32_ops_map.html`) showing the
  pattern-of-life AD re-aiming and SACRED threading around it, live scoreboard (SACRED 0.096 vs
  naive 0.18-0.23 over 80 serials on real terrain). *(Status revised 2026-08-09, Kilian: a
  working render, not a deliverable; the project's visual centrepiece is the imperial-sacred
  Mission Control app.)*
- **Thesis fit:** the aerial positive now lives on the REAL Kaliningrad map (Obj-2's visual +
  the map-conditioned transfer the branch was built for), retiring the "synthetic lattice" caveat;
  the boundary map's adaptive register has a real-geography cell; the Obj-2 visual asset is the
  imperial-sacred Mission Control app (status revised 2026-08-09).

## 31. The dynamic-yardstick repair + the Phase-1 hardening pre-registrations (gen34/gen35/gen36)  (2026-07-23 · branch `gen08-interdiction` + worktree `../sacred-gen29` · probes `scratch/dyn_exact.py`, `scratch/gen35_{kdyn_probe,mmc_check}.py`, `scratch/dyn_yardstick_repair.py`, `scratch/gen34_family_probe.py`, `../sacred-gen29/scratch/gen36_label_probe.py`)

- **Kilian's direction (2026-07-23, deadlines set aside):** harden the current frontier without
  damaging banked claims, then complete the LLM experiments. Phase 1 = (1) adversary-family act,
  (2) dynamic high-K corner, (3) the gen29 reopening. All oracle-only this session; no training
  launched.
- **The repair (found by the gen35 design probe, verified two independent ways):** the undamped
  RVI behind `oracle_refs`' `history_opt` OSCILLATES on the deterministic-transition window MDP
  and is wrong on every cell tested (both directions; -57% worst). Exact truth = Karp minimum
  mean cycle = damped RVI (`scratch/dyn_exact.py`; 10/10 agreement to 5 decimals). The aerial
  branch had found and fixed the same defect (`dbf385d`, 2026-07-17: "rotation-beats-optimum
  test") - the fix never crossed back; gen31/32 aerial yardsticks are sound. Corrected-yardstick
  appendices landed in the gen19 and gen27 ledgers: gen19's STRONG "reaches the dynamic optimum"
  is retired (SACRED = 1.21x the exact optimum, which plain rotation ATTAINS on that m=4
  instance); gen27's pooled optimum-ratio restates 1.74x -> 1.97x. Every PRIMARY and causal
  control is UNAFFECTED (iid_eq is exact enumeration). Process lesson recorded: a solver defect
  fixed on one worktree stayed live on the sibling that shared the pattern.
- **gen34 pre-registered** (`experiments/gen34_hidden_adversary.md`): the enemy TYPE drawn
  hidden per episode from a five-doctrine family; the exact type-blind cap and per-type optima
  are computable, the inference gap is 1.36-2.04x (largest on the gen27 held-out class), the
  playbook-fitted Bayes-MAP row captures ~80% of it, and every specialist counter-doctrine blows
  up off-diagonal. PRIMARY = beat the blind cap zero-shot (impossible for any type-blind
  object), causal control = intel columns zeroed.
- **gen35 pre-registered** (`experiments/gen35_dyn_kboundary.md`): the corrected landscape shows
  rotation IS the exact optimum at every K on m=4 instances (scoping fact), while on 71-33
  (m=6) the naive-rule gap widens 1.31x -> 1.56x with K. K=2/3 cells on 71-33, gen19 trainer
  unmodified: the programme's first pre-registerable "beats every two-line rule" bar.
- **gen36 pre-registered** (`../sacred-gen29/experiments/gen36_multiod_rescue.md`): executes
  gen29's two pre-committed options - distillation control first (labels verified: anchors
  reproduce exactly, sparse 2-11-atom supports, ~1 s for all 26 cells), then the single
  permissible dense-credit re-aim (exact telescoping per-stream decomposition), B gated on A,
  the wall-location matrix pre-written (capacity vs dynamics).
- **State:** suite 167 green; all probes committed with JSONs regenerable from seeds; every
  launch awaits Kilian's explicit go.

## 32. Phase-1 verdicts in one day: the first beats-every-baseline cell, and the capacity wall sighted twice  (2026-07-23/24 · branches `gen08-interdiction` + `gen29-multiod` · ledgers `experiments/gen35_dyn_kboundary.md`, `../sacred-gen29/experiments/gen36_multiod_rescue.md`, `experiments/gen34_hidden_adversary.md`)

- **gen35 (dynamic K-boundary, 71-33 m=6): the milestone.** K=3 PRIMARY PASS 3/3 seeds +
  pooled (0.1406 < best two-line rule 0.1539, -8.6%) - the programme's FIRST cell where "the
  trained policy beats every naive baseline" was the pre-registered bar. K=2 = a tie AT the
  rule (pooled +0.5%): the boundary sits between K=2 and K=3. STRONG missed (1.38x the exact
  optimum; ~26% of the rule-to-optimum slack collected). No-window control clean (0.2328, rw
  pinned 0 = causal). Sharpener: tabular window-Q at MATCHED budget fails both cells
  (0.1083/0.1759) - unlike gen26's static register, this value is not no-net collectable.
  Worst-case committing premiums 1.72x/1.51x disclosed.
- **gen36 (the gen29 reopening): the wall is CAPACITY.** Distillation with EXACT coordinated
  labels fails both tiers 0/3 seeds (pooled 1.80-2.49 vs bar 1.44); CE plateaus far above the
  label-entropy floor = the class cannot fit the correlated targets even in-sample. Step B
  (the one permissible re-aim) correctly NOT launched per the pre-registered gate. gen29
  closes with its mechanism separated: conditioning capacity, not training dynamics.
- **gen34 (hidden adversary family): FAIL per the pre-written branch, with the same wall.**
  Pooled held-out 1.373x the type-blind cap, 0/18 crossings - no type inference. Nuances: the
  intel channel is causally useful short of inference (control interim 2.135 vs sighted ~1.5;
  control stopped 5520/12000 on Kilian's instruction); the policy lands ~7% BELOW the composed
  anti-repeat rule. The oracle landscape (inference worth 1.39-2.04x held-out, brittleness
  cross-table, ~80% playbook row) is the act's banked contribution. Two independent sightings
  of the head's conditioning-capacity limit in one day (gen36: prefix identity; gen34: member
  identity) = the sharpest architecture-directed future-work sentence the thesis has.
- **State:** all three ledgers folded and committed; Kilian 2026-07-24: skip the control
  remainder, proceed to Phase 2 = gen33 completion (anchor repair, then the metric-2 full
  re-run).

## 33. The LLM arc resolves: worse-than-random at numbers, decisive at language - and the first LLM-assists-SACRED positive  (2026-07-24/25 · branches `gen08-interdiction` + `gen29-multiod` · ledgers `experiments/gen38_llm_enemy_id.md`, `../sacred-gen29/experiments/gen37_reasoning_curation.md`)

- **Kilian's Phase-2 pivot (24 Jul):** the staged gen33 metric-2 curriculum act was dropped for
  "SACRED enhanced via LLM reasoning". Two acts, both pre-registered, both decided cheaply at
  the mechanism level before any training spend.
- **gen37 (route curation): REJECTED at the ceiling.** The LLM shortlists route-triples WORSE
  than random at every prune size (LP-over-shortlist held pooled: llm 1.66/1.57/1.21 at
  M=10/15/50 vs random 1.42/1.47/1.10; oracle-exact, ~52 LLM calls, zero training). The
  trained half (does restriction per se unlock SACRED) was aborted on Kilian's instruction
  before any run completed - logged as future work, nothing claimed.
- **gen38 (enemy identification): the strand's POSITIVE, all three steps.** V1: the LLM reads
  behavioural intelligence narratives and classifies the five gen34 doctrines at 100% (60/60),
  crossing the gen34 type-blind wall 6/6 with the exact counters (0.063 = omni cap; keyword
  control 80% accurate yet WORSE THAN BLINDNESS operationally, 0.194 vs blind 0.114, crosses
  0/6). Robustness: the reasoning-vs-lookup gap WIDENS under contradictory intel (LLM 0.80 and
  6/6 vs keyword 0.45); honest fragility at terse+contradictory (0.40, 0/6) = the
  confidence-hedge regime. V2: a TRAINED type-conditioned SACRED (per-route type-threat
  column; the naive one-hot was inert - disclosed repair) crosses the wall 3/3 seeds at
  pooled 0.664x the blind cap (gen34's blind generalist: 1.373x), capturing ~75% of the exact
  inference gap, with the LLM-supplied type indistinguishable from truth (delta <= 0.003).
- **The arc's honest shape (the thesis sentence):** across B2 -> gen33 -> gen37 -> gen38, the
  LLM is measurably USELESS-TO-HARMFUL at the quantitative registers of this pipeline
  (numeric mixtures, terrain-grounded composition, combinatorial curation) and decisively
  VALUABLE at the language->decision register (doctrine identification from prose), where it
  supplies exactly the ingredient the RL side provably lacked (gen34/gen36's
  conditioning-capacity wall). Where-LLMs-help is now a measured map, not an opinion.
- **State:** suites green both worktrees; all ledgers folded and committed; the one open
  computational thread = the cheap gen37 restriction follow-up (future work).

## 34. gen39: the concealment act, and the LLM arc's most complete answer  (2026-07-25 -> 28 · branch `gen28-aerial`, worktree `../sacred-aerial` · `experiments/gen39_concealment.md`; handover `HANDOVER_AERIAL_28-07-26.md`; trainer `scripts/train_gen39_conceal.py`)

- **Goal (prospective):** give concealment a MEANING (a defender that must locate the enemy), then
  ask the two questions the LLM arc had left open: can a language model COMPOSE a better enemy
  force than a hand-tuned doctrine, and does training SACRED against LLM-composed enemies produce
  a better defender?
- **Headline results.** (i) **The mechanic works, with an internal control:** sight is worth
  1.26-1.37x to the defender against a force on revealing ground and **exactly 1.00x** against a
  concealed one, same map, same rules. Operating point pinned on narva at the untouched weapons
  table (kgd/ukraine/fulda held out); the four-map screen (12,960 cells) passes both gates on 86%
  of real cells. (ii) **Step 2 is an LLM POSITIVE with its binding control passing:** llama's
  forces beat the gen32 doctrine against the best simple defender (0.0747 vs 0.0603, every
  clause; qwen partial 0.0613), and relabelling forest/open in the brief changes both models'
  compositions and collapses their forces 10-13x - **the first licensed terrain-reasoning claim
  of the LLM arc**, where gen33's equivalent had failed. (iii) **Step 3 FAILED 0/3 seeds** (the
  llm curriculum beat random composition by 29% but lost to the tuned doctrine). (iv) **Phase 1
  diagnosed it, oracle-only: curriculum value tracks the enemy's IRREDUCIBLE THREAT** - its damage
  against a defender that already knows where it is - and LLM forces were concealment gambits
  (0.0007 vs the doctrine's 0.0215). (v) **Step 5 fixed it:** every curriculum rebuilt by a
  matched 16-evaluation search, **PRIMARY PASS 3/3 seeds (llm16 0.1288 vs tuned control 0.1677,
  23% better, paired -0.0389 +/- 0.0031), and it TRANSFERS zero-shot to all three unseen theatres
  (3/3 seeds each; llm16 also leads local16 on 9/9 map-seed pairs).**
- **What we learned about where LLMs help (Phases 1c-1f, all oracle-only, ~1 h of calls).**
  Briefing is not the constraint (a robustness clause changed the model's behaviour without
  changing its outcome); **grounding was** (its declared coverage matched reality 12-40% of the
  time, rising to 91% once given a readable slot catalogue); what remains is **combinatorial
  search**; and in a 1,313,400-force space at equal simulation budget the LLM **leads every method
  at 8-16 evaluations and is overtaken by hill-climbing by 96** - a bounded SAMPLE-EFFICIENCY
  claim, which is the honest one and the operationally relevant regime.
- **Boundaries that survive everything (binding):** **no arm beats the best simple OBSERVING RULE
  on any cell on any of the four maps**, so gen39 licenses no "trained policy beats the rules"
  sentence; llm16 and local16 are statistically indistinguishable in-distribution (paired -0.0066
  +/- 0.0265); everything is per-model and the models REVERSE between tasks (llama leads at
  composition, qwen at the grounded slot task).
- **Method notes worth inheriting.** Nine result blocks in the ledger are superseded and left
  visible with their reasons. Four corrections were made mid-arc and disclosed rather than
  quietly fixed: an eval defect (151k network calls per checkpoint, now closed-form and tested),
  duplicate per-transition graphs in replay (~1 GB/run, the true cause of every "system time"
  crawl after two mis-diagnoses), a single-field test that produced a PASS which was a
  field-selection artefact, and a **pre-declared prediction that FAILED** (kgd was pre-declared a
  negative cell for the llm arm; it turned out to be its strongest map, recorded as made-and-wrong).
- **Thesis fit:** the aerial act gains a concealment/information-channel mechanic with an internal
  control (Obj-2/Obj-1); the LLM arc gains its clearest positive (composition, terrain-grounded)
  AND its clearest bounded claim (sample efficiency), completing the where-LLMs-help map begun in
  B2/gen33/gen37/gen38; Obj-5's comparative discipline gains the matched-budget-control pattern.

## 35. gen40 + gen41: the dynamic-register structure laws, the fairness tiers, and the doctrine-head transfer positive  (2026-08-04 -> 08 · branch `gen08-interdiction` · `experiments/gen40_dyn_sensitivity.md`, `experiments/gen41_deepwindow_zst.md`)

- **Goal (prospective):** answer Kilian's Act-3 fairness gripe (the composed rule wins 4.3 only
  because it is handed both insights; the exact optimum needs the enemy's full model) with
  measurement, then design and run an experiment where SACRED beats its competition under a
  like-for-like information discipline.
- **Headline results:** (i) **gen40 (oracle-only, ~400 exact cells):** the dynamic register's
  structure laws - rotation is exactly optimal at w = m-1 (K- and R-invariant); rules fail at
  essentially every other w >= m, deepest at w a multiple of m; rule failure grows with the
  coverage fraction K/m to the exact wall (K ~ 4-5); menu padding widens rule failure and raises
  a ceiling only menu-wide play can reach; NO computable adversary extension preserves the game
  past the wall (the pre-committed negative fired: the wall binds the GAME). Corridor counts
  m >= 7 exist only in Istanbul; Kyiv's arterial max degree is 6. (ii) **gen41 Act 1 (w=6=2m,
  K=2): FAIL 0/3 as pre-registered** - every practical object sits 0.93-1.36x the cap while the
  exact optimum sits at 0.58x; the window channel bought only 0.03-0.25 of ratio (vs gen27's
  0.8): a CHANNEL-CONTENT failure (aggregate frequency cannot express order-dependent
  window-steering). The w-axis band map completed: learning pays at w ~ m, nothing tested
  collects by w = 2m. (iii) **The gate discipline built and validated:** the three-tier fairness
  ladder (Tier 0 map-only, Tier 1 mechanism-told, Tier 2 outcome-earning); Gate 1 = the
  representability CERTIFICATE (exact witnesses inside the policy class: at w=3/K=2 the
  count-conditioned optimum ATTAINS the true optimum and a linear-feature witness reaches 0.478x
  cap vs the composed rule's 0.656, weights near-universally (0, 0, -40)); Gate 2 = the
  single-instance rung. (iv) **Act 2 (full net, w=3, K=2): PRIMARY FAIL 0/3** (pooled 0.943;
  causal gap restored at 0.8 of ratio; the failure isolated to instance-tuned ENCODER variants
  - the certified universal weights transfer perfectly, SAC did not converge to them).
  (v) **Act 3 (the doctrine head - identical config, one flag masking the encoder out of the
  actor): PRIMARY PASS 2/3** (pooled 0.783 +/- 0.032; beats the cap 6/6 on every seed; beats
  every Tier-0 rule 3/3 and every Tier-2 adaptive learner on 2/3 seeds, seed 0 missing the
  self-tuned-composed gate by 0.0005, judged FAIL as registered); STRONG FAIL (the told-rules
  tier at 0.656 stays ahead); control 1.161 at 0/6 (causal). The 0.943 -> 0.783 ablation
  confirms the Act-2 autopsy BY INTERVENTION. Verification appendix: the trained weights under
  argmax score 0.6625 (= the composed rule), so the residual gap decomposes measurably into
  entropy stochasticity (~0.12) and weight calibration (~0.18), both training-efficiency
  artefacts inside a certified-adequate class.
- **What we learned:** (i) screens measure OPPORTUNITY and representability gates measure
  EXPRESSIBILITY, and both belong BEFORE training (the w=6 act would have been killed by a
  20-line certificate); (ii) the encoder pathway, not the game, was the ZST calibration
  bottleneck at K=2 - three parameters transfer better than the GNN; (iii) fairness tiering by
  information consumed resolves the told-rule dispute without dropping baselines; (iv) ops
  lessons at cost: SAC updates batch 32 FULL-CITY graphs (+8.45 GB/arm at Kyiv size, the OOM),
  replay observations must share per-instance payloads (fixed, +4 contract tests, suite 171),
  zsh does not word-split inline flag bundles, and launches are verified at first-print level.
- **What it means for the thesis:** Act 3's fairness dispute ends in a measured settlement - the
  like-for-like claim SACRED now owns (beats every rule that is not TOLD the enemy's mechanism,
  zero-shot, causally attributed) plus the honest concession that told rules remain ahead; the
  w-band map and the two boundaries (channel content at w=2m; encoder calibration at transfer)
  bound the flagship's regime from both sides; gen27 remains the flagship, gen41 the mechanism
  act beside it.
- **Thesis fit:** Obj-1 (the band map + wall law), Obj-3 (the doctrine head as the
  entropy-as-mixed-strategy mechanism laid bare: the policy IS three weights), Obj-5 (the
  three-tier ladder as the comparative discipline's final form), ZST (the rescued like-for-like
  transfer claim). Ledger pointers only; numbers live in gen40/gen41.

## 36. gen43: the consolidated Act-2 instrument - one instance, both registers, one K-axis  (2026-08-08 · branch `gen08-interdiction` · `experiments/gen43_unified_kboundary.md`)

- **Goal (prospective):** Kilian's consolidation direction: the thesis's Act 2 stitched two
  experiments on two instances with mismatched K columns (gen26 static, gen35 dynamic).
  Rebuild it as ONE instrument: 71-33 (m=6, R=11, kx=8), both adversary registers, one
  K-ladder run to each register's own measured wall, reusing banked cells where licensed.
- **Headline results:** (i) **Free probes first:** exact LP feasible to K=4 (v* 0.1276 to
  0.5106; the inv-vuln stack EXACTLY optimal at K=1); greedy-vs-exact stack fidelity 0.0000
  at K<=4; the static right edge measured (mixing's value over the best committed route dies
  between K=8 and K=9, det 0.8325 optimal from K=9; K=9/10 oracle-only); dynamic K=4 cheap,
  K=5 nonexistent on this menu (gen40 wall). (ii) **The reuse licence:** banked static K=5/6
  and dynamic K=2/3 reused verbatim - code identity proven by inspection (two inert gated
  diffs), oracle side byte-exact, and bit-replay measured to NOT exist (two identical
  invocations differ; macOS updated between the banked batches): pinned-SHA + n=3 spread is
  the standard, and the new K=4/K=7 cells bracket the seam smoothly (clause satisfied).
  (iii) **The batch (8 new cells, ~6.4 h, Kilian-launched):** static SACRED tracks the naive
  frontier from behind at every K (0.160/0.328/0.463/0.605 at K=1-4 vs exactly/near-optimal
  stacks; ties at K=7/8 as the frontier collapses onto det). **Dynamic K=4: PRIMARY PASS 3/3
  AND pooled (0.1820 +/- 0.0036 vs best rule 0.2152, -15.4%)** - the beats-every-rule region
  runs from K=3 to the game's wall with WIDENING margin (slack collected 26% -> 43%); STRONG
  fails (1.31x the exact optimum); matched-budget tabular window-Q fails at both new cells;
  worst-case committing premiums 1.60x/1.35x disclosed. (iv) One latent defect surfaced and
  repaired post-results: the exact-path JSON writer crashed after training completed (logs +
  checkpoints carry those cells); one-line fix, suite 171 green.
- **What we learned:** (i) the two registers now separate on a single instrument - the
  committed game extinguishes randomisation at high K while the observant game rewards it
  ever more, to the wall; (ii) the reversal claim ("the ordering flips when the adversary
  watches") no longer spans instances; (iii) bit-replay of multi-threaded training never
  existed on this stack - the honest reproducibility unit is pinned-SHA code identity plus
  seed spread, now ledgered as a measured fact; (iv) a crash after the last print costs
  nothing IF per-eval artefacts and checkpoints are written incrementally.
- **Thesis progression:** Act 2 rebuilds from one ledger: two mirrored tables and one figure
  (static curve ending at the death of mixing, dynamic curve ending at the computability
  wall), replacing the stitched gen26/gen35 presentation; 35-159's exact K=3 crossing
  survives as a one-sentence second-instance replication remark.
- **What it means for the thesis:** the Act-2 argument sharpens to its final form - where
  the enemy commits, buy the two-line stack (exactly optimal at K=1, never beaten by
  training on this instance); where the enemy watches, learning collects value no rule or
  tabular learner reaches, growing with the budget to the exact wall.
- **Thesis fit:** Obj-5 (the unified comparative instrument), Obj-1 (both walls as measured
  laws), Obj-3 (function approximation load-bearing in the dynamic register). Numbers live
  in `experiments/gen43_unified_kboundary.md` only.
- **Extension (2026-08-09, same ledger):** Kilian asked the dynamic arm to reach K=8; the
  probe found the "K=4 wall" was gen40's sweep work-guard, not the game's end (correction
  disclosed in the ledger), and the EXACT game extends to K=6 for minutes of oracle compute
  while K=7/8 are measured out on memory and cost (2.8/12.4 GB loss matrices), with
  heuristic-proxy adversaries barred by the gen40 tier-E negative. The two new exact cells
  ran overnight: **K=5 PRIMARY PASS 3/3 (pooled 0.2175 +/- 0.0041 vs rule 0.2743, -20.7%)
  and K=6 PRIMARY PASS 3/3 (0.2638 +/- 0.0020 vs rule 0.3295, -19.9%)**, both STRONG-fail
  at 1.24x the exact optimum (the closest any dynamic cell has come). The
  beats-every-rule region now spans K=3 through K=6, the ENTIRE computable range past the
  K=2 tie, with slack collected rising 26% -> 43% -> 57% and plateauing ~56%, while the
  total value of history declines (iid/opt 2.25 -> 1.90), the game's saturation and the
  policy's mastery of what remains, measured together.

## 37. The model-capability measurement chain: the gen39 brief repair, the gen42 ladder, the gen43 exam, and the gen44 budget sweep  (2026-08-06 -> 09 · branch `gen28-aerial`, worktree `../sacred-aerial` · ledgers `gen39_concealment.md` (repair section), `gen42_capability_ladder.md`, `gen43_exam.md`, `gen44_budget_sweep.md`; all oracle/eval-only, no training)

- **Goal (prospective):** put the LLM arc's per-model claims on measured ground. Repair the
  gen39 diagnostic chain's briefing defect, then ask whether general model capability predicts
  performance on the pipeline's LLM registers (a capability ladder within one family, gen42),
  rebuild the underpowered instrument for resolving power (the forty-question exam, the aerial
  gen43), and put error bars on curriculum authoring at every search budget (gen44).
- **Headline results:** (i) **The v1-brief defect repaired** (1c/1d/1e had briefed the v1
  physics table while every scorer used v2; 167 calls affected, everything else clean by
  construction). Re-run at pinned bars: "briefing is not the constraint" softened (a truthful
  brief roughly doubles the median yet every arm stays 5-10x short of the bar), grounding ~12%
  and UNMOVED by truthful physics, the 1e llama-vs-qwen slot reversal RETIRED (a defect
  asymmetry; corrected values near parity), and "what remains is combinatorial search"
  re-established on corrected artefacts, the only citable ones for that chain. (ii) **gen42**
  (six rungs, Qwen3.5 2B/4B/9B/27B + Qwen3.6-27B thinking off/on): endpoints separate (4B
  below 27B at the search-bound slot register), every finer contrast drowns in n=8 sampling
  noise; its two amendments (B-COMP underpowered at n=16; only the staircase's endpoints
  survive bootstrap) are binding. (iii) **The gen43 exam** (40 items, seven configurations on
  one paper, zero format failures): size helps and CUMULATIVELY (2B->27B +0.300 share of
  ceiling, 4B->27B +0.134, CIs excluding zero; no single step separates); generation and
  thinking stay unresolved with tight CIs; thinking changes the ANSWER on 23/40 items while
  moving the SCORE by about as much as re-rolling the sampling seed (the amendment's
  noise-floor rows, seed range 0.0219 vs effect 0.0295); llama-3.3-70b sits measurably BELOW
  both 2026 27Bs (overturning a gen42 clump); the 2B's gen42 format problem was
  harness-scoped, not model-scoped. (iv) **gen44** (nine repeated searches per configuration):
  2B/4B SEPARATE from 27B as curriculum authors at usable budgets, correcting gen42's n=1
  flat-band reading; every LLM author beats hill-climbing at EVERY budget at the CURRICULUM
  level (the banked defender-level tie stands; the two levels had been elided and are now
  distinct claims); llama vs crown-thinking indistinguishable at budget 16, which EXCLUDES
  curriculum strength as the step-5c transfer carrier.
- **What we learned:** power is a design property (per-item pairing at n=40 resolved what n=8
  medians could not); any small effect must beat the instrument's own noise floor (seed
  re-rolls); an n=0 LLM result is a transport question before it is a capability reading; and
  single-draw search curves mislead (three gen42 readings revised by the exam and the sweep).
- **Thesis fit:** Obj-3's LLM strand gains its capability axis (scale cumulative, generation
  null, deliberation answer-changing but not score-moving, all per-model per-register); the
  where-LLMs-help map's core prediction, that the search-bound register stays far below its
  ceiling at every scale, survives the whole family.

## 38. gen39 steps 5c-5e: the author-level transfer effect and the diversity carrier  (2026-08-08 -> 10 · branch `gen28-aerial` · `experiments/gen39_concealment.md` steps 5c/5d/5e; fresh-set harness `scratch/gen39_zeroshot2.py`)

- **Goal (prospective):** Kilian's additional step-5 row: does a curriculum authored by the
  best qwen with thinking ON train a better defender than llama's llm16 at the identical
  matched budget? The gen42 three-link argument predicted NULL; running the arm made that
  prediction judgeable.
- **Headline results:** (i) **The null prediction FAILED** (recorded made-and-wrong):
  qwenthink16 PASSES the step-5 primary 3/3 and its defender transfers better on 9/9 fresh
  map-seed pairs against every arm, at curriculum strength matched to <1%; in-distribution
  the advantage is suggestive only (paired p ~0.2 at n=3). (ii) **The comparability repair:**
  the banked zero-shot test sets were unrecoverable (scores saved, laydowns not; disclosed),
  so fresh laydown-saved sets were built once; the banked zero-shot pattern REPLICATES on
  them (llm16 over local16 8/9), and the pre-declared kgd negative prediction fired wrong
  again (kgd is llm16's strongest map). (iii) **5d:** a second qwenthink authoring roll
  matches strength (0.0387 vs 0.0390) and diversity (Jaccard 0.187 vs 0.153) while sharing
  ZERO of its 16 top laydowns: an author has reproducible STYLE without reproducible CHOICES,
  the shape a diversity mechanism needs and a content mechanism forbids. (iv) **5e, the
  decisive grid** (2 authors x 3 rolls x 3 seeds): PRIMARY PASS at the locked bar, complete
  rank separation of the author transfer means (qwen 0.2129 +/- 0.0039 vs llama 0.2407 +/-
  0.0052, exact permutation p 0.05); the mediation row fires (Jaccard-vs-transfer rho 0.83
  over all six rolls, within-author consistent); strength matched ~3% grid-wide stays
  excluded (gen44); the narva in-distribution column rank-separates too.
- **What we learned:** the transfer effect is a property of the AUTHOR, not of an authoring
  roll; curriculum DIVERSITY is the supported (not proven) carrier; defender-level effects
  need not follow curriculum-level measures; the step-5c +/-0.0004 seed-spread anomaly did
  not recur (a one-off).
- **What it means for the thesis:** the LLM arc's strongest positive shape: at matched search
  budget and matched curriculum strength, the thinking-mode reasoner's curricula train
  defenders that transfer better to unseen theatres, an author-level finding, per-model as
  always. The designed diversity-manipulation test and the 5d replication training are
  recorded future work, not run.
- **Thesis fit:** Obj-3 (the LLM-assists-SACRED line upgraded from a candidate to an
  author-level finding), Obj-5 (matched-budget and rank-separation discipline at its
  cleanest).

## 39. gen45: the unified corridor game - Acts 4 and 5 on one substrate  (2026-08-09/10 · branch `gen28-aerial` · `experiments/gen45_unified_corridor.md`)

- **Goal (prospective):** Kilian's consolidation direction: rebuild the gen32 real-corridor
  positive on the exact gen39 substrate so the thesis's Acts 4 and 5 share ONE game (one
  terrain table, one quota-sampled emplacement set, one multiplier field, one enemy model
  whose only dial is how far a team may relocate between serials; the gen32 "searchlight" is
  the gen39 machinery's flat full-map-relocation limit, regression-anchored at 3.9e-12).
- **Headline results:** Phase-0 hunt PASSED at the preferred pin (w=2, DOC32, tau 0.10; G1
  min 3.71 vs bar 2.0, G2 12/12; the corridor DEEPER than gen32's original hunt); the attempt
  wave passed its gate 3/3; the confirmation wave on the pristine gated set passed every bar:
  **PRIMARY at the bar's ceiling (beats the static cap 18/18 seed-field cells, pooled
  0.351x), STRONG (1.46x the exact dynamic optimum vs the 2.5x bar), CAUSAL (blinded control
  1.242x at 0/6 with its information weights pinned at 0.000000)**; the ENTIRE payoff-blind
  dynamic rule family beaten 18/18; fitted doctrine-informed rules ~1.2x ahead, disclosed as
  the standing pattern; worst-case committing premium 1.52x pooled; every digit verified by a
  second instance.
- **What we learned:** the recency channel is largely INERT on this game (one seed sweeps 6/6
  with a POSITIVE recency weight); the doctrine/threat column (~-20 on every seed and wave)
  carries the result, sharpening the Act-4 mechanism sentence; drift is the smallest of any
  act; the overnight-chain ops pattern (bash-launched nice-0 waves, a gate-guarded
  confirmation that alone may spend the pristine set) is worth inheriting.
- **What it means for the thesis:** Act 4's real-corridor claims rebuild from gen45; gen32 is
  SUPERSEDED and its numbers may never sit in a table beside gen45's (prose may say the claim
  SHAPE reproduces, each number attributed to its own game). The licensed sentence stays
  "beats every static object and every payoff-blind dynamic rule, discovered unaided".
- **Thesis fit:** Obj-1/Obj-2 (one game and one enemy model across both aerial acts), Obj-5
  (the gated-set single-evaluation ceremony), ZST unaffected (field-level zero-shot is the
  act's evaluation axis by construction).
