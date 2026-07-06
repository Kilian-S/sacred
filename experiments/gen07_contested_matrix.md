# Generation: gen07_contested_matrix (Act IV: does adversarial co-training buy unexploitability?)

- **status: CLOSED 2026-07-06: the matrix never ran; the pre-launch probes + BR gate produced the
  decisive finding that MOTIVATED THE INTERDICTION REDESIGN (`REDESIGN_INTERDICTION.md`).** This
  ledger is the record of the exploitability-on-the-contested-destination-arena attempt. Outcome:
  the arena-scoping probes (capacity, stress) showed the unpredictability lever is thin-to-negative
  against the *reactive* congestion attacker, and the CORRECTED BR gate (all five fixes applied,
  §B9.iv below) showed a best-response attacker plateaus at **0.35× random** with entropy pinned
  and near-zero Q-spread. **Root cause (the flat attack landscape): congestion is
  observable/reroutable/reversible, so every route-reach block causes similar cascading damage,
  random is already near-optimal, and no learned adversary has an edge.** This is a structural
  property of the congestion adversary, not a fixable optimisation issue, and it is exactly why the
  project pivots to interdiction (hidden/irreversible/pre-committed → a security game where the
  mixed strategy provably wins). The five learnability fixes (branch `gen07-contested`, suite 109
  green) and the whole evaluation discipline carry forward to gen08. **Original pre-registration
  preserved below for the record; the matrix decision metrics were never triggered.**
- **strategy/rationale:** `DIRECTION.md` (exploitability register, five fixes, contested skin).
- **branch policy:** all gen07 code on branch `gen07-contested`; `main` carries no `src/`
  changes (ROADMAP Phase B separation policy).

## Question (fixed before looking)

**Does adversarial co-training (curriculum ATLA against an adversary population) produce a
dispatcher that is HARDER TO EXPLOIT by attacks tailored to it than an identically-trained
non-adversarial dispatcher, at bounded clean cost?** Exploitability = the damage of the
strongest attack prepared against a frozen policy (DIRECTION.md §2). Motivating in-house
evidence (ledgered): gen05 BR +1667 vs deterministic greedy; gen06 A3.2 (aimed-attack
robustness declines as clean training specialises the policy); gen06 A3.3 (the deficit is
policy, not temperature).

## Arena (contested resupply; TO-FINALISE by the B9 probes, recommendation recorded)

Recommended: **dynassign dynamics + route-reach attacker surface**: Poisson assignment (the rung
where protagonists demonstrably reach competence: gen03/06 band) with `antag_reach="route"` and
full-block antagonist so that BOTH scripted and learned attacks aim along committed routes (the
surface where learned BRs demonstrably work, gen05). Budget TO-FINALISE by the recoverability
probe (target: fitted-scripted attack costs greedy +30-60% with attacked delivery in a trainable
band, NOT the gen06 collapse regime). Escalation rule (pre-registered): if the B9.v coping-channel
probe FAILS on this arena (see Gates), the arena moves to the fixed hybrid with the learnability
fixes, and this ledger is re-issued BEFORE any training.

**Arena-scoping probes (2026-07-06; capacity + load; `scratch/capacity_probe.py`,
`scratch/stress_sweep.py`; greedy rollouts only, NO training; pin TO-FINALISE slots, never
outcomes):**
- **Capacity stays 1 (probe-refuted raising it).** Raising truck capacity 1→3→5 de-stresses the
  system (clean W 5908→1133→1036; attacker bite D 4768→1783→697) and DESTROYS the exploitability
  lever (lever 217→−39→−88). The lever is a stress phenomenon, not a capacity one; capacity slack
  absorbs the disruption predictability would otherwise cost. So capacity-1 (the original
  placeholder) is the regime where the thesis is testable. This also settles B8: rolling-ALNS
  degenerates at capacity 1 → use greedy (already reactive/rolling) as the Obj-5 reactive
  reference; ALNS is future work (it earns its keep only in the higher-capacity VRP we've ruled
  out).
- **Load (λ) is the difficulty knob → TO-FINALISE by the powered stress sweep.** The 12-instance
  probe hinted the lever roughly doubles from λ=0.06 (217, ratio 1.36, delivery 0.75) to λ=0.08
  (491, ratio 2.5, delivery 0.56) before collapse at 0.10-0.12. The powered sweep
  (`scratch/stress_sweep.py`, 40 instances, per-instance lever 95% CI) pins the operating λ =
  the largest CI-positive lever whose delivery stays trainable (well above the gen06 collapse
  band). Honest caveat: even at the sweet spot the greedy-measured lever is ~10% of D (a LOWER
  bound; crude ε-random assignment), so the destination-mode lever is real but thin; the hybrid
  routing arena remains the recorded escalation if it proves too thin for a learned policy to
  capture.

## Arms (identical env/reward/nets/hparams; only training-time exposure differs)

| arm | training | seeds |
|---|---|---|
| `greedy` | none (deterministic reactive reference) | - |
| `vanilla` | no adversary | 3 |
| `dr` | random-attack exposure under the SAME curriculum schedule as `sacred` | 3 |
| `sacred` | curriculum ATLA vs the adversary population (scripted seeds + successively trained BRs; mixture weights logged) | 3 |
| `vanilla@tau` | EVAL-TIME row: the selected vanilla checkpoints sampled at raised temperature matched to sacred's realised policy entropy (the "just add noise" control; no training; machinery = `scratch/gen06_matched_temperature.py` generalised) | - |
| `erb_*` (OPTIONAL, supervisor item e) | ERB-seeded variants of vanilla/sacred for the Obj-3 time-to-competence ablation | 2-3 |

Common training config TO-FINALISE at launch (episodes, switch cadence, γ per B5, entropy
targets per B2, curriculum schedule per B3, counterfactual twin per B1). Gradient budgets
matched across arms.

## Attack portfolio and the exploitability estimator (PRE-REGISTERED)

Per arm `a`, the tailored portfolio `P(a)`:

1. **Fitted scripted attacks**: `targeted` and `pathrand` families, each with per-victim
   parameter fitting on VALIDATION instances only (fitting grid TO-FINALISE at launch, fixed
   before any test-instance contact);
2. **Learned best response** `br_a`: one antagonist (learnability package: factored head,
   route-reach mask, motion features, counterfactual reward, lowered entropy target) trained
   against frozen `a` for an equal budget (300 ep) per arm;
3. `random` (undirected floor, kept inside the max as a sanity lower bound).

W = mean total_wait over **30 paired test instances** (seed base 10_000_019; validation
20_000_019; protagonists stochastic with the standard per-episode crc32 seeding; BR attackers
deterministic). D(a, atk) = W(atk) − W(none), paired per instance.

> **Exploitability:** `Expl(a) = max over atk in P(a) of mean D(a, atk)`.
> **PRIMARY:** `dExpl = Expl(vanilla) − Expl(sacred)`, per seed pairing and pooled.
> **Success = pooled dExpl > 0 with a paired-bootstrap 95% CI excluding 0 (resampling
> instances, recomputing the per-arm max within each resample), AND ≥ 2/3 seed pairings
> individually positive, AND the competence gate + clean-premium bound below hold.**

**Secondary headline (reported, not gating the primary):** `Expl(sacred) < Expl(greedy)` with
the same portfolio construction fitted to greedy: the deterministic classical dispatcher's
exploitability is the operational comparison the contested framing cares about.

**Gates and bounds on interpretation (pre-registered):**
- **Competence gate** (gen05 lesson): every learned arm's W(none) within +15% of greedy's,
  else that arm's rows are flagged competence-compromised.
- **Clean-premium bound**: W(sacred, none) − W(vanilla, none) ≤ +10%; if exceeded, the primary
  is reported but the claim downgrades to "unexploitability bought at clean cost" (frontier
  branch), not headline success.
- **Statistical reporting rule** (A3.4 standard): pooled instance-level CI AND per-pairing sign
  consistency AND the 3-pairing t sensitivity, always reported together.

**Secondaries (reported):** the gen06-style held-out rows (D under random/pathrand/targeted and
dD between arms: continuity with Act II; Tier 2); the `dr` and `vanilla@tau` rows under the
full portfolio (causal isolation: exposure-at-all vs aimed exposure vs raw noise); per-arm
realised policy entropy at the selected checkpoints; budget-axis sweep curves (eval-only) for
Expl and D; Obj-4/ZST eval-only extensions per supervisor decisions (separate sections appended
if funded).

**Checkpoint selection:** per arm, min mean attacked W on VALIDATION instances under a
validation attacker that is NOT in any test portfolio (TO-FINALISE at launch: a third scripted
variant reserved for selection only), never on test attacks or test instances.

## Pre-launch gates (B9; all cheap, all pass/fail recorded here before launch)

- **B9.i** suite green (≥83 + new tests), on branch `gen07-contested`.
- **B9.ii** timing probe: s/ep both phases + twin-rollout overhead; publish the compute
  envelope; no launch without it.
- **B9.iii** competence/recoverability probe: greedy band reachable; fitted-scripted attack in
  the +30-60% band on greedy; attacked delivery within the trainable band (0.4-0.8) at the
  curriculum's opening strength.
- **B9.iv BR gate** (victim = greedy):
  **PASS = D(br vs greedy) ≥ 1.25 × D(random vs greedy)** on held-out validation instances.
  FAIL consequence: exploitability proceeds on the fitted-scripted portfolio alone (the max is
  still well-defined) and the BR failure is itself a reported finding.

  > **RUN PRE-REGISTRATION (2026-07-06, before looking, promoted to the pivotal go/no-go).**
  > The greedy lever probes (`stress_sweep.py`, `hybrid_lever_probe.py`) showed crude
  > unpredictability vs a REACTIVE attacker is thin/negative, so the whole direction hinges on
  > whether an ANTICIPATORY best-response attacker can exploit a competent deterministic policy.
  > This gate tests exactly that. Setup: BR antagonist trained vs FROZEN greedy on the hybrid
  > route-reach arena (`scripts/br_gate.py`, 300 ep, existing flat head + route-reach mask; the
  > factored head is deferred and only built if this comes in marginal). Eval:
  > `evaluate_portfolio.py --problem hybrid --br gate=<actor> --attackers
  > none,random,targeted,gateway,br_gate --instances 24 --seed-base 20000019`.
  > Pre-registered readout: **PASS** = D(br) ≥ 1.25×D(random); **STRONG** = additionally D(br) ≥
  > D(targeted) (reproduces gen05's transferred +1667 > +1154, now trained). **FAIL** (BR ≤
  > random) = the learned attacker cannot exploit even a competent deterministic victim in
  > route-reach → the exploitability metric cannot rest on a learned BR; escalate (build the
  > factored head and re-gate, or fall back to fitted-scripted-only exploitability, or, if that
  > lever is also thin, the exploitability direction is not viable and we freeze on the gen03-06
  > diagnosis). Result appended below when training + eval complete.
  >
  > **RESULT (2026-07-06): GATE FAILED.** BR trained 300 ep vs frozen greedy, eval on 24 held-out
  > validation instances (greedy W(none)=847):
  >
  > | attacker | D vs greedy |
  > |---|---|
  > | random | +1031 ± 84 |
  > | targeted (scripted) | +1154 |
  > | gateway (scripted) | +714 |
  > | **br_gate (trained BR)** | **+871** |
  >
  > br/random = **0.84** (PASS needed ≥ 1.25): the trained BR is WEAKER than random blocking, and
  > weaker than the scripted `targeted` attack (1154). This reproduces gen04's ratio (0.84)
  > EXACTLY: even in the route-reach arena, against a competent deterministic victim, the learned
  > attacker cannot beat random. (Eval validity: the scripted rows reproduce gen05's greedy rows
  > exactly: targeted 1154, gateway 714.) The training log's rising attacked-latency (~2123 at
  > ep300) was exploration noise; the learned deterministic attack is weak (871): the classic
  > gen03/04 entropy-pinning signature. gen05's +1667 was a TRANSFER artifact (BR trained vs
  > exploitable learned policies, transferred to greedy), not a robustly trainable BR vs greedy.
  >
  > **Consequence:** the exploitability metric cannot rest on a learned BR against a competent
  > deterministic victim. Nuance kept for the decision: greedy's optimal congestion-aware
  > rerouting may be the HARDEST possible victim (a moving target); the metric's actual victims
  > are the learned arms, and the portfolio-max means a weak BR is simply dominated by the fitted
  > scripted attacks. Escalation decision is Kilian's (see the session record): factored head
  > re-gate, a minimal vanilla-vs-sacred training slice, or freeze on gen03-06.
  >
  > **SELF-CORRECTION + CORRECTED GATE (2026-07-06).** The gate above used the UNFIXED attacker
  > (no counterfactual reward, default 0.5*ln(N) entropy target, gamma 0.99): it is a gen04
  > REPLICA, so reproducing gen04 was expected and it does NOT test whether the fixes rescue the
  > attacker. Corrected gate (running): `--arena contested --reward-baseline twin
  > --antag-target-entropy 0.5 --gamma 0.997`, victim = frozen greedy, 300 ep. Rationale: the
  > zero-sum twin reward makes the attacker's reward = +(remaining - clean-greedy baseline) = its
  > MARGINAL damage stripped of the exogenous queue baseline (the SNR root cause of the small
  > Q-spread that entropy-pins the attacker); the lower entropy target lets a policy with real
  > Q-spread commit. Same PASS bar: D(br) >= 1.25 x D(random) on 24 held-out contested instances
  > (random vs greedy ~1718 in gen06; scripted targeted ~5000 = the STRONG bar). Eval:
  > `evaluate_portfolio.py --problem contested --br gate=<actor> --attackers
  > none,random,pathrand,targeted,br_gate --instances 24 --seed-base 20000019`. Result appended
  > when done; if PASS, a no-fix contested control run attributes the effect to the fixes.
  >
  > **CORRECTED GATE RESULT (2026-07-06): FAILS ROBUSTLY, and reveals the deeper mechanism.**
  > Held-out (24 val instances, greedy W(none)=6729): D(random)=+4733±213, D(pathrand)=+4618±176,
  > D(targeted)=+4920±218, **D(br_fixed)=+1666±185**. br/random = **0.35** (WORSE than the unfixed
  > hybrid gate's 0.84). Robust across snapshots: D(br) ep100=457, ep200=1890, ep300=1666: the
  > counterfactual reward DID help it climb (457->~1800) but it plateaus at ~0.35-0.39x random.
  > Telemetry (the diagnosis): antagonist entropy stayed **pinned at ~2.2 throughout** despite the
  > lowered 0.5 target, while alpha COLLAPSED (0.80->0.08): so the near-uniform policy is NOT held
  > up by entropy pressure (alpha is tiny) but by a **near-zero Q-spread** across block actions.
  > The counterfactual reward cleaned the TOTAL damage signal (Antag R stable ~4000) yet could not
  > create Q-spread that is not in the problem.
  >
  > **The unifying finding (deeper than gen04's "entropy pinning"):** on the stressed arena (ρ≈1)
  > the attack landscape is **large but FLAT**: every route-reach block causes similar cascading
  > queue damage, so there is no differentially-best attack to learn, and RANDOM blocking is
  > already near-optimal (4733 ≈ 96% of scripted's 4920). Where the landscape might be
  > differentiated (low stress) it is thin (capacity/stress probes). So a learned adversary has no
  > regime with an edge over simple random/scripted blocking: the attack surface is flat-where-
  > large and thin-where-differentiated. Entropy pinning is a SYMPTOM of the flat Q-landscape,
  > which is a property of the "block edges in a queueing network" adversary, not a fixable
  > optimisation issue. This explains gen03/04/06's below-random learned attackers at a mechanistic
  > level, and it also means the contested arena's attacks SATURATE (~4600-4920, all near-ceiling),
  > leaving no headroom to differentiate a vanilla from a sacred victim (gen05 ceiling-compression
  > in another guise). **Consequence:** the learned-BR component of the exploitability metric is
  > dead here (fair test, deep mechanism); freeze recommendation on record (session).
- **B9.v coping-channel probe**: an ε-randomised greedy (assignment noise, ε grid on validation
  instances) must reduce D under the fitted `targeted` attack relative to deterministic greedy
  by a nonzero margin (CI excluding 0). FAIL consequence: the unpredictability channel is dead
  in this arena → escalation rule above (arena moves; ledger re-issued; no training happens
  against a dead channel).

## Pre-registered interpretive branches

1. Primary met + Tier-2 secondaries ≈ 0: *adversarial co-training buys worst-case
   (unexploitability) but not average-case robustness*: the expected two-register outcome;
   Act IV headline.
2. Primary met + `dr` ≈ `sacred`: exposure-at-all suffices in this arena; the claim downgrades
   from "adversarial" to "attacked training"; reported as such (the controls exist exactly to
   force this honesty).
3. Primary met + `vanilla@tau` ≈ `sacred`: raw entropy suffices; claim downgrades to "noise is
   enough here"; the frontier comparison (clean cost at matched Expl) then decides whether
   co-training earns anything.
4. Primary NOT met (with gates passed): the five fixes are insufficient for the benefit even in
   its native register: Act IV becomes the sharpened impossibility result; freeze-and-write on
   the three-act diagnosis (still a complete thesis).
5. Any gate fails: recorded consequence fires (see the gate); no post-hoc metric changes, ever.

## Commands (sketch; EXACT commands + SHA pinned in the launch record)

```bash
# Phase B build gates first (branch gen07-contested), then per arm x seed:
PYTHONPATH=. python scripts/train_sacred.py --problem contested [--vanilla | --dr | --curriculum-population ...] --episodes <E> --seed <k> --group gen07_contested_matrix ...
# selection (validation attacker, validation instances), BR trainings (300 ep per arm),
# fitted-scripted grid on validation, then the portfolio:
PYTHONPATH=. python scripts/evaluate_portfolio.py --problem contested --policy ... --br ... --attackers none,random,<fitted...>,br_* --instances 30 --out experiments/gen07_portfolio_pair<k>.json
```

## Launch record (EMPTY: to be filled at launch; binding from that moment)

- git SHA: -
- arms/seeds: -
- pinned TO-FINALISE slots (arena, budget, curriculum schedule, entropy targets, γ, episodes,
  selection attacker, fitting grids): -
- B9 gate outcomes: -
