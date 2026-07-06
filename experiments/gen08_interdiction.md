# Generation: gen08_interdiction (Act IV: adversarial RL that provably works: convoy routing as a security game)

- **status: DRAFT PRE-REGISTRATION (opened 2026-07-06). Runs NOTHING yet.** Becomes binding when
  the interdiction env (ROADMAP Phase I1) is built + unit-tested, the feasibility slice (I2) is
  set up, and Kilian approves launch. Design: `REDESIGN_INTERDICTION.md`. Why this replaces gen07:
  the congestion adversary has a FLAT attack landscape (gen07 corrected BR gate = 0.35× random);
  interdiction is a Stackelberg security game where the mixed-strategy defender provably wins.
  Proof (game-theoretic, before training): `scratch/interdiction_game_probe.py`: deterministic
  routing 100% intercepted → minimax mixed 17-33% on the real Kaliningrad graph.
- **git SHA:** to pin at launch (new branch `gen08-interdiction` off `main`).

## Question (fixed before looking)

**On a contested road network, does a SAC dispatcher trained adversarially (SACRED: SAC entropy +
ATLA vs a learned interdictor) learn a MIXED-STRATEGY route policy that is less exploitable to
interception than shortest-path routing and than a non-adversarially-trained SAC, approaching the
computable minimax equilibrium?** The security-game structure guarantees the gap EXISTS
(deterministic = maximally exploitable); the empirical question is whether deep RL *learns* toward
it from experience on a real graph, and by how much, vs the baselines.

## Arena (locked decisions, Kilian 2026-07-06)

**Kaliningrad graph, SINGLE convoy first.** One OD pair (base→FOB) with high edge-connectivity
(candidate: **33→71**, edge-connectivity 6, route length ~4.1; or 110→135, edge-conn 3, for
continuity with the campaign geometry). Antagonist commits **K** interdiction assets to edges per
sortie, HIDDEN from the defender until struck. Convoy routes base→FOB (next-hop routing first, so
the mixed strategy emerges from SAC per-step entropy; candidate-route-set form is the fallback).
Interception: crossing an interdicted edge → discrete high-magnitude loss. K swept (the difficulty
knob; the mixed-equilibrium band is roughly 1 ≤ K < #edge-disjoint-routes). Multi-convoy is LATER.

## Arms

| arm | training | role |
|---|---|---|
| `shortest_path` | none (deterministic) | the operational baseline; provably maximally exploitable (loss_det) |
| `equilibrium` | none (computed by the LP/double-oracle oracle) | ground-truth minimax value (loss_mixed) + the equilibrium mixed strategy |
| `vanilla` | SAC, no adversary (routes to minimise travel cost) | non-adversarial control; expected near the exploitable extreme |
| `sacred` | SAC entropy + ATLA vs a learned interdictor | the headline; expected to approach the equilibrium |

## Decision metric (PRE-REGISTERED: to finalise wording at env-lock, before any result)

Over held-out sorties (paired demand/instance seeds), and swept over K:
- **Exploitability** `Expl(arm)` = interception loss of the frozen policy under a BEST-RESPONSE
  interdictor (the learned interdictor AND the equilibrium/oracle interdictor; take the max, so a
  weak learned interdictor cannot flatter the defender: the gen07 portfolio-max lesson).

> **PRIMARY:** `Expl(sacred) < Expl(vanilla)` AND `Expl(sacred) < Expl(shortest_path)`, with the
> paired 95% CI excluding 0, across seeds. **STRONG:** additionally `Expl(sacred) ≈ loss_mixed`
> (SACRED reaches the equilibrium; distance-to-equilibrium small).

Secondaries (reported): distance-to-equilibrium over training (does SACRED converge toward the
minimax value?); clean (no-interdiction) travel-cost premium of sacred vs shortest-path (want
small: robustness at low nominal cost); the K sweep and edge-connectivity sweep as curves; the
defender's realised route distribution vs the equilibrium mixed strategy (mechanistic validation).

## Gates (cheap, pre-registered; gate expensive training on them: the campaign dogma)

- **G1 env-fidelity: PASSED 2026-07-06.** The env (`src/envs/interdiction.py`) reproduces the
  oracle's `loss_det` (deterministic defender, Monte Carlo → 1.0) and `loss_mixed` (equilibrium
  mixed defender vs equilibrium attacker → the minimax value, ±0.03) on synthetic AND Kaliningrad
  33→71 (gap ≥ 0.8). Oracle: `src/baselines/interdiction_oracle.py`. Tests:
  `tests/test_interdiction_{oracle,env}.py`.
- **G2 feasibility slice (I2, the GO/NO-GO): PASSED 2026-07-06.** Kaliningrad 33->71, K=1, 6
  edge-disjoint routes (oracle loss_det=1.000, loss_mixed=0.167). Defender SAC trained vs the
  ORACLE best-response interdictor (fictitious play on the empirical average, which converges),
  `scripts/train_interdiction.py`, 1500 sorties, seed 0:

  | arm | exploitability (interception under best-response) |
  |---|---|
  | shortest_path (deterministic classical) | **1.000** |
  | vanilla SAC (no adversary) | 0.275 |
  | **SACRED (adversarial)** | **0.235** (equilibrium 0.167) |

  **Adversarial training cut interception 100% -> 23%**, converging toward the computed equilibrium
  (trajectory 0.43 -> 0.20-0.24 over training; distance-to-equilibrium 0.068). The direction WORKS:
  a deep-RL routing agent learns a mixed strategy approaching the security-game equilibrium and is
  ~4x less exploitable than the deterministic classical baseline. **This is the project's first
  positive result.** Config that mattered (a fixed reward-scaling bug + fictitious-play averaging):
  reward_scale 1.0, interception_loss 10, lr_actor 3e-4, best-response to the empirical AVERAGE
  play (not the instantaneous policy, which oscillates/chases).

  **Honest caveats (refinements for the full experiment I3):** (1) the SACRED-vs-vanilla gap is
  THIN here (0.235 vs 0.275) because this instance is SYMMETRIC (6 equivalent disjoint routes ->
  uniform equilibrium), so vanilla's max-entropy SAC mixes incidentally and is already fairly
  robust; the deterministic classical baseline is the clean contrast. To separate SACRED from
  vanilla cleanly, I3 needs ASYMMETRIC instances (non-uniform equilibria: shared-edge routes, K>=2,
  or heterogeneous vulnerability) where vanilla's uniform-ish mixing is measurably suboptimal and
  SACRED must learn the specific non-uniform equilibrium. (2) A trailing-window average (vs the
  all-history running average used here) would sit closer to 0.167. (3) Multiple seeds + the K /
  connectivity sweeps are the I3 matrix.

## I3 pre-registration (DRAFT, opened 2026-07-06 after the I2 pass; binding at launch with SHA)

**Goal:** open the SACRED-vs-vanilla gap that the symmetric I2 instance could not show (uniform
equilibrium: vanilla's incidental mixing is near-optimal there), via ASYMMETRIC instances with
non-uniform equilibria; plus seeds, and the K / connectivity axes as curves.

**Design decisions (Kilian, 2026-07-06):** asymmetry class = heterogeneous edge vulnerability
first (class (c): probabilistic interception, no action-space change; shared-edge route sets and
the route-trie policy are the later class (b)); vulnerability model = LENGTH-DERIVED band
(exposure scales with transit time: candidate-edge lengths mapped affinely into a band; objective,
graph-derived, no hand-tuned threat map). **Correction recorded:** K>=2 alone is NOT an asymmetry
source: on edge-disjoint routes with hard interception the equilibrium is uniquely uniform for
every K (best response = the top-K defender masses), so K stays the budget/sweep axis only.

**Instances (pinned by the oracle probe `scratch/vuln_band_probe.py`, 2026-07-06: pinned by
probes, never by outcomes):**

| id | OD | routes | interception | K | loss_det | shortest | uniform | equilibrium | uniform/eq |
|---|---|---|---|---|---|---|---|---|---|
| A (I2 headline) | 33->71 | 6 disjoint | hard | 1 (sweep 1-3) | 1.000 | 1.000 | 0.167 | 0.167 | 1.00x |
| **B (primary)** | 33->71 | 6 disjoint | band (0.15, 0.95) | 1 (sweep 1-2) | 0.266 | 0.449 | 0.158 | **0.063** | **2.51x** |
| C (connectivity contrast) | 110->135 | 3 disjoint | band (0.15, 0.95) | 1 | 0.690 | 0.950 | 0.317 | 0.258 | 1.23x |

Instance B's equilibrium is strongly non-uniform (d in [0.066, 0.237], d_i ~ 1/p_i*, closed form
verified against the LP in tests); uniform mixing is 2.51x suboptimal, which is exactly the
separation the vanilla control cannot track. Instance C shows where calibrated mixing has little
headroom (few routes, similar p*): reported as the connectivity axis, not a gap claim.

**Arms:** shortest_path (deterministic operational default), uniform (uncalibrated-mixing
reference row, computed not trained), vanilla (SAC, travel-cost objective, no adversary), sacred
(SAC vs the oracle best-response interdictor on the empirical average play, as I2), equilibrium
(oracle ground truth). Learned-antagonist co-evolution is the I3 follow-on build, not this matrix.

**Seeds and budget:** seeds {0, 1, 2} per trained arm; 3000 sorties per arm (I2 converged by
~1500; 2x margin); measured ~0.27 s/sortie (2026-07-06 smoke) -> ~27 min per instance-seed run
(both trained arms), ~2.4 h serial for the matrix, ~3x parallelisable on the M4.

**Decision metric (PRIMARY, on instance B):** exploitability of the TRAILING-WINDOW empirical
play (window 500) under the oracle best-response interdictor, end of training:
> `Expl_win(sacred) < Expl_win(vanilla)` on >= 3/3 seeds AND in the pooled mean, and
> `Expl_win(sacred) < Expl(shortest_path)`.
Rationale for the window reading as primary: the Stackelberg adversary best-responds to the
defender's observed PATTERN of play, which is the empirical mixture (also the fictitious-play
quantity that converges); the instantaneous-policy reading `Expl_policy` oscillates around it (FP
dynamics, visible in the smoke) and is reported as a SECONDARY alongside `Expl_avg` (all-history,
the I2 continuity metric). **STRONG:** `Expl_win(sacred)` within 0.05 of loss_mixed on B.

**Secondaries (reported, not gated):** distance-to-equilibrium trajectories; the defender's route
distribution vs the equilibrium mixed strategy; vanilla's late-training collapse check (does its
window exploitability rise toward the shortest-route bound as entropy anneals?); K sweep on A
(value K/6) and B; the A-vs-B-vs-C connectivity/asymmetry comparison; clean travel-cost premium
of sacred vs shortest_path.

**Honest instance property (recorded up front):** length-derived vulnerability correlates with
travel cost (long edges are both slow and exposed), so the nominal objective partially aligns
with the security objective and early-training vanilla can sit below uniform (seen in the smoke).
The pre-registered comparison is therefore on END-of-training play, and the probe already fixes
the relevant endpoints: vanilla collapsing to the shortest route lands at 0.449 vs equilibrium
0.063 on B. If vanilla does not fully collapse in 3000 sorties, that is reported as-is (its
mixture is still uncalibrated; the gap claim stands or falls on the measured window numbers).

**Gates:** G3 soft-instance fidelity: **PASSED 2026-07-06** (env reproduces the soft oracle
equilibrium by Monte Carlo, `tests/test_interdiction_env.py::test_G3_soft_env_reproduces_oracle_kaliningrad`;
closed-form equilibrium verified in `tests/test_interdiction_oracle.py`; suite 127 green).
Timing gate: measured (smoke above). **Launch: waiting on Kilian (never before this section is
finalised + SHA pinned).**

**Commands (matrix sketch):**
```bash
# instance B (primary), 3 seeds:
for s in 0 1 2; do PYTHONPATH=. .venv/bin/python scripts/train_interdiction.py \
  --sorties 3000 --seed $s --edge-vuln-band 0.15,0.95 --json-out results_B_seed$s.json; done
# instance A (headline, hard): drop --edge-vuln-band; K sweep: --K {1,2,3}
# instance C: --od 110-135 --edge-vuln-band 0.15,0.95
```

## Commands (sketch; exact + SHA at launch)

```bash
# equilibrium oracle (ground truth): already prototyped:
PYTHONPATH=. python scratch/interdiction_game_probe.py   # -> loss_det, loss_mixed, mixed strategy
# env build + tests (Phase I1), then the feasibility slice (Phase I2):
PYTHONPATH=. python scripts/train_sacred.py --problem interdiction --od 33-71 --K <k> [--vanilla | sacred flags] ...
# eval: interception under best-response + oracle interdictor, held-out sorties, K sweep.
```

## Notes for the incoming agent

- The equilibrium oracle (`scratch/interdiction_game_probe.py`) is the single most valuable asset
  here: it gives loss_det/loss_mixed and the equilibrium mixed strategy for any (graph, OD, route
  set, K): the ground truth SACRED is scored against. Harden it into a reusable module (ROADMAP I0).
- Reuse from `gen07-contested`: the counterfactual/twin reward idea (interception is already
  high-SNR, so it may be unnecessary here: test), the entropy machinery (now an ASSET), and the
  whole evaluation discipline (held-out, paired, dual-level stats, gate-before-train).
- Keep the SACRED invariants (SAC, protagonist/antagonist, RL). Single convoy, Kaliningrad, before
  any richness. Never train without pinning this ledger's SHA + finalised metric first.
