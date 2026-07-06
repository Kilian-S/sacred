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
- **G2 feasibility slice (I2, the GO/NO-GO):** on the ONE OD pair, `Expl(sacred) < Expl(shortest_path)`
  and SACRED trends toward `loss_mixed`. Expected PASS (the gap is structural). FAIL consequence:
  diagnose via the oracle (an RL-convergence problem, not a structural one: a strong attacker
  EXISTS here, unlike the congestion arena), then decide.

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
