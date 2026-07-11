# Generation: gen19_b1lite1 (B1-lite-1: within-episode pattern-of-life; the D restored to the headline game)

- **status: PRE-REGISTERED 2026-07-11 (Kilian: "build B1-lite-1 autonomously"); oracle screen DONE
  (numbers pinned below, BEFORE training); training binding at launch.**
- **git SHA:** the commit landing this ledger + the trainer.

## The game (within-episode dynamism)

An episode = S sorties on a fixed instance (35-159, the favourable post-fix headline instance).
Each sortie the fleet STACKS on one route (fleet-route). The interdictor commits K=1 assets by
SOFTMAX-best-responding (temperature tau) to the defender's REALISED routes over a trailing WINDOW
of w recent sorties - PATTERN-OF-LIFE: the enemy positions against your recent operations, not your
long-run mixed strategy. Soft interception, mission objective, latency-free (isolates the pure
dynamism effect). This is the first SACRED game with WITHIN-EPISODE dynamism (the "D" of SDVRP,
absent from the single-shot headline); the defender's optimal play is HISTORY-DEPENDENT (its own
recent pattern is state).

## Oracle screen (scratch/within_episode_screen.py, DONE 2026-07-11, no training)

Because the softmax-BR adversary is a deterministic-transition function of the window, the
defender's optimal history-dependent policy is an average-cost MDP over the window state, solved
EXACTLY by relative value iteration. Screened 35-159 & 62-97 over w in {1,2,3}, tau in {0.05,0.15}:

| instance | V_eq (single-shot stacked eq) | operating point | static_det | iid_eq (static-mixed) | history_opt (dynamic) |
|---|---|---|---|---|---|
| 35-159 | 0.206 | **w=3, tau=0.15** | 0.613 | **0.147** | **0.049** |
| 35-159 | 0.206 | w=2, tau=0.15 | 0.613 | 0.151 | 0.029 |
| 62-97 | 0.216 | w=3, tau=0.15 | 0.735 | 0.155 | 0.038 |

**The ordering holds and is non-degenerate at tau=0.15** (sharp tau=0.05 makes the adversary
trivially dodgeable -> history_opt ~ 0, degenerate; tau=0.15 keeps it a real game): static-det is
destroyed (0.613), the static-mixed defender lands ~its equilibrium (0.147), and the HISTORY-AWARE
defender exploits the adaptive adversary's predictability to **0.049 = 3.0x lower than static-mixed**.
Dynamism pays, with a computable optimum. **OPERATING POINT LOCKED: 35-159, w=3, tau=0.15.**

## The SACRED experiment (arms + metric, PRE-REGISTERED)

`scripts/train_b1lite1.py`: a menu-select fleet-route SAC policy that OBSERVES the window (per-route
recent frequency delivered undiluted at the head as a 3rd route-feature column [cost, vuln,
window_freq], lr 3e-2 - the gen11b/A1 mechanism) and routes; episodes of S=40 sorties chained with
bootstrapping (the window is state; the action shifts it), gamma 0.95; adversary = analytic
softmax-BR (tau 0.15) to the window; reward = -interception_loss * expected mission-failure.

| arm | value |
|---|---|
| static_det (deterministic) | 0.613 (oracle) |
| iid_eq (static-mixed, history-BLIND) | 0.147 (oracle) |
| **SACRED (history-aware)** | trained |
| history_opt (dynamic optimum) | 0.049 (oracle) |

> **PRIMARY (dynamism learned): SACRED's stationary-tail per-sortie mission-failure < iid_eq 0.147**
> (beats the history-blind static-mixed defender = it learned to condition on its own pattern), on
> >= 2/3 seeds. **STRONG: within 0.03 of history_opt** (<= 0.079). Best-checkpoint discipline as
> standing. A history-BLIND control (same policy, window feature ZEROED) is the causal check: it
> must sit at ~iid_eq, confirming the gain is the window conditioning, not the arch.

Screened numbers pin the instance/operating point BEFORE training (house rule); seeds {0,1,2}.

**Launch (2026-07-11):** `scratch/gen19_b1lite1.sh` (3 seeds history-aware + 1 no-window control,
8000 sorties, eval-every 500). Smoke (240 sorties) already PASSED the primary: SACRED 0.131 <
iid_eq 0.147 with the window-feature weight strongly negative (rw[2] -> -4.35 = the policy avoids
recently-frequent routes, the history-aware dodging the oracle predicts).

## RESULT (2026-07-11, 3 seeds + control, ~1.5 h): **PASS on every clause, STRONG, with a clean causal control and a robust worst-case row**

| arm | per-sortie mission-failure vs the pattern-of-life adversary |
|---|---|
| static_det (deterministic) | 0.613 (oracle) |
| iid_eq / **NO-WINDOW control** (history-blind) | 0.147 (oracle) / **0.148 (measured control)** |
| **SACRED history-aware** | **0.050 +/- 0.001** (best-ckpt, 3 seeds) |
| history_opt (dynamic optimum) | 0.049 (oracle) |

> **SACRED reaches the dynamic optimum: 0.050 +/- 0.001 vs history_opt 0.049**, PRIMARY 3/3
> (< iid_eq 0.147), STRONG 3/3 (within 0.03 of history_opt). Per-seed best 0.049-0.050.

**The causal control is textbook:** the SAME policy with the window feature ZEROED (`--no-window`)
lands at **0.148 = iid_eq exactly** - history-blind, it can only play the static-mixed strategy;
given the window, it learns the history-optimal dynamic policy. The gain (0.148 -> 0.050) is
CAUSALLY the window conditioning, not the architecture. The window-feature weight trains strongly
negative (rw[2] ~ -4 to -5) = the policy learns to AVOID recently-frequent routes (anti-repeat),
exactly the anticipatory dodging the oracle predicts.

**WORST-CASE robustness row (addressing the expansion-critique flag that the softmax adversary is a
QUANTAL-RESPONSE, not worst-case, opponent):** the history-aware policy's MARGINAL route
distribution is a genuine spread mixed strategy (0.22/0.22/0.20/0.16/0.06/0.04), and its single-shot
exploitability under a NON-ADAPTIVE oracle best-response is **0.219 ~ the single-shot equilibrium
V_eq 0.206** (only +6%). So the policy is NOT fragile: it gives up essentially nothing against a
worst-case non-adaptive attacker while exploiting the realistic pattern-of-life adversary to 0.050.
The claim is therefore honest and bounded: *against a pattern-of-life (bounded-rationality,
window-adaptive) interdictor, a history-aware defender reaches the dynamic optimum, and its routing
remains a sound equilibrium mixed strategy against a non-adaptive worst-case interdictor.*

**What is established (the D restored, and SACRED essentially solves it):** the first SACRED game
with WITHIN-EPISODE dynamism. A history-aware RL policy, given its own recent pattern-of-life,
learns a history-dependent routing that reaches the computable dynamic optimum (0.050 vs 0.049),
beating the history-blind static-mixed strategy 2.9x, with the gain causally attributed to the
window feature and no worst-case fragility. This reconnects the anticipation-vs-reactive tension
(Ritzinger) inside a game where anticipation about one's OWN observable pattern provably pays, on a
computable optimum. Caveats: latency-free, single instance (35-159), fleet-route stack (one route/
sortie); the demand-side S (Poisson arrivals) is full B1, not this rung.

**LADDER (gen19, per-sortie mission-failure):** static_det 0.613 > iid_eq/no-window 0.148 >
**SACRED 0.050** ~ history_opt 0.049 (worst-case non-adaptive: SACRED marginal 0.219 ~ eq 0.206).
