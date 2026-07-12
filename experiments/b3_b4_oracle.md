# Block B oracle items: B3 risk-aversion spectrum, B4 multi-OD correlation-gap probe

- **status: PRE-REGISTERED 2026-07-12 (NEXT_STEPS_MASTER Block B items B3/B4; ORACLE-ONLY, no
  training); results appended per item.**
- **git SHA:** the commit landing this ledger + the two scripts.

## B3: the risk-aversion spectrum (`scratch/b3_risk_spectrum.py`)

**Why (CRITIQUE_EXAMINER §6 item 9):** the objective finding is currently binary (risk-neutral:
the deterministic-vs-mixed gap collapses; mission: it holds). One oracle sweep turns the
modelling choice into a measured LAW ("the price of predictability as a function of
loss-aversion") and immunises against "you picked the objective where you win".

**Design:** objective family = `linear` (risk-neutral) and `threshold` P(>= m lost) for m = N..1
(m=1 = the mission objective, maximal loss-aversion). Compute loss_det, loss_mixed and det/eq per
objective on (a) the two headline instances at N=3 (m in {3,2,1} + linear) and N=5 (m in {5..1} +
linear); (b) the 40-OD Kaliningrad screen population at N=3 (median curve). Report det/eq vs
loss-aversion; figure `assets/b3_risk_spectrum.png`. Descriptive; expectation (recorded): the gap
rises monotonically with loss-aversion, ~1 at risk-neutral.

### B3 RESULT (2026-07-12): a THREE-REGIME law, sharper than the pivot-era binary

(det/eq per objective; artefacts `models/runs/b3_risk_spectrum.json`, `assets/b3_risk_spectrum.png`)

| instance | linear (risk-neutral) | P(>=2) | mission P(>=1) |
|---|---|---|---|
| 35-159 N=3 | 1.83 | 0 (both values 0: degenerate) | **3.39** |
| 62-97 N=3 | 1.66 | 0 (degenerate) | **3.23** |
| 35-159 N=5 | 1.44 | 1.29 (non-degenerate at N=5) | **3.13** |
| 62-97 N=5 | 1.35 | 1.58 | **3.38** |
| population median (40 ODs, N=3) | 1.34 | 0 | **2.30** |

**The law (replaces the binary "risk-neutral dilutes / mission holds"):** the objective family has
THREE regimes at K=1. (i) Risk-neutral: a modest gap (1.3-1.8x) that deterministic spreading
partly closes. (ii) Intermediate thresholds (m >= 2 while the fleet fits on edge-disjoint
routes): DEGENERATE IN FAVOUR OF DETERMINISM: a spread plan makes P(>= m lost) exactly zero, so
neither mixing nor calibration matters at all (both values 0). At N=5 (fleet no longer fits
disjointly) m=2 re-enters the non-degenerate zone (1.29-1.58). (iii) The any-loss mission
objective: randomisation essential (2.3-3.4x, growing with the population's headroom).
**Thesis sentence:** the mission objective is not "the objective where we win": it is the unique
member of the family in which the deterministic planner cannot escape by spreading, and the sweep
maps exactly where that boundary sits (m relative to fleet-vs-disjoint-route capacity). The
"price of predictability" curve is now measured, not asserted.

## B4: the multi-OD correlation-gap probe (`scratch/b4_multiod_probe.py`)

**Why (CRITIQUE_EXAMINER §6 item 10):** all convoys currently share one OD. The multi-OD game
(different destinations sharing corridor edges) is only worth building if CORRELATED joint
routing beats INDEPENDENT per-convoy mixing by a real margin.

**Design:** N=2 convoys, one per destination: sample Kaliningrad triples (s, t1, t2) with deg >= 3
and route-set candidate-edge overlap (Jaccard >= 0.05: corridor-sharing); K=1 over the union of
candidate edges; mission objective under independence given the interdiction set. Compute:
v_joint = exact minimax over JOINT route pairs (LP over R1 x R2); v_indep = best PRODUCT
distribution (alternating per-convoy LP best response, 5 restarts; a local optimum, i.e. an UPPER
bound on the true independent value, disclosed); v_det = best deterministic pair. Correlation gap
= (v_indep - v_joint) / v_joint over ~15+ triples.

> **Pre-committed reading:** median correlation gap >= 10% on corridor-sharing triples => the
> multi-OD game is a justified Tier-3 build (correlation without stacking exists). < 10% =>
> a clean scoping negative for free: the single-OD stacking game already captures the
> coordination content at this family, and the Tier-3 item is dropped with a measured basis.

### B4 RESULT (2026-07-12): median correlation gap 14.4% - the multi-OD game is JUSTIFIED

15 corridor-sharing triples (candidate-edge Jaccard 0.05-0.34), N=2, K=1, mission
(artefact `models/runs/b4_multiod_probe.json`):

> **median correlation gap (v_indep - v_joint)/v_joint = 14.4%; >= 10% on 10/15 triples**
> (range ~0-26%; the gap concentrates where corridor sharing is material, Jaccard >= ~0.15).
> v_det >> v_indep > v_joint everywhere (e.g. 119->62,278: det 0.458, independent-mixing 0.199,
> correlated joint 0.159).

**Pre-committed branch: JUSTIFIED.** Correlated joint routing of convoys with DIFFERENT
destinations beats the best independent per-convoy mixing by a real margin on corridor-sharing
triples, i.e. genuine coordination content exists WITHOUT stacking. The Tier-3 multi-OD
interdiction game (the move toward the VRP of the title) has a measured basis; the learnability
risk (gen18's exploration boundary) stands as recorded. Note v_indep is a local-search upper
bound (disclosed), so the true gap can only be LARGER.
