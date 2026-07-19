# Generation: gen30_secure_flp (security-aware facility location: the Obj-4 oracle act at the placement tier)

- **status: PRE-REGISTERED 2026-07-19 (mission brief from Kilian, supervisor direction
  2026-07-19; ENTIRELY oracle/eval-only, additive scratch code + this ledger + figures, no
  training anywhere, so the act runs under full autonomous authority per the standing
  "oracle probes are free" rule). This ledger is committed BEFORE any analysis code exists;
  results are appended below and nothing above the RESULTS line changes after.**
- **git SHA:** the commit landing this ledger (pre-registration); analysis commits pin their own.

## Why (data-backed; all sources committed)

1. **Objective 4, verbatim from the assessed literature review:** "Incorporate SBO into the
   SACRED framework, utilising a neural network metamodel to approximate facility location and
   fleet composition, thereby enabling the holistic, simultaneous evaluation of strategic supply
   chain design alongside the operations-level SDVRP." This act is that sentence built on the
   oracle machinery that survived every critique.
2. **The security value of a depot design is an exact, cheap oracle quantity** (the induced
   interdiction game's equilibrium; seconds-scale LP). At K=1 the disjoint heuristic sits at or
   near that optimum (R0a), so no trained policy is needed to price a design; the R0a wording
   applies throughout: we price whatever policy will actually be deployed.
3. **The gen29 screen proved the multi-stream correlation gap is the project's ONLY gap that
   survives a complete hostile baseline family** (median 31% vs the m-pairing cap over 55 cells;
   B4's original 14.4% row; probes `738ddd1`, `e6c29e2`, gen29's own screen). Depot and corridor
   OVERLAP is what creates that gap at design level, and classical facility location prunes
   overlap as waste. That tension is the act's headline question.
4. D2 measured strategic/operational tier coupling and B1 the joint-vs-sequential integration
   gap; this act completes that arc at the placement tier.

## The game (pinned; identical mechanics to the committed b4/gen29 probes)

Kaliningrad primary (`scratch/b4_multiod_probe.build_graph`, imported so the graph is
byte-identical to probes `738ddd1`/`e6c29e2`). **F=3 demand streams** (one convoy each, so three
convoys in the air per sortie: the standing N=3), **K=1** hidden interdictor committing one edge
of the union candidate list, soft interception via `length_band_vulnerability` band (0.15, 0.95),
**k_extra=8 menus** (`build_route_set`: edge-disjoint prefix + k-shortest), **mission objective
P(>=1 of 3 lost)** (the B3 law: the additive objective is provably correlation-gap-free, so the
loss-averse coupling is load-bearing). The **security value of a design = the exact joint
equilibrium `v_joint`** (LP over joint route triples), always reported with the complete family
beside it.

## Baseline family (NON-NEGOTIABLE on every security number; the recursion dogma)

Per design: `v_det` (best deterministic joint plan) · best INDEPENDENT product (alternating
per-stream LPs, 4 restarts; an upper bound on the independent class, disclosed) ·
deconflict-uniform (payoff-blind) · **the in-sample m<=4-pairing cap (`best_m_pairings`, the
hardest row and the honest denominator for every redundancy claim)** · exact joint equilibrium.
Standing wording (not recomputed): tabular FP with the same oracle ties the equilibrium on any
single instance, so no sentence anywhere reads "SACRED superior"; design conclusions are about
DESIGNS priced by oracle values.

## Component A: the (cost, security) design frontier

- **Design** = a single depot site `s` serving the fixed target set.
- **Site population:** all nodes with degree >= 3 whose route sets to all three targets are
  valid (4 <= R_f <= 14, the b4_widen band; joint tensor under the standing size guard), capped
  at 150 by seeded subsample if larger. Every surviving site is reported (prevalence, no
  cherry-picking).
- **Targets (primary):** T = (212, 188, 195): gen29's screened headline targets, so source 147
  is a known committed anchor. DISCLOSED: this target set was originally screened for a large
  coordination gap; therefore a second, UNSCREENED seeded random target set is run as a
  robustness row (does the frontier shape replicate?).
- **Cost** = classical FLP service cost: sum over targets of the shortest-path `w`-length from
  `s`. Opening costs: constant across single-depot designs, so omitted from A (recorded).
- **Deliverables:** the Pareto frontier over (cost, v_joint) with the p-median (cost-argmin)
  and security-argmin marked; the knee (Pareto point nearest the normalised utopia corner,
  min-max normalisation over the population; recorded convention); the quantification
  sentence ("the cost-optimal depot pays X% security vs the security-optimal; the knee buys
  most of it back for Y% extra cost"); prevalence histograms over ALL sites (v_joint and
  gap-vs-cap); and a **policy-robustness row**: Spearman rank correlation between v_joint and
  v_cap across sites (if the ranking survives, the frontier's design advice does not depend on
  deploying an exact-equilibrium mixer: the R0a deployed-policy discipline at design level).
- **Expectation (pre-written):** cost and security are in tension (the cost-argmin pays a
  material premium vs the security-argmin). **Fail branch (pre-written): if the premium is
  < 5% or the argmins coincide, placement on this graph is security-free and that is the
  reported finding.**

## Component B: redundancy priced by the correlation gap (the headline component)

- **Design** = a depot pair (d1, d2) from the screened site list. Pairs are sampled to SPAN the
  separation axis: stratified by inter-depot shortest-path distance quantiles, seeded, with NO
  payoff peeking at selection time (the anti-circularity rule: the R0c lesson); every sampled
  pair is reported.
- **Two variants per pair, identical openings (2 = 2) and identical classical service cost by
  construction:**
  - **CLASSICAL (what FLP builds):** each stream is served only by its nearest-by-service-cost
    depot; its route set is `build_route_set(nearest_depot, t)`.
  - **REDUNDANT (dual-servability):** each stream may be served by EITHER depot; its route set
    is the union of both depots' route sets for that target (shape-agnostic joint tensor, per
    the b4_widen machinery).
  The classical joint play set is a subset of the redundant one, so
  v_joint(redundant) <= v_joint(classical) always; **the difference is the exact value of the
  dual-servability redundancy that classical FLP prunes.**
- **Cost accounting (decision recorded):** openings matched; nearest-assignment service cost
  identical between variants; the redundancy's real price is the OPERATING premium = expected
  route length under the security-optimal joint mixture minus the classical service cost,
  computed exactly from the equilibrium marginals and reported per design (the R0b fleet-cost
  discipline).
- **Honest metric (the mission's binding form):** every gain is reported **vs the m-pairing CAP**,
  two ways: (i) gap-vs-cap on the redundant design (how much of its value needs true joint
  mixing vs is reachable by a napkin mixture of <= 4 plans); (ii) v_cap(redundant) vs
  v_joint(classical) (redundancy under naive deployment vs the classical design under PERFECT
  deployment: the deployment-robust value of redundancy).
- **References:** the best single depot from Component A (its cost and v_joint) as the
  one-central-depot comparison lines.
- **Deliverables:** the value-of-overlap curve (relative v_joint gain classical -> redundant vs
  inter-depot distance and vs candidate-edge Jaccard between the depots), prevalence over all
  sampled pairs, and the headline sentence in whichever claim shape the numbers support.
- **Expectation (pre-written):** redundancy buys a material equilibrium gain that survives the
  cap row. **Fail branch (pre-written, verbatim from the mission): if overlap buys < 5% vs the
  cap, that is the finding, reported plainly.**

## Component C: the Obj-4 metamodel rider (DROP FIRST if time bites)

Fit the existing `SurrogateMLP` (`src/sbo/surrogate.py`, the F3 pipeline pattern) on CHEAP
pre-solve design features -> v_joint, over the pooled A + B design rows. Features (no LP at
query time): per-stream service costs, per-stream route counts and disjoint-route counts
(min-cut), union candidate-edge count, mean/max candidate-edge vulnerability, inter-depot
distance and depot edge-set Jaccard (0/self for single-depot designs), redundancy flag.
70/30 seeded split; report held-out RMSE, Spearman rho, and argmin regret (the true rank of the
surrogate's chosen design). **Expectation: Spearman >= 0.8** (the F3/D3 precedents 0.894/0.959);
fail = reported as measured. The D1-style acquisition loop at matched budget vs random search is
optional and drops first.

**Fleet composition (decision recorded):** NOT in the design space. One convoy per stream is
pinned (the gen29 convention); per-stream fleet sizing would conflate placement with allocation
and needs new game conventions, so it is recorded as future work in one line, per the mission's
"your call, say so in the ledger".

## Held-out city row (the zero-shot design row, ZST spirit)

**Gdansk** (`data/maps/gdansk/`, never used to tune anything in this act; loader mirrors
`src/utils/graph_utils.load_osm_graph_and_demands` edge conventions exactly: w =
max(1, round(length_m/100, 1)), largest connected component; disclosed as a reimplementation
because the shared loader requires a demand-tasks file). Repeat the two headline figures at
reduced scale (~60 sites, ~30 pairs, seeded UNSCREENED random targets): does the (cost,
security) tension and the redundancy value replicate on a graph none of this act's choices
ever saw?

## Design decisions ledgered (one line of reasoning each)

1. Graph identity: import the committed probes' builder for Kaliningrad (byte-identical
   comparability with `738ddd1`/`e6c29e2`).
2. Security value = exact joint equilibrium, family beside it (baseline-completeness dogma;
   never det/eq as an aiming metric anywhere in this act).
3. Primary targets = gen29's (disclosed as screened); unscreened second draw + Gdansk carry the
   anti-cherry-pick burden, plus full-population prevalence on every headline.
4. Pair sampling stratified by distance, seeded, payoff-blind (no gap screening of designs: the
   act measures the axis, it does not aim along it).
5. Opening-cost model: omitted within matched-opening comparisons; A is all p=1, B is all p=2,
   and the A-vs-B reference is disclosed as one extra opening (no invented prices).
6. Knee convention: min-max normalised distance to the utopia corner (standard, recorded).
7. Fleet size N fixed (see Component C note).
8. Single-process sweep with BLAS thread caps exported (the cap-all-pools dogma, applied even
   though no multi-process launch happens).
9. Figures to `assets/gen30_*.png`; artefacts to `models/runs/gen30_secure_flp*.json`; numbers
   live only in this ledger.

## Claim shapes (pre-written; use whichever the numbers support, never more)

- A lands: "cost-optimal placement silently pays X% security; the (cost, security) frontier and
  its knee are computable exactly, per design, in seconds."
- B lands: "security-aware facility location buys redundancy that classical FLP provably prunes;
  its value is the correlation gap, measured against the complete hostile rule family including
  oracle-assisted mixtures."
- C lands: "a neural metamodel prices strategic designs against the operational security game,
  enabling the holistic, simultaneous evaluation Objective 4 promised."
- Any component fails its expectation: the boundary is the finding, stated first, on our terms.

## Compute envelope

Every design is a seconds-scale LP (b4_widen precedent: F=3 cells solve in seconds). Budget:
~150 (A) + ~2 x 40 (B) + ~120 (Gdansk) + robustness draws ~ 400-500 joint solves; target well
under 1 h wall, hard ceiling 3-4 h per the mission. Single process; BLAS pools capped.

## RESULTS (appended per component; nothing above changes)
