# GEN29_MULTIOD_HANDOFF.md: the closing experiment (three-stream coordination; build brief for the implementing Fable instance)

> **Provenance.** Written 2026-07-18 by the critic instance (Critic Aerial) at Kilian's request,
> after his sign-off on the framing: *the multi-OD coordination register is the CLOSING
> EXPERIMENT of the thesis, presented as the final cell of the boundary map, never as a rescued
> trophy.* You (the implementer) share the standard HANDOVER context; what you have NOT seen are
> the two probes committed 2026-07-18 (`738ddd1`, `e6c29e2`) whose numbers this brief rests on.
> **Reproduce them first** (oracle-only, minutes):
> `PYTHONPATH=. .venv/bin/python scratch/b4_joint_napkin_probe.py` and
> `... scratch/b4_widen_probe.py`. Do not build anything until their numbers match this file.
>
> **Authority and gates (Kilian's standing rules, unchanged):** full autonomy for the build, the
> oracle screen and the ledger; suite green after any `src/`/`scripts/` change with raw output
> pasted; commit artefacts in the session that produces them; numbers live only in the ledger.
> **PAUSE before any training run**: present the screen verdict, the pinned bars and the compute
> envelope, and launch only on Kilian's explicit in-conversation go.

---

## 0. Why this act exists (the data, since you did not run the probes)

Every register of this project so far has died the same death: a short rule matched or beat the
trained policy once the baseline family was complete (the disjoint-route finding on roads,
CRITIQUE_16-07-26 §1; on the aerial branch, the critic's 2026-07-18 measurements: the banked
Tier-1 "win" 0.742 loses to the act's own tabular-FP row 0.555 and to a best-5-route stack
0.599, and the theatre's advertised 1.63x naive gap deflates to 1.26x against a payoff-blind
separation rule). The root cause is structural: in every single-destination game, the optimal
defence is a small object (a short list of separated routes with sensible weights), so a napkin
rule sits essentially ON the optimum and learning can at best tie it.

The multi-OD game is the one register where that ceiling provably lifts, and it has now been
measured against the most hostile baseline family constructed in this project:

1. **`scratch/b4_joint_napkin_probe.py`** (committed `738ddd1`; 15 corridor-sharing triples
   s -> t1, t2 on Kaliningrad, mission objective, K=1): the joint equilibrium sits below the
   best INDEPENDENT product by median **14.4%** (reproduces the banked B4 row); a payoff-blind
   COORDINATED napkin rule (uniform over all deconflicted route pairs) is WORSE than
   independence on 15/15 triples; and even the **in-sample cap** (the best uniform mixture over
   <= 4 joint plans picked WITH full payoff knowledge, i.e. a rule allowed to cheat) leaves a
   median **11.7%** gap. No small hand-built lottery closes it. This is the first register
   where the hostile complete-baseline screen ran BEFORE any trainer existed and the gap
   survived; on the aerial branch the identical probe methodology killed the banked claim.
2. **`scratch/b4_widen_probe.py`** (committed `e6c29e2`): the gap's ceiling by axis:
   - **Screening** (the standing heuristic-gap dogma): the best sampled F=2 cells sit at
     **46%** (`147->212,188`) and **32%** (`119->62,278`) above the in-sample cap.
   - **K=2 COMPRESSES** (46 -> 25%, 32 -> 17%): attacker budget saturates everyone equally,
     the aerial "firepower compresses, structure widens" law replicating on roads. **K stays 1.**
   - **A THIRD stream (F=3, K=1) is the widener:** six cells at 25-55% above the cap (median
     ~37%; best `147->212,188,195`: joint eq 0.205 vs cap 0.317 = **55%**, and 57% above any
     payoff-blind rule), with the exact joint LP still solving in seconds.

**The class separation, in one sentence:** independent per-stream rules cannot express
correlation at all; small hand-built correlated lotteries cannot cover the web of jointly
balanced combinations three streams need; the optimal defence is a calibrated correlated
mixture over many joint plans, which a policy emitting sequential joint decisions CAN express.
That empty space above every rule (~37-55% on screened cells) is the entire prize; no previous
register had any.

**Framing (binding, Kilian's decision):** this is the closing cell of the thesis's boundary map
("where does learning pay in contested routing?"). It also moves the game toward the
multi-destination VRP the March title promised (the standing title-versus-delivery concession).
Every fail branch is writable: if training cannot capture the gap, that is the final measured
boundary and the map framing survives. No wording anywhere may say "SACRED is superior"
unconditionally: a solver still computes any single fully-specified instance (the tabular-FP
row will tie the equilibrium, as it does everywhere; report it ungated with the standing
"best-response-oracle methods" wording); the unique claim lives in the zero-shot generalist
form and in the coordination mechanism row.

---

## 1. The game (pinned; deviate only with a recorded reason)

- **Three supply streams** from one base s to three destinations t1, t2, t3 whose candidate
  route sets share corridor edges (Kaliningrad graph; menus via the existing `build_route_set`,
  k_extra 8, band (0.15, 0.95), the standing length-band vulnerability).
- **One convoy per stream per sortie; K=1 interdictor** committing one edge from the UNION of
  the three candidate edge sets (hidden, pre-committed). Soft interception via the standing
  band; **mission objective P(>= 1 of the 3 lost)** (the additive objective provably has ZERO
  correlation gap: value linear in the marginals; the loss-averse coupling is load-bearing,
  the B3 law extended, stated in the ledger).
- **Defender = ONE policy routing the streams SEQUENTIALLY** (stream 1, then 2 observing 1's
  committed route, then 3 observing both): coordination lives inside one policy's sequential
  joint action. This is the pattern that trains (multiconvoy/R3 design) and it avoids the
  gen18 boundary (independent learners never experience rare joint behaviour; here the
  conditioning is architectural, not coincidental).
- **Adversary during training = per-instance smooth FP** (`fp_dynamics.py` verbatim: softmax BR
  to the trailing joint-play window, fresh sample per sortie, tau 0.05, window 250).
- **Estimator = EXACT joint distribution by conditional enumeration** (dist1: 1 forward; dist2
  given each r1: R1 forwards; dist3 given each (r1, r2): R1*R2 forwards; ~200-400 forwards per
  instance per eval at R <= 14 — cheap; no Monte Carlo anywhere). TAP over the last 3 evals;
  best-checkpoint + validation-set selection (the gen24/v3.1 precedent); drift disclosed.

---

## 2. Baseline family (PRE-REGISTERED IN THE LEDGER BEFORE THE SCREEN RUNS; this exact list)

Every screen cell and every ladder carries ALL of, scored under the same oracle BR:

1. best deterministic joint plan (= loss_det, the certificate for every deterministic planner);
2. **best INDEPENDENT product** (alternating per-stream LPs, >= 4 restarts; disclosed as an
   upper bound on the independent class) — the B4 row;
3. per-stream disjoint-stack products (each stream plays its max-flow heuristic independently)
   — the R0a heuristic, composed;
4. **deconflict-uniform** (payoff-blind: uniform over minimum-overlap joint plans);
5. **the in-sample m-pairing caps, m <= 4** (best uniform mixture over m oracle-picked joint
   plans; the machinery is in `b4_joint_napkin_probe.best_m_pairings`) — the hardest row; the
   screen's aiming metric is **gap vs THIS cap**, not vs independence and never det/eq;
6. tabular smooth FP with the same joint BR oracle (expected to tie eq; ungated; wording rule);
7. equilibrium (exact joint LP; per-seed refs at pool build, the LP-degeneracy wobble dogma).

The wording rule from every prior act binds: no comparative sentence survives unless it clears
whichever of these rows the results actually clear.

---

## 3. Build plan (in order; the screen comes before any trainer code)

**Step 0 — reproduce the probes** (§0). Then open `experiments/gen29_multiod.md` with the §2
baseline family and DRAFT bars (§4) before anything else runs.

**Step 1 — the R0-style screen (oracle-only, free).** Extend `b4_widen_probe.py` into a proper
screen: sample ~40-60 valid (s, t1, t2, t3) cells (route sets 4-14 per stream, pairwise
corridor sharing, joint LP within the RAM guard), compute every §2 row, and shortlist by
**gap-vs-cap** subject to non-degeneracy (eq value in (0.05, 0.9); non-uniform joint equilibrium
so smooth FP has a gradient — the F1 flat-landscape killer). Deliverables: the prevalence
figure (gap-vs-cap distribution over the population, headline cells marked — the anti-cherry-
pick exhibit), the headline cell (current candidate `147->212,188,195` at 55%; verify it and
find its peers), a train pool (~15-20 cells), 6 gated held-out cells, 4 validation cells.
Known-good starting triples: `147->212,188,{195,115,127}`, `119->62,278,{181,0,59}`.

**Step 2 — env (new file, additive; nothing existing changes).**
`src/envs/multiod_interdiction.py`: sequential three-stream routing presenting the standard
observation/menu contract (`menu_route_node_idx` per active stream in featurize_state's sorted
row order — the node-ordering contract test is mandatory; `taken_node_frac` from earlier
streams' committed routes; zero-padded node ids). Interdictor commits an edge index from the
union candidate list; analytic mission reward. **Head features (the railroading lessons,
three occurrences on aerial): NO cost channel at the head.** Per-route head columns =
[worst-vulnerability, **overlap-with-committed** (edge-share fraction with earlier streams'
routes this sortie)] — the second column is the coordination signal and the road dogma applies:
it must reach the head UNDILUTED and the critic must value it (follow_w-pattern param groups
with the dedicated head-term lr 3e-2/1e-2). All three streams mix: equal entropy targets
(ent-frac 0.5, alpha floor 0.20), NO leader/follower split (every stream must hedge; there is
no structural copying here — different destinations).

**Step 3 — trainer.** `scripts/train_multiod_generalist.py`, the gen16/v3.1 recipe: pool from
the screen, per-instance smooth FP, per-instance menus riding transitions, exact conditional-
enumeration evaluator, validation-set checkpoint selection (select-on-train and select-on-test
dual-reported), per-eval checkpoints, thread caps (`OMP_NUM_THREADS=1` etc. + torch caps,
`nice`) on any multi-process launch.

**Step 4 — tests (suite green, raw output pasted):** joint payoff vs brute force on a tiny
synthetic instance; env fidelity gate (Monte-Carlo reproduces the oracle's loss_det and
loss_mixed, the G-M1 pattern); exact-estimator vs sampled-rollout agreement; node-ordering
contract; deterministic pool build across seeds; blinded-mode flag byte-identical otherwise.

**Step 5 — timing probe + 240-sortie smoke (plumbing gate, not performance), then PAUSE:**
present screen verdict + bars + envelope to Kilian; the batch launches only on his go.
Expected envelope for planning: ~3 forwards+update per sortie, pool build minutes (joint LPs),
12,000-16,000 sorties/seed, 3 seeds at 3-parallel = an M4 evening; refine from the probe.

---

## 4. Bars (DRAFT here; PINNED in the ledger after the screen names the cells, before the trainer)

- **Tier 1 (headline cell, in-pool):** best-checkpoint exact joint TAP **< that cell's
  in-sample cap row** (the ~0.317-class number, NOT the weaker independent row) on >= 2/3 seeds
  AND pooled. STRONG: <= halfway cap -> eq. Tabular-FP row beside it, ungated. With a ~55%
  ceiling, clearing this requires genuine calibration, not spreading: the untrained-context row
  (probe the random-init policy on every cell, the aerial lesson) is mandatory in the ledger.
- **Tier 2 (THE ACT'S PRIMARY; the supremacy form): zero-shot.** On the 6 gated held-out cells,
  the validation-selected checkpoint beats **each cell's in-sample cap row** on >= 4/6 AND
  pooled, on >= 2/3 seeds. STRONG: pooled ratio-to-eq <= halfway from the cells' mean cap ratio
  to 1.0. (Beating an oracle-fitted rule zero-shot, on instances never seen, is the sentence no
  prior register could even attempt; if only the independent/napkin rows are cleared, the act
  re-scopes to that honestly.)
- **MECHANISM/CAUSAL CONTROL (mandatory, the gen27/no-window pattern):** a blinded arm
  (earlier-stream observation zeroed: streams route independently within one net) must land
  ~the best independent product, NOT near the sighted policy. The gain sighted-vs-blinded is
  then CAUSALLY the coordination channel — this row is what makes the act science rather than
  a score.
- **Reported rows (ungated, standing):** worst-case premium; fleet-cost column; per-seed refs;
  final-iterate drift; per-cell results with no averaging-away of the hard cell.
- **Fail branches (pre-written, all writable):** Tier-2 partial = the transfer boundary of
  coordination, measured; total fail = "the correlation gap exists (oracle-proven, 37-55%) but
  model-free self-play at thesis scale cannot capture it" — the final boundary-map cell, still
  a result under the standing framing.

---

## 5. Known risks and their standing answers

- **gen18 (learned coordination fails):** answered architecturally (sequential conditioning in
  one policy, not independent learners) + the undiluted overlap head term + the blinded control
  to prove the channel carries. If the sighted arm ties the blinded arm, that IS the gen18
  boundary replicating in the last register — report it, do not chase past the pre-committed
  attempt count (one re-aim maximum, then close).
- **Railroading (three aerial occurrences):** no reward-irrelevant channels at the head;
  watch rw magnitudes in the eval prints; validation selection.
- **Saturating bandit:** not expected (3 decisions/sortie x ~20-instance pool = real state
  diversity), but the diagnosis method is on record if train curves stay flat at init level.
- **Baseline recursion:** if any new naive rule occurs to you mid-act (it happened on every
  prior act), measure it at the oracle level IMMEDIATELY and fold it into the family before
  results are read; bars are never moved after launch.
- **Do not touch** the aerial branch, the live B2 conversation's files, or any banked ledger;
  all gen29 code is additive, on a fresh worktree branch `gen29-multiod` off `e6c29e2`.

---

## 6. Definition of done

Branch `gen29-multiod` with: the screen + prevalence figure + shortlist (committed with the
ledger BEFORE any trainer); env + trainer + tests (suite green, raw output pasted); the smoke +
envelope presented to Kilian at the PAUSE gate; after his go, the 3-seed batch + blinded
control, results appended to `experiments/gen29_multiod.md` with every §2 row; chronicle entry;
HANDOVER banner refresh; everything committed. The thesis sentence this act is allowed to earn
if Tier 2 lands: *in the one contested-routing register where no simple rule, however assisted,
can express the optimal defence (measured: 37-55% above every hand-built lottery on screened
cells), one policy learns coordinated fleet plans and carries them zero-shot to instances it
has never seen — and a blinded control shows the coordination channel is causally the
mechanism.* If it does not land, the boundary map closes with a measured edge instead — and
the thesis is finished either way.
