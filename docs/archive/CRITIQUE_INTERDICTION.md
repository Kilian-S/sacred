# CRITIQUE_INTERDICTION.md: fresh-eyes critique of the interdiction programme (Fable, 2026-07-09)

> Requested by Kilian 2026-07-09: a holistic, examiner-grade critique of the interdiction/security-game
> arc (gen08/gen09), its fit against the five research objectives of the assessed literature review,
> a codebase audit, and an informed outlook on SBO, ZST and scaling. Sources: every file in the
> HANDOVER read order (all docs, all ledgers gen01-gen09, both PDFs, the core code), the full test
> suite re-run (**146 passed**, matching the record), two empirical probes run this session
> (node-ordering demonstration; `scratch/gen09_exact_reeval.py`, eval-only, no training). No `src/`
> code was changed. Companion to `CRITIQUE.md` (2026-07-02), which reframed the campaign; this file
> critiques what the reframe became.

---

## 0. Verdict in five sentences

The interdiction pivot is intellectually sound, unusually well documented, and the two banked
headlines are real, pre-registered results scored against a computable ground truth; by MSc
standards the methodology is well above the bar. But the thesis is currently exposed on four
fronts: (1) a **project-wide representation bug** (the policy head indexes node embeddings with a
different node ordering than the featuriser, so every trained network in the project has been
reading a fixed permutation of the wrong nodes' embeddings) that does not invalidate the banked
comparisons but does undercut the GNN methods narrative, weakens the gen03/04 learned-adversary
mechanism story, and makes any ZST claim impossible until fixed; (2) the **"why not just solve the
LP" question is not yet answered by a trained result**: as implemented, SACRED trains against an
oracle best response that itself materialises the full objective matrix, so the current code
inherits the very combinatorial wall the scaling figure holds against the oracle; (3) two of the
five objectives (Obj-4 SBO entirely, Obj-5's "varied disruption" clause) plus the aim-level ZST
promise are **not yet evidenced by any run**; and (4) the locked multi-convoy number carries a
small but real estimator bias: re-evaluating the saved checkpoints with the exact (not Monte-Carlo)
occupancy distribution gives **best-checkpoint TAP 0.295 +/- 0.024, not 0.283 +/- 0.021** (same
qualitative ladder, but the citable number should be the exact one). All four are addressable
before the Aug 3 freeze, and the ranked programme in §8 fits the remaining runway.

---

## 1. Fit against the five research objectives (verbatim, from lit review §2.2)

The guidance PDF requires the conclusions chapter to revisit each objective and confirm whether it
was met. Here is the honest scoring as of today.

**Obj 1: "Formulate the SDVRP as an asymmetric zero-sum Markov game, defining discrete action
spaces for a protagonist dispatcher and an environment-altering antagonist agent."**
Met, with two deltas that must be named rather than papered over. (a) What was promised is a
simultaneous-move Markov game in the RARL lineage; what the positive results are built on is a
**Stackelberg security game with hidden pre-commitment**. That is a strengthening (it is the game
class where the thesis mechanism provably matters), but it is a different solution concept and the
thesis needs a sentence saying the change was deliberate and why. (b) The "environment-altering
antagonist **agent**" in both banked headlines is the **oracle best response**, not a learned
agent. The learned-antagonist co-evolution demonstration (ROADMAP F2 / I3.4) never ran. The
campaign (gen03-06) did field learned antagonists and is honest about their failure; but an
examiner reading Obj-1 plus Obj-3's "policy coevolution" will notice that the positive act
contains no two-sided learning. Either run the one-instance F2 demo or scope the claim explicitly
("the oracle interdictor is the strongest possible adversary; a learned interdictor is bounded by
it"). Also note the quiet problem-class slide: the headline game has no stochastic demand, no
capacity, no dynamism within a sortie; the "SDVRP" of the title lives in Acts I-II, and the
security game is a repeated matrix game on a real road graph. Frame this as deliberate refinement
(the campaign showed the congestion SDVRP is the wrong game for adversarial RL), not as scope creep.

**Obj 2: "Design and implement a visual, interactive multi-agent simulation environment..."**
Met. Environment rungs behind one CLI, the interdiction and multi-convoy layers, fidelity gates
against the oracle (G1, G3, G-M1), the PyGame visualiser and the interactive Kaliningrad view,
146 green tests. Uncontroversial; the fidelity-gate discipline is worth a methods paragraph.

**Obj 3: "Develop the SACRED framework, utilising a SAC architecture and ATLA, and investigate the
efficacy of ERB bootstrapping via population-based metaheuristics to accelerate training
convergence and ensure policy coevolution."**
Substantially met, with re-interpretations to declare. SAC's max-entropy machinery is genuinely
load-bearing (the entropy IS the mixed strategy: the thesis's most elegant point). "ATLA" in the
positive arc is realised as **fictitious play against an oracle best response**, and the FP
dynamics bracket (pure BR cycles / stale mixture parks / smooth FP converges; then the gen09
transient/best-checkpoint finding) is real, citable material that maps directly onto the review's
own fictitious-self-play paragraph (Heinrich & Silver). ERB bootstrapping is realised as
**demonstration bootstrapping** (forced-copy warmup against a frozen mixing leader + prioritised
replay of rare stacked transitions), which is a legitimate ERB result against a diagnosed
pathology, but it is not "population-based metaheuristic data" (the gen01 ALNS-demos pilot stayed
inconclusive at n=1). Say so; do not lean on "double-oracle is literally a population method"
unless double-oracle is actually run.

**Obj 4: "Incorporate SBO into the SACRED framework, utilising a neural network metamodel to
approximate facility location and fleet composition..."**
**Not met, and currently the biggest promise-delivery gap.** "N is a design lever" is a parameter,
not surrogate-based optimisation. Nothing metamodel-shaped has run. This is also the cheapest gap
to close: the repo already contains a dormant SBO scaffold (`src/sbo/surrogate.py`,
`src/sbo/flp_solver.py`, from the original build) and the oracle instance screen already computed
equilibrium properties over 72 OD pairs. The F3 demonstrator (placement/OD grid, oracle
equilibrium interception per candidate, small neural surrogate, argmax validated by full solve) is
an afternoon of eval-only work and converts Obj-4 from "descoped" to "demonstrated in reduced
form". Recommendation: do it. See §7.

**Obj 5: "Evaluate the performance and resilience of the SACRED framework against SOTA adaptive,
population-based metaheuristics and a baseline non-adversarially SAC-trained agent under varied
levels of network disruption."**
The two comparison clauses are met by the multi-convoy ladder (ALNS and vanilla both beaten, with
the ALNS-forced-stack fairness row a genuinely good piece of experimental design). Two exposures
remain. (a) The bespoke ~130-line ALNS over a 364-occupancy space will not survive being called
"SOTA" on its own; its real defence is much stronger and should be the framing: **it is verified
to reach `loss_det`, the optimum of the entire deterministic class**, so SACRED's margin holds
against ANY deterministic coordinator however sophisticated, ALNS is merely the certificate. Write
it that way. (b) "**Varied levels of network disruption**" has no trained evidence yet: the locked
headline is one instance, one K, one N. The single-convoy sweep (F1) was killed for good reasons;
the multi-convoy sweeps (M5: K, N, connectivity, a second OD) have not run. The oracle scan
(median gap 0.48 over 20 OD pairs) is oracle-level, not trained. One trained sweep tier is the
single most important remaining experiment for this objective. See §8.

**ZST (aim-level: "resilient, zero-shot transferable logistics policies").**
Untested, and currently **untestable**: the indexing bug (§5.1) means trained policies identify
routes through a fixed accidental permutation of node embeddings, i.e. they are instance-specific
lookup tables by construction; and the observation contains **no edge-vulnerability feature**, so
even a bug-free policy cannot in principle infer the hedge on an unseen threat map (it only ever
learns riskiness through reward on the training instance). ZST therefore requires (i) the bug fix,
and (ii) making the vulnerability map observable. Both are tractable; see §7. If not done, the
thesis must scope ZST honestly as designed-but-descoped with the mechanism argument ("the
mixed-strategy concept is graph-agnostic") plus the §7 architecture as future work.

---

## 2. Is the finding trivial? (the examiner's strongest attack, sharpened)

The storyline bank already anticipates "isn't this a game-theory tautology?" and answers that the
contribution is that deep RL *learns* toward the equilibrium from experience. That answer is
necessary but not yet sufficient, because of the following chain an examiner can assemble from the
ledgers alone:

1. The headline game has 12 routes and 364 occupancies; the exact equilibrium takes **1.4 s** by LP.
2. The trained artefact is (in fleet-route mode) a distribution over those 12 routes.
3. SACRED trains **against the oracle best response**, whose implementation materialises the same
   objective matrix the LP uses; the trainer touches it every attacker refresh, and the env builds
   it eagerly in its constructor. So the current training pipeline does not escape the oracle's
   combinatorics; it presupposes them.
4. The oracle-scaling probe honestly concedes the crossover is against *naive enumeration*, and the
   security-games literature (double oracle / column generation, the Tambe lineage the pivot
   itself invokes) solves far larger instances exactly.

So the "deep RL scales past the LP" argument is currently a projection, not a result, and its
strongest form (deployment cost + transfer) depends on ZST, which is untested. The finding is NOT
trivial, but its non-triviality currently rests on the parts of the story that are thinnest.
What rescues it, in order of strength:

- **The dynamics characterisation is genuinely novel material**: instance structure decides
  learnability (shared-edge asymmetry vs disjoint flatness, oracle-screened over 72 pairs); the FP
  discipline bracket; the reproducible equilibrium transient and best-checkpoint discipline; the
  chicken-and-egg of learned coordination and its critic-side resolution (`follow_w`). None of
  that falls out of the LP. It answers "when and why does adversarial deep RL find the
  equilibrium", which is a research question, not a tautology.
- **The negative campaign** (gen03-06) plus the flat-landscape diagnosis is a rigorous,
  literature-anchored contribution regardless.
- **The scaling/ZST claims need one trained datapoint each** to stop being projections (§7, §8).

Also position novelty against the actual adjacent literature: learning in security games is not
empty (PSRO/double-oracle with RL best responses; deep-learning approaches to green security
games, e.g. DeepFP-style work; RL for patrolling). The memory note is right that the assessed
review contains no Stackelberg/security-game section; the thesis MUST add that related-work
subsection, and the honest novelty claim is roughly: *max-entropy SAC as the mixed-strategy
generator for network interdiction on a real road graph, with a computable-equilibrium yardstick,
a coordination (multi-convoy, loss-averse) extension, and a measured account of the fictitious-play
dynamics*, not "first to combine RL and security games".

---

## 3. Logical and argumentative weaknesses in the storyline

1. **"Wins by construction" vs "F1 killed".** Act IV opens with "adversarial minimax training wins
   by construction" and later records that on symmetric instances adversarial training is a
   liability that destabilises the defender. These are reconcilable (the game guarantees the GAP
   exists; whether the TRAINING DYNAMICS reach it depends on the FP landscape) but the thesis must
   make that distinction explicitly, and should present instance asymmetry as a discovered
   **scope condition** (a finding) rather than an embarrassment. The oracle screen makes this easy.
2. **Best-checkpoint selection is on the reported metric.** The best checkpoint is chosen as the
   minimum of the same (noisy, 400-sample MC) estimator that is then reported. That is selection
   on noise; §6 quantifies it (exact re-evaluation: 0.295 vs the reported 0.283). Adopt the exact
   number, or at minimum disclose the estimator and its noise. Also state precisely what the
   deployable artefact is: the best "checkpoint" TAP is an **ensemble object** (the mean route
   distribution of the last five eval snapshots), i.e. a sampled route table, not a single network.
   That is operationally fine (the planner samples from a table) but must be described accurately.
3. **The objective function was selected because SACRED wins under it.** The loss-averse
   mission-failure objective is both operationally defensible and oracle-proven load-bearing
   (risk-neutral dilutes the gap to ~0), and the trail is pre-registered. Present the objective x
   interception-model x route-structure boundary map as a contribution table, and defend
   loss-aversion on operational grounds first (mission-abort criteria), with the "and this is
   where randomisation pays" as the finding. One robustness note worth adding: interceptions of
   stacked convoys are modelled as independent Bernoulli draws; under positive correlation
   (one ambush team vs one stacked column) stacking becomes even better under the mission
   objective, so independence is the conservative case. One sentence pre-empts the question.
4. **gen03/04's mechanism chain is now partly confounded** by the §5.1 bug: the learned
   antagonist's edge scoring also consumed permuted node embeddings, so "the learned adversary
   cannot learn to attack even with full observability" has an implementation-artefact channel in
   addition to the diagnosed entropy-pinning/SNR channel. The flat-landscape conclusion itself
   survives (it rests on the representation-independent facts that random blocking achieves ~96%
   of the scripted attack and that scripted >> learned), but the entropy-pinning telemetry story
   should be softened to "consistent with", and ideally one cheap post-fix BR re-gate would
   discriminate. The thesis's self-correction culture makes this survivable; hiding it would not be.
5. **"All five objectives met" overclaims** (gen09 ledger, HANDOVER banner). Obj-4 is not met by
   N-as-a-lever, and Obj-3's ERB clause is met in adapted form. The conclusions chapter (which the
   guidance PDF requires to revisit each objective) should use the §1 scoring above, deltas named.
6. **The multi-convoy cost dimension is silently dropped.** The single-convoy ledger reports
   cost(TAP) beside exploitability (the frontier framing); the multi-convoy headline reward is
   pure mission-failure with zero travel-cost weight for both SACRED and ALNS, and no fleet travel
   cost is reported. The randomised stack plausibly pays a material detour premium; an examiner
   will ask. Reporting the best-checkpoint mixture's expected fleet cost beside ALNS's plan cost
   is a five-minute eval-only addition. Do it.
7. **Duplication drift.** The same numbers now live in eight documents (HANDOVER, README, ledger,
   REDESIGN §10, STORYLINE, ROADMAP, SYSTEM, PROGRESS). The 0.257-to-0.283 correction shows the
   cost of that. For the thesis phase, declare the ledgers the single source of numbers (the
   stated house rule) and let the prose documents carry pointers, not values; a stale 0.283 vs
   0.295 repeat of the same incident is otherwise likely.

---

## 4. Methodological oversights (beyond §3)

1. **One instance, one K, one N carries the multi-convoy claim** (see Obj-5 above): the sweep tier
   is the highest-value remaining training work.
2. **No held-out evaluation instance anywhere in the positive arc**: single-convoy B2-S was
   pre-registered and never run; the multi-convoy generalisation evidence is oracle-only. Even one
   second OD pair trained at the headline config would blunt the "you screened the instance where
   you win" reading (the screening is disclosed and principled, but a replication is stronger).
3. **n=3 seeds with population std, no CI**: acceptable at MSc scale, but say "3 seeds, mean +/-
   population std, individual seeds shown" and resist significance language; the campaign's own
   dual-reporting rule (pooled + per-seed) was better discipline than the gen09 summary applies.
4. **The vanilla control's entropy configuration** (default 0.45·ln R target) is doing quiet work:
   vanilla lands at ~0.945 partly because its incidental mixing is cost-tilted on this instance.
   The B2 instance-design argument (no cost-driven mixture can imitate the equilibrium) covers the
   single-convoy case; for multi-convoy, one sentence on why vanilla cannot stack-and-randomise
   regardless of temperature (its objective contains no adversarial signal at all) closes the hole.
   Report vanilla's best-checkpoint too, for symmetry with SACRED's selection privilege.
5. **Eval estimator inconsistency between the two headline generations**: single-convoy TAP uses
   the EXACT policy route distribution (trie product / first-hop probs); multi-convoy TAP uses a
   400-sample Monte-Carlo occupancy estimate. §6 shows the difference matters at the reported
   precision. For fleet-route mode the exact distribution is one forward pass; for the general
   sequential policy it is an N-level conditional enumeration (12 + 144 forwards for N=3, R=12),
   still cheaper than 1200 sampled rollouts. Adopt exact evaluation everywhere.
6. **`switch_every` semantics changed silently between generations** (B2-P3 refreshed attacker
   weights every 50 sorties; gen09 kept 200 with per-sortie sampling). STAB-3 records the
   hypothesis that this does not matter; it was never tested at 50. Minor, but it is the one
   untested knob in the "mechanism port" story (path A in the ledger, closed by decision).

---

## 5. Codebase audit: bugs

### 5.1 CRITICAL: node-ordering mismatch between the featuriser and every consumer

`featurize_state` (`src/agents/networks.py:82`) builds the node-feature matrix over
**`sorted(nodes_dict.keys())`**. Every consumer computes indices into that matrix from
**dict insertion order**: `ProtagonistSAC.select_action` (`sac.py:342`), both passes of
`ProtagonistSAC.update` (`sac.py:414/430`), `AntagonistSAC.select_action` (`sac.py:910`) and its
update path, `hop_probs` in both trainers, and `menu_route_node_idx`
(`src/envs/multiconvoy_interdiction.py:158`). On the Kaliningrad graph insertion order is not
sorted order. Demonstrated live this session (62->97 headline instance):

```
active node: 62
insertion-order index used by select_action: 76
row 76 of the featurized (sorted) matrix is node: 167
featurizer marks is_active_here=1 at row 249 (node 62)
=> the policy head reads h[76] (node 167) instead of h[249] (node 62)
```

The bug is present since the initial commit (`git log -S`), so **every trained network in the
project** (campaign and interdiction arcs, protagonist and antagonist) has been scoring candidates
with the embeddings of a fixed permutation of the wrong nodes; in menu-select mode the per-route
mean-pooled "route embeddings" pool the wrong node sets. The 146-test suite is self-consistent
with the same convention on both sides of each test, which is why it never fired.

**What it does and does not invalidate.** The permutation is fixed and bijective, identical in
training and evaluation, and applied equally to every arm; so all banked comparisons remain
internally valid, the oracle/ALNS/equilibrium numbers are untouched (pure LP), and the RL results
stand as achieved-under-handicap. What it degrades: (a) the GNN's spatial inductive bias is
effectively destroyed (the head reads structurally unrelated embeddings, distinguishable only as
fixed identifiers), so the methods chapter's "graph-structured policy" narrative is inaccurate
as-is; (b) it plausibly contributes to the residual distance-to-equilibrium in both headlines, to
the learned-follower saturation (notably, the two fixes that finally moved coordination, the
`taken` term and `follow_w`, are precisely the ones that BYPASS the GNN pathway and deliver the
signal straight to the head: consistent with the pathway being scrambled), and to the gen03/04
learned-attacker failures (§3.4); (c) it makes ZST structurally impossible (the learned "route
identities" are accidents of one instance's insertion order).

**Fix (one line plus regression tests):** make the consumers use the featuriser's ordering, e.g.
`node_ids = sorted(observation["nodes"].keys())` at the four construction sites (and
`menu_route_node_idx`); add a test asserting `select_action`'s active index matches the row where
`featurize_state` set `is_active_here`. Old checkpoints memorised the permuted readout, so they
remain evaluable only at their pinned SHAs (already the ledger rule: never compare across git
states). New training after the fix is a new generation. **Recommendation: fix now, then re-run
the two headline configs (3 seeds each, roughly 1-2 h total at 3-parallel) as gen10 with the same
pre-registered metrics.** If the numbers improve toward the equilibrium, the thesis strengthens
and the bug becomes a methods finding; if unchanged, the banked numbers stand and the fix is a
disclosed correction. Either outcome is safe; not knowing is not.

### 5.2 Role-alpha inconsistency in the Bellman target

`sac.py:505`: the soft state value `v_next` always uses `self.alpha` (the primary/leader
temperature), including when the next decision belongs to the follower group whose actor loss uses
`alpha_foll`. With leader alpha ~0.4 and follower alpha ~0, follower-successor targets carry an
inflated entropy bonus. Affects only the learned-follower arc (role_alpha mode), one plausible
small contributor to its saturation. Fix: select the temperature by the NEXT state's
`alpha_group`, as the actor loss already does by the current state's.

### 5.3 Estimator noise and selection bias in the locked headline (quantified, resolved)

`policy_occ_dist` estimates the occupancy distribution from 400 sampled rollouts; the
best-checkpoint is the minimum over 12 evals of that noisy reading (§3.2, §4.5). Exact
re-evaluation of the saved gen09-HEADLINE per-eval checkpoints (`scratch/gen09_exact_reeval.py`,
eval-only: one forward pass per checkpoint, fleet-route occupancy = stacked leader distribution):

| seed | exact best-ckpt TAP @ sortie | ledger MC best TAP |
|---|---|---|
| 0 | 0.281 @ 500 | 0.281 @ 500 |
| 1 | 0.274 @ 400 | 0.260 @ 400 |
| 2 | 0.329 @ 500 | 0.310 @ 500 |

**Exact best-checkpoint TAP = 0.295 +/- 0.024** (ledger MC: 0.283 +/- 0.021). Same best sorties,
same qualitative ladder (0.295 << ALNS 0.699, 1.37x the equilibrium 0.216), but the MC minimum was
optimistic by ~0.012, i.e. the selection-on-noise bias is the same order as the reported seed
spread. **Recommendation: cite the exact 0.295 +/- 0.024 (or re-lock on the exact estimator) and
switch multi-convoy evaluation to exact distributions** (fleet-route: one forward; general
sequential policy: conditional enumeration, 157 forwards for N=3 R=12, cheaper than the current
1200 sampled rollouts per eval).

### 5.4 Scaling wall inside the trainer, not just the oracle

`MultiConvoyInterdictionEnv.__init__` eagerly materialises the full `[#occ x #iset]` objective
matrix, and the trainer's attacker (both `latest` and `smooth`) does full matrix-vector products
against it every refresh. Consequently the advertised "SACRED is flat in the axes that blow up the
oracle" is not true of the current implementation: at N=3, K=3 (the proposed scaled run) the env
constructor itself builds the 28.8M-entry matrix. See §7 (scaling) for the fix that makes the
scaled run honest.

### 5.5 Minor

- `save_checkpoint`/`load_checkpoint` omit `log_alpha_foll`, its optimiser, and `follow_w`
  optimiser state: resuming a role-alpha run silently resets them (not currently exercised).
- `train_multiconvoy.py:374`: vanilla reference hardcoded to 0.945 when `--skip-vanilla` (disclosed
  in-line; keep it out of any citable table).
- Stale comment `sac.py`-side: `played` histogram is labelled "latest-BR + eval" but eval rollouts
  never touch it (correct behaviour, wrong comment).
- `featurize_state` rebuilds the per-node Python feature list every call (~276 rows); vectorising
  with numpy would shave the observe/act path, though `update()` dominates runtime.
- `AntagonistPolicyValueNet.head` rebuilds `edge_features_dict` (zip over all directed edges) per
  call: O(E) Python-dict churn on the campaign path.
- In menu-select mode truck positions never move, so the older docs' "the env exposes earlier
  convoys' routes via truck positions" is stale for the headline path: the signal is carried
  solely by `routed_convoys`/`taken_node_frac` (fine, but the methods text must match).

---

## 6. What the audit does NOT find

For balance: the SAC core is correct on the things that usually go wrong (alpha-loss sign fixed
and regression-tested, twin critics with detached targets, grad clipping, SMDP gamma^dt, batched
encoder verified equivalent); the oracle LPs are clean and closed-form-verified in tests; the env
fidelity gates (Monte-Carlo reproduction of loss_det/loss_mixed) are exactly the right kind of
plumbing test; the walk-trie branch-product distribution is exact as claimed; the ALNS is verified
against `loss_det` (which is the only property the argument needs); the smooth-FP helper is a
faithful single source of truth; and the pre-registration/ledger/pinned-SHA discipline is the best
I have seen at this level. The retraction culture (static-3b, the 0.257 transient) is a strength
the thesis should own openly.

---

## 7. Outlook: SBO, ZST, scaling (the requested informed view)

**SBO (Obj-4).** Highest value per hour in the whole remaining programme, because the oracle makes
the expensive evaluation cheap and the repo already has the scaffold (`src/sbo/`). Concrete
reduced-form demonstrator, eval-only, one afternoon: (i) design space = candidate base/FOB
placements (or OD pairs) x fleet size N; (ii) exact objective per design = equilibrium
mission-failure `loss_mixed` (and/or the trained-policy exploitability) from the oracle, seconds
each at K=1, and the 72-pair instance screen already computed half of it; (iii) fit the existing
`SurrogateMLP` on a subsample of designs (features: OD distance, route count, shared-edge mass,
leader entropy H/lnR, vulnerability stats); (iv) validate the surrogate's argmax design by full
solve, report rank correlation + regret of the argmax. That is a genuine "neural metamodel
approximating facility location / fleet composition" in the promised sense, scoped honestly
("interdiction-aware placement, reduced form"). The full SBO loop (acquisition, refinement,
coupling into training) stays future work. Without this, Obj-4 must be written as descoped with
supervisor agreement; with it, all five objectives have at least a demonstrated form.

**ZST.** Currently blocked twice over: the §5.1 bug makes trained policies instance-memorised, and
the observation carries **no edge-vulnerability feature**, so no policy can infer a hedge on an
unseen threat map even in principle (riskiness is only ever learned through reward on the training
instance). The honest ZST ladder, in increasing ambition: (0) after the bug fix, transfer the
retrained B2-P3/gen10 policy to one held-out OD and report the ladder there (the F4 afternoon;
expect partial transfer at best, and say why); (1) add edge vulnerability as an edge-feature
column (one column, back-compatible via the established width-slicing) and train ONE policy across
sampled OD pairs / vulnerability maps (menu-select is naturally suited: route scores are pooled
node embeddings, an instance-agnostic interface), then evaluate zero-shot on held-out pairs
against each pair's own oracle equilibrium: that is the crown-jewel experiment the aim promised,
and it is the point where the GNN actually earns its place (post-fix, the embeddings finally mean
something). Budget ~2-4 days including a pre-registered gate; risk moderate (multi-instance
training may need more sorties); payoff: the aim's headline sentence stops being future work.
If the runway does not allow (1), do (0) and write (1) as the designed next experiment. I
recommend attempting (1) only after the sweeps (§8) are banked.

**Scaling.** The current figure (naive LP wall at K>=3 vs linear SACRED) is honest about being
naive-enumeration-relative, but as argued in §2 and §5.4 it is doubly soft: double oracle solves
much larger games exactly, and current SACRED training itself materialises the objective matrix.
The fix that makes the scaled run meaningful: a **matrix-free best-response attacker**. For the
mission objective, a route's interception under an interdiction set is 1 minus a product of
per-edge survivals, so expected mission failure of a defender mixture is monotone and submodular
in the interdicted edge set; the greedy K-edge best response therefore carries the classic
(1 - 1/e) guarantee, costs O(E·K·S·N) per refresh (S = support of the empirical play, tiny), needs
no objective matrix at all, and can be spot-checked against the exact BR at K<=2. With that
attacker: (i) the env no longer builds `obj_matrix` eagerly (evaluation-only, on demand); (ii) the
N=3, K=3 scaled run becomes a genuine "trained where the naive oracle is infeasible" datapoint,
with the honest caveat that the yardstick at that size is the greedy BR (report the guarantee);
(iii) the thesis's scaling paragraph can then say something defensible: exact solvers win the
small-instance benchmark outright; the RL policy's case is amortisation (deployment forward pass,
transfer across instances) plus attacker-side approximability, and here is one trained point past
the naive wall. I would NOT claim more than that: against a competent column-generation solver the
LP remains formidable, and pretending otherwise is the one place an operations-research examiner
could turn hostile. The strongest scaling claim available to this thesis runs through ZST, not
through wall-clock.

---

## 8. Recommended programme to the Aug 3 freeze (firm, ranked; every launch remains Kilian's go)

1. **Fix §5.1 + §5.2, re-run both headline configs as gen10** (3 seeds each; ~1-2 h CPU total;
   pre-register "same metrics, post-fix"). Protects the entire empirical base and unblocks
   everything below. Do first.
2. **Adopt exact evaluation and the exact headline number** (0.295 +/- 0.024 for the current lock,
   or the gen10 exact value once it exists); add the fleet-cost column (§3.6). Eval-only, hours.
3. **F3 SBO demonstrator** (Obj-4, §7). Eval-only, one afternoon. Converts the weakest objective.
4. **M5 sweep tier on the post-fix code**: K in {1,2,3} x N in {2,3,5} at the headline OD plus one
   second screened OD, 3 seeds on the headline cell, 1 seed elsewhere, reported as curves against
   each cell's oracle ladder (~an evening at 3-parallel). Closes Obj-5's "varied disruption"
   clause and the single-instance exposure in one stroke.
5. **ZST step (0)** (held-out OD transfer of the gen10 single-convoy policy; F4 afternoon), then,
   only if the calendar still breathes, **ZST step (1)** (vulnerability-observable multi-instance
   training, §7).
6. **Optional, if 1-5 land early**: the matrix-free greedy-BR attacker + the N=3 K=3 scaled run
   (§7 scaling); and/or the one-instance learned-antagonist co-evolution demo (F2), which is worth
   exactly one post-fix attempt given §3.4.
Drop-first order if the calendar bites: 6, then ZST(1), then the second OD of 4. Items 1-3 should
not be dropped under any schedule; they are cheap and they are what an examiner will probe.

---

*Artefacts of this critique: this file; `scratch/gen09_exact_reeval.py` (+ output quoted in §5.3);
the node-ordering demonstration (§5.1, reproducible one-liner against the headline env). Suite
state verified this session: 146 passed. No `src/` changes made; no training launched.*
