# CRITIQUE_PREFREEZE.md: fresh-eyes critique of the whole programme, post-gen10 (Fable, 2026-07-10)

> Requested by Kilian 2026-07-10: a holistic, examiner-grade critique of the thesis as it stands
> after the gen10 post-fix arc, its fit against the five research objectives (read verbatim from
> the assessed literature review), a fresh codebase audit (bugs missed by the 2026-07-09 audit,
> plus performance work relevant to scaling), and an informed outlook on SBO, ZST and scaling,
> with a ranked programme for the remaining runway. Sources: the complete HANDOVER read order
> (all docs, all ledgers gen01-gen10, both PDFs, all core code listed in HANDOVER §2.8 plus the
> interdiction/multiconvoy stack), suite re-run (**149 passed**, matching the record), and three
> cheap probes run this session (route-overlap probe on 62-97 k8; a shared-dict staleness
> demonstration; the gen10_mc2.sh diff check). No `src/` change made; no training launched.
> Companion to `CRITIQUE.md` (2026-07-02, reframed the campaign) and `CRITIQUE_INTERDICTION.md`
> (2026-07-09, found the node-ordering bug). This file critiques what the fix did to the story.

---

## 0. Verdict in six sentences

The programme remains, by MSc standards, exceptionally well run: pre-registration, pinned SHAs,
fidelity gates, disclosed retractions, and now a bug-fix arc handled with the same discipline.
The single-convoy headline is in its strongest state ever (gen10-SC 0.276 on corrected
representations, every clause passed, supersession confirmed). But gen10 has created the thesis's
sharpest structural problem: **the two headlines now live on opposite sides of the representation
fix**, and the multi-convoy citable number (exact 0.295 at SHA `ad70a9c`) was produced by networks
whose "route embeddings" were an accidental identity hash, which means it evidences that
adversarial fictitious play can find a near-minimax mixture over twelve abstract arms, not that a
graph-structured policy learned anything about the map; the honest post-fix reproduction (0.447,
twice, tightly) says the current architecture cannot yet do the latter. Two of the five
objectives (Obj-4 entirely; Obj-5's "varied disruption" clause) plus the aim-level ZST promise
still have no trained evidence, and the Final Activities Report deadline (30 July) means the real
experimental runway is about 2.5 weeks, not the 3.5 the Aug-3 freeze suggests. The audit found
one new latent defect in the campaign-era path (buffered observations mutate in place; another
disclosed confound for the gen03-06 mechanism story, not for the banked comparisons), one
untested mechanism candidate for the gen10-MC regression (the fleet-route trainer pushes follower
transitions with near-zero entropy targets into the same actor that must keep the leader mixed),
and the known scaling wall (eager objective matrix) still unimplemented-around. The ranked
programme in §8 fits the runway and, if item 2 (gen11) lands, resolves the structural problem.

---

## 1. Fit against the five research objectives (verbatim, lit review §2.2), post-gen10

The guidance PDF requires the conclusions chapter to revisit each objective and confirm whether
it was met; Methods/Analysis/Discussion carries 50% of the thesis mark. Scoring as of today:

**Obj 1: "Formulate the SDVRP as an asymmetric zero-sum Markov game, defining discrete action
spaces for a protagonist dispatcher and an environment-altering antagonist agent."**
Met with declared deltas, unchanged from the 2026-07-09 scoring: (a) the positive results live in
a Stackelberg security game with hidden pre-commitment, not the simultaneous-move RARL game the
review promised (a strengthening, but say so); (b) the "antagonist agent" in every banked
positive result is the oracle best response, not a learned agent (the F2 one-instance
co-evolution demo still has not run); (c) the headline game has no stochastic demand, no
capacity, no within-sortie dynamism: the S and the D of "SDVRP" live in Acts I-II. All three
deltas are presentable as deliberate refinement backed by the negative campaign; none is
presentable by silence.

**Obj 2: "Design and implement a visual, interactive multi-agent simulation environment..."**
Met. Suite 149, fidelity gates (G1/G3/G-M1), the visualisers, the interactive Kaliningrad view.

**Obj 3: "Develop the SACRED framework, utilising a SAC architecture and ATLA, and investigate
the efficacy of ERB bootstrapping via population-based metaheuristics..."**
Substantially met, with the same re-interpretations to declare: ATLA realised as fictitious play
against an oracle BR; ERB realised as demonstration bootstrapping (forced-copy warmup + prioritised
stack replay), not population-based-metaheuristic data. The gen10 arc *adds* Obj-3 material: the
role-alpha Bellman-target correction and the finding that a representation defect can flatter
learning are both methods-chapter content. The `follow_w` arc remains the strongest Obj-3 asset.

**Obj 4: "Incorporate SBO into the SACRED framework, utilising a neural network metamodel to
approximate facility location and fleet composition..."**
**Still not met, and still the cheapest gap to close.** Unchanged since 2026-07-09: the F3
demonstrator (placement/OD grid, oracle equilibrium objective per design, `SurrogateMLP` fit,
argmax validated by full solve) is an eval-only afternoon; `src/sbo/surrogate.py` and
`flp_solver.py` sit unused; the 72-pair instance screen already computed half the dataset. Every
week this is not done, the risk grows that the freeze arrives with Obj-4 at "descoped".

**Obj 5: "Evaluate the performance and resilience of the SACRED framework against SOTA adaptive,
population-based metaheuristics and a baseline non-adversarially SAC-trained agent under varied
levels of network disruption."**
The two comparison clauses are met (both ladders); the framing rule stands: ALNS's defence is
that it is *verified to reach loss_det, the optimum of the entire deterministic class*, so the
margin holds against any deterministic coordinator; never call the 130-line ALNS "SOTA" on its
own. **"Varied levels of network disruption" still has zero trained evidence**: one instance, one
K, one N carries the multi-convoy claim; the M5 sweep tier has not run; the oracle scan is
oracle-level. This remains the single most important outstanding experiment for the objective,
now behind gen11 in priority only because gen11 decides which code the sweeps should run on.

**ZST (aim-level: "resilient, zero-shot transferable logistics policies").**
Half-unblocked by gen10. The node-ordering fix removes the "instance-memorised by construction"
blocker for the single-convoy walk-mode policy, so ZST step 0 (transfer the gen10-SC policy to a
held-out OD, score against that OD's oracle) is now a *meaningful* afternoon rather than a
guaranteed null. The deeper blocker stands: the observation still carries **no edge-vulnerability
feature**, so no policy can infer a hedge on an unseen threat map even in principle; and on the
menu-select path the gen10-MC result shows the head cannot yet discriminate correct embeddings
even on the training instance. ZST step 1 (vulnerability-observable, multi-instance training)
therefore depends on gen11's per-route feature injection, which is exactly the same mechanism.

---

## 2. The two-headline asymmetry (the sharpest issue the thesis now has)

The examiner's strongest post-gen10 attack is no longer "is this a game-theory tautology?" but:

> "Your single-convoy headline was re-run after the representation fix and improved (0.362 to
> 0.276), so you kept the post-fix number. Your multi-convoy headline was re-run after the fix
> and regressed (0.295 to 0.447), so you kept the pre-fix number. Why should I accept a result
> produced by networks you have demonstrated were reading the wrong nodes' embeddings?"

The disclosed answer in the ledgers (the permutation was fixed, bijective, identical across
arms/train/eval, so the comparison is internally valid) is true but incomplete, because the fix
changed WHAT the result is evidence of:

- Pre-fix, each route's mean-pooled "embedding" pooled a fixed arbitrary set of wrong nodes: a
  route-identity hash. A policy scoring twelve distinct random signatures is functionally a
  **tabular learner over twelve abstract arms**. The pre-fix 0.295 is therefore strong evidence
  that *SAC-as-smooth-fictitious-play finds a near-minimax mixture of a 12-action matrix game
  under adversarial pressure* (a real, citable claim: the FP dynamics, the transient, the
  best-checkpoint discipline all stand) and **no evidence at all that graph-structured deep RL
  learned the security game on the map**.
- Post-fix, the same pipeline on honest embeddings lands at 0.447 twice (0.447 +/- 0.029 and
  0.447 +/- 0.008 with the role-target fix reverted and the horizon doubled): a reproducible
  plateau. The Obj-5 ordering survives (0.447 << ALNS 0.699 << vanilla 0.859), so the comparative
  claim is bug-robust, but the near-equilibrium margin currently belongs only to the tabular-ish
  regime.

Consequences for the write-up (whatever gen11 does):
1. State the multi-convoy claim at the strength the evidence supports: the *comparative* ladder
   (beats the deterministic-optimal certificate and the non-adversarial control) is bug-robust;
   the *near-equilibrium* margin (1.37x) is a pre-fix result whose representation caveat is
   disclosed; the post-fix reproduction is 0.447 and is reported beside it.
2. Do not let any sentence imply the GNN contributed spatial understanding to the multi-convoy
   headline. The single-convoy gen10-SC result is now the place where correct representations
   demonstrably helped (44% of the residual gap was the bug): make THAT the "the graph pathway
   matters" exhibit.
3. gen11 is not polish; it is the experiment that decides whether the thesis's multi-convoy act
   is "deep RL solves the fleet security game" or "FP finds matrix-game equilibria and here is a
   measured account of why the neural part fell short". Both are writable; only one needs no
   asterisk.

On "is the finding trivial" more broadly, the 2026-07-09 answer stands: the LP is the yardstick,
not the method; the non-trivial content is the dynamics characterisation (instance structure
decides learnability; the FP discipline bracket; the reproducible equilibrium transient; the
coordination chicken-and-egg and its critic-side resolution), the negative campaign, and now the
representation-flattery finding, which I would present as a first-class methods contribution:
**"suite green plus results improved" certifies nothing about representations; a fixed bijective
permutation of node embeddings improved measured performance while destroying the model's claimed
semantics.** That is a rare, honest, generalisable observation and examiners reward it.

---

## 3. Logical and argumentative weaknesses in the storyline (current state)

1. **The pre-fix/post-fix ladder mix.** The citable multi-convoy ladder pairs pre-fix SACRED
   (0.295) with pre-fix vanilla (~0.945); gen10 produced post-fix vanilla 0.859 and post-fix
   SACRED 0.447. Keep the two ladders strictly separate in every table (the house never-compare-
   across-git-states rule applies to the thesis's own figures); a mixed ladder would be the exact
   incident the rule exists to prevent.
2. **The gen10-MC2 attribution is stated one notch too strongly.** "The regression is
   attributable to the representation change itself" is justified relative to the two confounds
   MC2 removed (role-target rule, horizon), but inside that umbrella at least two distinct
   mechanisms remain unseparated: (a) the parameter-free mean-pooled menu head cannot separate
   overlapping routes (route node-set Jaccard on 62-97 k8: mean 0.38, max 0.88, 56/132 pairs at
   or above 0.5, probe this session); (b) a trainer-side conflict I found in the audit (§5.1):
   fleet-route mode pushes the followers' transitions, tagged with near-zero entropy targets,
   into the SAME shared actor that must hold the leader near 0.5*ln R, on states that post-fix
   differ from the leader's only in feature column 14. The gen10-MC telemetry (H_lead pinned at
   0.00 while the leader alpha climbs to 71: the tuner raising temperature against saturated
   logits) is at least as consistent with (b) as with (a). Cheap discriminator: §8 item 2.
3. **"All five objectives met" still overclaims** wherever it survives in the doc web (gen09
   ledger, HANDOVER banners): Obj-4 is not met by "N is a design lever"; use the §1 scoring.
4. **The multi-convoy fleet cost is still unreported** (2026-07-09 critique §3.6, unactioned):
   the randomised stack plausibly pays a real detour premium (route costs on the headline
   instance span 26.3 to 52.2); one eval-only column beside ALNS's plan cost closes it.
5. **Vanilla is never given the selection privilege SACRED gets** (best-checkpoint): report
   vanilla's best-checkpoint TAP once, for symmetry. Post-fix vanilla is also currently n=1.
6. **The objective-selection story** (loss-averse chosen because randomisation pays there) is
   fine and pre-registered; keep leading with the operational defence (mission-abort criteria)
   and the boundary map (objective x interception model x route structure) as a contribution
   table. The independence-of-interceptions conservatism sentence is still worth one line.
7. **The related-work hole is now urgent**: the assessed review has no Stackelberg/security-game
   section, and the thesis's actual contribution sits inside that literature (ARMOR/PROTECT
   lineage, PSRO/double-oracle with RL best responses, DeepFP-style solvers, network-interdiction
   OR work). The honest novelty sentence remains: max-entropy SAC as the mixed-strategy generator
   for network interdiction on a real road graph, scored against a computable equilibrium, with a
   loss-averse multi-convoy coordination extension and a measured account of the fictitious-play
   dynamics. Do not claim "first RL in security games".
8. **Duplication drift got worse, as predicted**: the standing numbers changed twice in 48 hours
   and at least SYSTEM.md §5, README, TASK.md banners and parts of HANDOVER still carry 0.283 or
   0.362 as current. Before writing starts, do one pass declaring the ledgers the only source of
   numbers and stripping values (not pointers) from the prose docs.

---

## 4. Methodological oversights (beyond §3)

1. **One instance, one K, one N** still carries the multi-convoy claim; M5 has not run (§8 item 4).
2. **No held-out instance anywhere in the positive arc**: B2-S pre-registered and never run; the
   second screened OD in the sweep tier fixes this and the sweep clause in one stroke.
3. **n=3 seeds, mean +/- population std**: keep the "individual seeds shown, no significance
   language" discipline in the thesis; the campaign's dual-reporting rule was the better standard.
4. **`switch_every` 50-vs-200** remains the one untested knob in the "mechanism port" story
   (recorded as path A, closed by decision; keep it in limitations).
5. **The Final Activities Report (30 July) is a harder rail than the freeze**: the guidance
   expects the bulk of research done by then, and the report plus presentation costs days. Plan
   the experimental programme to be effectively complete by ~27-28 July; treat Aug 3 as slack for
   re-runs, not for new directions.

---

## 5. Codebase audit: findings NEW in this pass

The 2026-07-09 audit's §5 items were re-verified: 5.1/5.2/5.3 are fixed and regression-tested
(suite 149); 5.4 (eager objective matrix in the trainer/env) is **still unfixed** and still
contradicts the "SACRED is flat in the axes that blow up the oracle" line as implemented; the
§5.5 minors stand (checkpoint save/load still omits `log_alpha_foll`, its optimiser, `follow_w`
optimiser state; per-eval checkpoints save only the actor state_dict, so reloading a menu actor
requires reattaching `menu_routes`/`follow_w` as `train_multiconvoy.py` does for the frozen
leader). New findings:

### 5.1 Fleet-route mode trains the shared actor on follower transitions with conflicting entropy targets (design defect, plausible gen10-MC contributor)

`train_multiconvoy.py::train_defender`: in fleet-route mode all N steps of every sortie are
pushed. Followers hard-copy the leader, so their "actions" are not policy samples; after
`follower_warmup` (default 250, active in the headline config) their states are tagged
`target_entropy = 0.05*ln R` and `alpha_group = 1`. The discrete-SAC actor loss does not depend
on the stored action; it is soft policy iteration on the state. So from sortie 250 onward,
two-thirds of the replay pushes train the one shared actor toward near-argmax on states that are
identical to the leader's decision state except for column 14 / the `taken` head input, while the
leader's third of pushes (plus its alpha tuner) demands entropy 0.5*ln R on the same graph.
Pre-fix, the identity-hash embeddings gave the head twelve well-separated signatures, which
plausibly let it satisfy both pressures; post-fix the signatures collapsed (§2) and the observed
failure signature (logit saturation: H_lead 0.00 while alpha climbs to 71, then late near-pure
drift) is what this conflict predicts. **Test cheaply**: in fleet-route mode, push ONLY the
convoy-0 transition, made terminal with the sortie reward (the follower steps carry no decision
content in this mode anyway). A few lines, flag-gated, one 3-seed run (~15 min at 3-parallel).
If the plateau moves, the gen10-MC attribution sharpens from "the representation change" to a
specific, fixable trainer artefact; if it does not, gen11's feature injection remains the lever
and the diagnosis is cleanly closed either way. I recommend folding this flag into the gen11
pre-registration as a factor rather than a separate generation.

### 5.2 Campaign-era buffered observations mutate in place (latent staleness defect; another disclosed confound for gen03-06 mechanisms, not for the comparisons)

`GraphEnv.observe()` returns `"nodes": self._obs_nodes, "edges": self._obs_edges` by reference;
demand arrivals/deliveries and `set_congestion` mutate those dicts in place; the transition
builder and `step_protagonist`/`step_antagonist` shallow-copy only the top level and the trucks
sub-dict. Demonstrated this session: a stored observation's node demand and edge congestion
change under later env steps. Since `SMDPTransition.feature_cache` is built lazily at the first
`update()` that samples the transition, every campaign-era buffered state's demand and congestion
columns reflect the environment at first-featurisation time (or end of that episode), not
decision time; the cache docstring's "immutable, buffered state" assumption is false. Scope:
**all gen01-gen07 rungs** (dynamic demand and/or congestion). The banked interdiction and
multi-convoy paths are unaffected (node/edge dicts are static there; trucks dicts are per-call
snapshots). Consequences: (a) the pre-registered campaign comparisons stand (the defect applies
identically to every arm); (b) the gen03/04/06 mechanism narratives ("the critic cannot resolve
which blocks worked", SNR stories) acquire a second implementation-artefact channel alongside the
node-ordering bug, and the thesis should soften those mechanism claims to "consistent with"
wording (the flat-landscape conclusion survives on its representation-independent evidence:
random ~= 96% of scripted; scripted >> learned). Fix is post-freeze work (deep-copy or snapshot
the two sub-dicts at transition creation, plus a contract test asserting a buffered state is
insensitive to later env mutation); for the thesis it is a limitations paragraph, and it makes
the disclosed-confound story consistent rather than worse: the same lesson as §2, stated once.

### 5.3 Minor (new)

- **Scale inconsistency in the correlation signal**: featurise column 14 carries the fraction of
  earlier convoys through a node (`1/N` units) while the head-level `taken` input carries raw
  counts (`rc.count(r)`); learned weights absorb the difference, but document it or unify it.
- **`_FEATURIZE_CACHE` is keyed on the node-id tuple only** (`networks.py:98`): two graphs with
  the same node ids but different edges/coordinates would silently share cached edge indices and
  normalisation constants. Harmless today (one Kaliningrad graph); a real trap for ZST step 1,
  where multiple instances will coexist in one process. Key it on (nodes, edge-set hash) before
  any multi-instance training.
- `train_multiconvoy.py:426`: the `--skip-vanilla` fallback hardcodes 0.945, which is now a
  PRE-FIX number; if the follower-bootstrap path is ever re-run post-fix, that line prints a
  stale reference (keep it out of citable output, or update to the gen10-VAN 0.859 with a
  comment pinning its SHA).
- The gen11 proposal exists only inside `experiments/gen10_postfix.md`; per house rules it needs
  its own pre-registered ledger before launch. The working tree is clean apart from a mode-bit
  change on `scratch/gen10_mc2.sh` (chmod +x); commit or discard it.

### 5.4 What this audit did NOT find

Re-checked and clean: the node-ordering fix is used at every consumer site I could find
(`select_action`, both update passes, both trainers' `hop_probs`, `menu_route_node_idx`,
antagonist paths) and the regression tests cover the adversarial case; the role-alpha target fix
is correctly conditioned on the NEXT state's group with the legacy flag byte-preserving; the
exact fleet-route evaluator is exact (one forward pass mapped onto stacked occupancies); the
smooth-FP helper matches its single-convoy semantics; the analytic expected-mission-failure
reward is a correct variance-reduction over the Bernoulli resolve; the LP constructions and the
closed-form checks remain sound; `route_one`'s per-step observation snapshots are safe in both
menu and first-hop modes.

---

## 6. Performance: what actually matters for scaling

Measured reality: `update()` dominates training (~0.37-1.0 s/sortie regimes); the featurise
Python loop is secondary; eval is now exact (one forward) on the headline path. The items that
matter are all on the attacker/oracle side:

1. **Kill the eager objective matrix** (audit 5.4, still open). `MultiConvoyInterdictionEnv.
   __init__` materialises `[#occ x #iset]` and `objective_matrix` fills it in a Python double
   loop with a Poisson-binomial convolution per entry; at the proposed N=3, K=3 scaled instance
   that is 28.8M convolutions and 230 MB before training starts, and it is why the "SACRED is
   flat in the oracle-blowing axes" sentence is not yet true of the code.
2. **Matrix-free greedy best response** (the 2026-07-09 §7 design, still the right one). For the
   mission objective, expected mission failure of a defender mixture is monotone submodular in
   the interdicted edge set, so the greedy K-edge BR carries the (1 - 1/e) guarantee, costs
   O(E*K*S*N) per refresh over the trailing-window support S, needs no matrix, and can be
   spot-checked against the exact BR at K <= 2. With it, the attacker refresh, the training
   reward (a single payoff column), and the exploitability eval (BR against a distribution
   supported on <= window occupancies) all run without ever enumerating C(E, K).
3. **Lazy objective rows** for whatever still needs exact values: compute M[occ, :] on demand
   with an LRU keyed on the occupancy; the defender only ever visits a tiny corner of the
   occupancy simplex.
4. Lower priority: vectorise `featurize_state`'s node loop with numpy; cache the static 13
   columns per graph and patch only column 14 in menu mode (the observation is otherwise
   constant across sorties); key the featurise cache correctly (§5.3).

---

## 7. Outlook: SBO, ZST, scaling (the requested informed view, post-gen10)

**SBO (Obj-4).** Unchanged verdict, higher urgency: the best value-per-hour in the remaining
programme and the only objective at "not met". The reduced form is an afternoon: design space =
screened OD pairs (the 72-pair scan) x fleet size N (x optionally K); exact objective per design
= `loss_mixed` from the oracle (seconds at K=1); features = OD distance, route count, shared-edge
mass, leader entropy H/lnR, vulnerability stats, N; fit `SurrogateMLP`; validate the argmax
design by full solve; report rank correlation and argmax regret. Scope it honestly as
"interdiction-aware placement / fleet sizing, reduced form" and write the full SBO loop
(acquisition, refinement, coupling to training) as future work. A stretch worth one sentence in
the thesis even if not run: the surrogate predicting the TRAINED policy's exploitability rather
than the oracle value would couple SBO to SACRED itself; that is post-thesis work.

**ZST.** The picture improved materially with gen10. Step 0 (held-out OD transfer of the
gen10-SC single-convoy policy, scored against that OD's own oracle ladder) is now meaningful
because the policy was trained on correct embeddings; it is an eval-only afternoon and I would
run it regardless of outcome, with the expectation set to "partial transfer, and here is why"
(the policy has never seen another OD's threat geometry: vulnerability is still not observable).
Step 1 (the crown jewel: one policy trained across sampled OD pairs/vulnerability maps, evaluated
zero-shot against each held-out pair's equilibrium) requires exactly the gen11 mechanism: per-route
cost and vulnerability delivered undiluted to the head (and/or an edge-vulnerability feature
column). That convergence is the strongest argument for gen11: it is simultaneously the
multi-convoy recovery attempt and the ZST enabler. Budget 2-4 days with a pre-registered gate;
attempt only after gen11 and the sweep tier are banked. If it does not fit, write step 1 as the
designed next experiment with step 0's measured transfer as the evidence base.

**Scaling.** The honest chain is unchanged but should now be stated with gen10's lesson folded
in: (i) exact solvers win the small-instance benchmark outright (1.4 s vs ~20 min at the headline
size) and double-oracle/column generation extends them far beyond naive enumeration, so wall-clock
alone will never carry the claim; (ii) the implementable claim is amortisation and attacker-side
approximability: a trained policy is a forward pass per sortie, transfers across instances (ZST),
and trains against a matrix-free greedy BR with a (1 - 1/e) guarantee where exact enumeration is
infeasible (K >= 3); (iii) one trained datapoint past the naive wall (N=3, K=3 after §6 items 1-2)
would convert the scaling section from projection to result, with the disclosed caveat that the
yardstick there is the greedy BR. I would still not claim more: an OR examiner knows what column
generation does to this game class. The strongest scaling story runs through ZST (re-solve per
instance vs transfer), not through wall-clock.

---

## 8. Recommended programme to the freeze (ranked, costed; every launch is Kilian's go)

Rails: Final Activities Report + presentation by 30 July (plan experiments complete ~27-28 July);
freeze Aug 3 HARD; thesis + poster 10:00, 28 Aug (12,000 words).

1. **F3 SBO demonstrator** (Obj-4; §7). Eval-only, one afternoon, zero training risk, converts
   the only "not met" objective. There is no defensible reason left to sequence anything ahead
   of it. Do first.
2. **gen11: menu-head discriminability + the follower-push diagnostic** (the thesis-integrity
   experiment; §2, §5.1). One pre-registered generation, two factors: (a) lever-2-pattern learned
   weights on per-route COST and worst-case VULNERABILITY at policy and critic heads (both
   quantities computable from the game; spreads on the headline instance are 26.3-52.2 and
   0.555-0.95, so the signal is real); (b) fleet-route pushes leader-only terminal transitions
   (flag). 2x2 or the two single-factor arms first, 3 seeds each at ~15 min per 3-seed config.
   Success = post-fix best-checkpoint TAP <= 0.295 (supersedes the pre-fix headline on honest
   representations); partial success still sharpens the §2 narrative from one confound to a
   measured decomposition. Optionally add a third arm worth one sentence in Methods: a learned
   per-route identity embedding at the head, which reconstructs exactly what the bug accidentally
   provided and cleanly separates "identity capacity" from "transferable features".
3. **Multi-convoy fleet-cost column + vanilla best-checkpoint row + post-fix vanilla to 3 seeds**
   (§3.4-3.5). Eval-heavy, hours, closes three examiner questions.
4. **M5 sweep tier** on whichever code state gen11 selects: K in {1,2,3} x N in {2,3,5} at the
   headline OD plus ONE second screened OD; 3 seeds on the headline cell, 1 elsewhere; curves
   against each cell's oracle ladder (~an evening at 3-parallel). Closes Obj-5's varied-disruption
   clause and the held-out-instance gap together.
5. **ZST step 0** (held-out OD transfer of gen10-SC; afternoon). Then **ZST step 1** only if
   items 1-4 are banked and the calendar still breathes (2-4 days; §7).
6. **Optional, strictly if early**: matrix-free greedy BR + the N=3 K=3 scaled run (§6-7); the
   one-instance learned-antagonist co-evolution demo (F2), which after the node-ordering fix
   deserves exactly one attempt because the pre-fix antagonist evidence is confounded twice over
   (§5.2 and the 2026-07-09 §3.4).
7. **Docs number-hygiene pass before writing starts** (§3.8): ledgers become the sole number
   source; prose docs keep pointers only.

Drop order if the calendar bites: 6, then ZST(1), then the second OD in 4, then 5. Items 1-3 are
cheap and are precisely what an examiner will probe; they should survive any schedule.

---

*Artefacts of this critique: this file; the route-overlap probe and the shared-dict staleness
demonstration (both quoted inline; reproducible one-liners); suite verified 149 passed this
session. No `src/` changes; no training launched.*
