# CRITIQUE_EXPANSION.md: fresh-eyes critique of the completed expansion programme (Fable, 2026-07-11)

> Requested by Kilian 2026-07-11: a holistic, examiner-grade critique of the project as it stands
> after the expansion programme (gen13 through gen18 plus the A/B/C/D ledgers), its fit against the
> five research objectives (read verbatim from the assessed literature review), a fresh codebase
> audit (bugs and performance, with scaling past the oracle in view), an informed outlook on SBO,
> ZST and scaling, and ranked future paths. Sources: the complete HANDOVER read order (every doc,
> every ledger gen01-gen18 plus a2/a3/a4/b4/d1/d2/d3/f3/zst_step0 and the untracked gen19, both
> PDFs, all load-bearing code), the full suite re-run (**161 passed**, matching the record), and
> two cheap probes run this session (the select-on-test re-computation over the saved gen15/gen16
> JSONs; the gen19 oracle screen re-run, which reproduces its ledger numbers exactly). No `src/`
> change made; no training launched. Companion to `CRITIQUE.md` (2026-07-02),
> `CRITIQUE_INTERDICTION.md` (2026-07-09) and `CRITIQUE_PREFREEZE.md` (2026-07-10); this file
> critiques what the expansion programme delivered.

---

## 0. Verdict in six sentences

The expansion programme did what it set out to do: every one of the five research objectives now
has trained, pre-registered evidence in at least a demonstrated form, the aim-level ZST promise is
realised at the held-out-CITY level (gen16, the strongest single result in the project), Obj-4 has
gone from the weakest objective to a complete three-tier SBO stack (F3, D1, D2, D3), and the two
failed gates (gen17, gen18) closed their questions cleanly and made the standing caveats stronger,
not weaker. By MSc standards this is now an unusually complete and unusually honest body of work;
the binding risk has moved from evidence to **selection and framing**: nineteen generations, two
headline games, a ZST arc, an SBO stack and a rigorous negative campaign must fit 12,000 words,
and the rubric puts 50% of the mark on Methodology/Analysis/Discussion and 20% on structure. The
sharpest remaining examiner attack is no longer "is this a tautology?" but **"the LP re-solves an
unseen instance exactly in 5 ms (your own A3 measurement), so what exactly does zero-shot transfer
buy?"**: the honest answer exists (§3.1) but is not yet nailed anywhere in the doc web. The
biggest remaining evidential gap against the objectives' own wording is that **no positive result
anywhere in the programme contains a learned adversary** (Obj-1's "environment-altering antagonist
agent" is an oracle best response in every banked ladder; the one clean post-fix attempt, F2, has
still never run). The audit found no new critical bug: the two findings that matter are a measured
(and survivable) selection-on-test subtlety in the gen15/gen16 best-checkpoint discipline (§4.2)
and the fact that the in-flight gen19 work sits uncommitted in the working tree with its solution
concept needing one careful framing decision before launch (§4.3). The ranked programme in §9
fits the real calendar (Final Activities Report 30 July; freeze 3 August): roughly one more week
of small, high-leverage experiments, then writing wins every conflict.

---

## 1. State verification (what this critique stands on)

- Suite: **161 passed** (re-run this session), matching the gen16-era record.
- **A second agent instance is concurrently active in this repo** (discovered mid-critique): it
  committed the gen19 trio (`6219522`, 09:25) with a smoke already PASSING the primary
  (0.131 < iid_eq 0.147) and launched the 3-seed gen19 batch, which is running as this file is
  written. The gen19 observations in §4.3 were made from the then-untracked files and stand;
  the "commit before launch" recommendation was fulfilled by the build instance itself, and the
  two pre-registration additions recommended in §4.3 become eval-only post-run rows.
- The gen19 oracle screen re-run this session reproduces the ledger's pinned numbers exactly
  (35-159, w=3, tau=0.15: static_det 0.613, iid_eq 0.147, history_opt 0.049).
- **Chronicle gap:** `SACRED_PROGRESS.md` stops at entry 18 (the 2026-07-10 overnight programme).
  There is no entry for gen13-lock, gen14, gen15, gen16, gen17, gen18 or the A/B/D ledgers, and
  the HANDOVER banner's pointer to "entries 17-19" dangles (no entry 19 exists). Kilian's standing
  instruction is that every significant run family gets an entry; the Final Activities Report will
  be written from exactly this chronicle. Cheap, and worth doing before memory of the arc fades.
- Duplication drift is under control this time: the numbers policy (ledgers as sole source) has
  held since 2026-07-10. The HANDOVER banner stack is now five layers deep and increasingly
  costly to parse; one consolidation pass before the writing phase would pay for itself.

---

## 2. Fit against the five research objectives (verbatim, lit review §2.2), post-expansion

The guidance PDF requires the conclusions chapter to revisit each objective and confirm whether it
was met. The honest scoring as of today; deltas that must be named are marked.

**Obj 1: "Formulate the SDVRP as an asymmetric zero-sum Markov game, defining discrete action
spaces for a protagonist dispatcher and an environment-altering antagonist agent."**
Met with three declared deltas, two unchanged from earlier critiques and one new. (a) The positive
results live in a Stackelberg security game with hidden pre-commitment, not the simultaneous-move
RARL game the review promised: a strengthening, but say so. (b) **The "antagonist agent" in every
positive result is still the oracle best response.** The learned-antagonist attempt (F2) has now
been recommended by two consecutive critiques and never run; after gen10's fix removed both
confounds on the pre-fix antagonist evidence, exactly one clean attempt remains scientifically
due. If it is not run, the conclusions chapter must scope Obj-1 explicitly ("the oracle interdictor
upper-bounds any learned adversary; learned-adversary co-evolution was studied only in the
negative campaign"). (c) NEW: gen17 strengthens Obj-1's dynamics content: four failed
hold-the-tail attempts across two instances and two eras make "the equilibrium is a reproducible
transient of last-iterate fictitious play" a systematic finding rather than a caveat. The
problem-class slide (no stochastic demand, no within-sortie dynamism in the headline game) stands;
gen19 is designed to return the D and is the right priority (§9).

**Obj 2: "Design and implement a visual, interactive multi-agent simulation environment..."**
Met, and strengthened: the multi-city extraction pipeline (arterial filter + 30 m consolidation +
the length-repair fix), the per-city LCC instance sampler and the four-city registry are genuine
environment contributions beyond the campaign-era machinery. Suite 161; fidelity gates G1/G3/G-M1.

**Obj 3: "Develop the SACRED framework, utilising a SAC architecture and ATLA, and investigate the
efficacy of ERB bootstrapping via population-based metaheuristics to accelerate training
convergence and ensure policy coevolution."**
Substantially met, with the same two re-interpretations to declare (ATLA realised as fictitious
play against an oracle BR; ERB realised as demonstration bootstrapping). gen18 is a real Obj-3
addition despite failing its bar: with all three original handicaps removed (post-fix embeddings,
favourable instance, properly-scaled `follow_w` that trained to 2.93), followers still collapse to
fixed routes, which upgrades "structural stacking" from caveat to measured boundary and points
future work at exploration, not signal. The one cheap item that would close the VERBATIM wording
("ERB bootstrapping via population-based metaheuristics") is C1: seed the replay buffer with
ALNS-plan demonstrations and run a time-to-competence ablation on 35-159 (~a day). Without it,
gen01 (inconclusive, n=1) plus the demonstration-bootstrap arc is what the words rest on.

**Obj 4: "Incorporate SBO into the SACRED framework, utilising a neural network metamodel to
approximate facility location and fleet composition, thereby enabling the holistic, simultaneous
evaluation of strategic supply chain design alongside the operations-level SDVRP."**
**Met, and now arguably the most complete objective.** F3 (surrogate regression, Spearman 0.894,
argmin regret 0), D1 (the acquisition loop proper: median 33 evaluations to the optimum vs random
never reaching it: PASS + STRONG), D2 (hardening tier; the L1 = 0.29 equilibrium-shift interaction
is exactly the "holistic, simultaneous" claim), and D3 (the composite: surrogate over the TRAINED
generalist, Spearman 0.959, and the policy-vs-oracle design-target rank correlation of **0.768**,
which is a genuine finding: designing against the deployed policy differs from designing against
the equilibrium abstraction, and only the RL + surrogate loop can do the former). Honest scope to
declare: the surrogate approximates game/policy evaluations at K=1 instance sizes where the exact
solve is cheap; the value proposition is the loop pattern and the policy-target capability, not
compute savings at this scale.

**Obj 5: "Evaluate the performance and resilience of the SACRED framework against SOTA adaptive,
population-based metaheuristics and a baseline non-adversarially SAC-trained agent under varied
levels of network disruption."**
Met, strongly. Both headline ladders sit on corrected code with n=10 CIs (gen14: MC 0.256
[0.246, 0.266]; SC paired dD 0.175 [0.137, 0.213] excluding zero, 10/10 seeds); the disruption
clause has 10/10 trained cells across K, N and two instances (gen12); the fairness rows
(ALNS-forced-stack; vanilla best-checkpoint; fleet-cost column showing the security premium equals
the equilibrium's own) pre-empt the natural attacks; B4 converts the independence assumption into
a robustness curve on which independence is provably conservative. The standing framing rule
remains: ALNS's defence is that it is verified to reach `loss_det`, the optimum of the entire
deterministic class; never call the in-house ALNS "SOTA" unqualified. One control gap remains at
the ZST level (§5.1).

**ZST (aim-level: "resilient, zero-shot transferable logistics policies that standard algorithms
cannot achieve").**
Realised: gen15 (held-out OD 1.59x), gen16 (held-out CITY 1.68x, beats each OD's deterministic
optimum on 17/18 cells, and the A2-rescue row showing multi-graph training fixes the measured
cross-graph failure). The transfer-difficulty ladder (same-graph OD 1.59 -> held-out city 1.68 ->
single-source cross-graph ~random) is exactly the kind of measured boundary an examiner rewards.
Deltas to declare: 6 ODs per held-out graph, one held-out city, N=3 K=1 only, all cities produced
by one extraction pipeline, and the selection subtlety in §4.2.

---

## 3. Is the finding trivial? (re-assessed after the expansion)

Materially stronger than at the 2026-07-09 assessment. The non-trivial content now stands on four
legs, in descending order of strength:

1. **The ZST arc.** A policy that routes fleets at 1.68x equilibrium on a never-seen city, and a
   measured account of WHY single-source transfer fails (graph-overfit GNN) and what fixes it
   (graph variety), is a research result no LP formulation touches: the LP has no notion of
   transfer at all. This is also the project's honest answer to "you screened the instance where
   you win": gen16's Gdansk was held out entirely.
2. **The dynamics characterisation**: instance asymmetry decides learnability (the 72-pair screen,
   the gen12/gen13 instance contrast); the FP discipline bracket; the systematic transient finding
   (gen17's four-attempt ladder); the identity-vs-semantics head decomposition (gen11b: E' 0.295
   vs B' 0.408, "memorisable identity beats transferable semantics where the equilibrium is
   flat"); and gen18's exploration-side coordination boundary. None of this falls out of the LP.
3. **The SBO composite** (D3's 0.768: the operationally-optimal design is not the
   equilibrium-optimal design), plus the D2 tier coupling.
4. **The negative campaign** (gen03-06) with its mechanism chain, as motivation and as a
   standalone methodological contribution.

What remains genuinely thin, and should simply be conceded rather than defended: at every measured
instance size, the exact solve is faster than the policy (A3: 5.1 ms vs 3.4 ms per instance, and
exact); wall-clock never carries any claim (already the recorded position); and the multi-convoy
act's coordination content is leader-mix plus structural stacking (gen18 makes this a clean
boundary statement).

---

## 4. Logical and argumentative weaknesses (ranked by examiner danger)

### 4.1 The ZST value proposition against a 5 ms LP (the sharpest open attack)

A3 measured the thing an OR examiner will quote back: on fresh instances at headline size, the LP
re-solve is **faster and exact** (5.1 ms, ratio 1.00) versus the generalist's forward pass
(3.4 ms, ratio 1.90). So "one policy, no re-solving" cannot be the ZST motivation at these sizes,
and any sentence that implies compute superiority ("proving ZST in a way an LP solve cannot
match", DIRECTION_EXPANSION axis A framing) is falsified by the project's own ledger. The honest
case for ZST has three legs and should be written exactly once, prominently:
1. **The LP needs the game; the policy needs an observation.** An exact solve requires the full
   instance specification (graph, OD, threat map, K) as a trusted model at decision time; the
   trained policy conditions on the observed map and produces behaviour where no solver or model
   pipeline is deployed. This is a deployment-architecture argument, not a wall-clock one.
2. **Scale**: past the enumeration wall (K >= 4-5, larger route sets, instance streams), exact
   re-solving stops being 5 ms; the amortisation frontier bends. This leg becomes a measured claim
   only if the A4 large-K training cell runs (§9).
3. **D3**: the design loop needs the DEPLOYED policy's exploitability, which no LP can score; ZST
   makes that evaluation one forward pass. The 0.768 correlation is the evidence this distinction
   matters.
Without this paragraph the gen16 act is exposed to a one-line rebuttal; with it, the act is safe.

### 4.2 Best-checkpoint selection on the held-out metric (measured this session; survivable, fix the reporting)

gen15/gen16 select the best checkpoint by the HELD-OUT mean ratio itself (disclosed in both
ledgers). Strictly, that makes the test set a validation set: the reported number is a minimum
over ~24 evaluations of the reported metric. Re-computed from the saved JSONs this session:

| run | select-on-TEST (as reported) | select-on-TRAIN (honest alternative) | final iterate |
|---|---|---|---|
| gen15 (held-out OD) | 1.592 +/- 0.096 | **1.592 +/- 0.096 (identical: same checkpoints chosen)** | 1.99 |
| gen16 (held-out city) | 1.677 +/- 0.072 | **1.733 +/- 0.149** (seed 1 moves 1.773 -> 1.941) | 2.20 |

Both PASS bars survive train-side selection (gen16: 1.733 <= 2.0, still below the ~1.99
random-init reference, and the 17/18 loss_det clause is a checkpoint-level property worth
re-stating at the train-selected checkpoint). Recommendation: dual-report both selections in the
thesis (one table row each), lead with select-on-train as the deployable claim, and keep the
select-on-test number as the optimistic bound. This converts a potential examiner "gotcha" into
another exhibit of the house discipline. Also report the final-iterate drift (1.99/2.20) beside
them, as the ledgers already do.

### 4.3 gen19 changes the solution concept, and the ledger's framing needs one correction before launch

The B1-lite-1 adversary (softmax best response, temperature tau, to a w-window of realised play)
is a **quantal-response attacker with bounded memory**: a behavioural model, not a worst-case
opponent. Two consequences the current pre-registration under-states:
1. Its own screen shows `iid_eq` = 0.147 **below** `V_eq` = 0.206: the pattern-of-life adversary
   is WEAKER than the minimax adversary even against static equilibrium play (the ledger's "the
   static-mixed defender lands ~its equilibrium" reads 0.147 as approximately 0.206; it is 29%
   below). The correct story: the game measures **exploitation of a boundedly-rational, adaptive
   adversary**, in which (i) determinism is destroyed (0.613), (ii) static mixing is already
   over-conservative (0.147), and (iii) history-aware play exploits the adversary's own adaptation
   down to 0.049, three times below static mixing and four times below the minimax value. That is
   a legitimate and well-precedented security-games register (quantal response is standard in the
   deployed-security-game literature), but it is a different claim from every prior act's
   "approach the minimax equilibrium", and the thesis must not blur the two.
2. **Add the worst-case row when the results land** (eval-only, minutes; the 3-seed batch was
   already in flight when this critique was written): the trained history-aware policy's
   exploitability against the ORACLE committing interdictor (best response to its induced
   stationary route mixture). This quantifies what the exploitation costs in worst-case terms and
   pre-empts "your dynamic policy is itself a pattern": the expected picture is that the
   history-aware policy pays a worst-case premium over V_eq, and reporting it is what keeps the
   act inside the thesis's robustness through-line. Also report the (w, tau) grid from the screen
   as curves (the operating point tau = 0.15 was chosen because tau = 0.05 makes the adversary
   trivially dodgeable; disclosed, but a sensitivity row makes it unassailable).
Also: the no-window causal control in the trainer is exactly right; and eval noise in the
best-checkpoint min (2000-sortie stochastic evals) deserves either a TAP-style smoothing or a
confirmation re-run at the selected checkpoint.

### 4.4 The SDVRP frame (standing, partially addressed)

The S of SDVRP (stochastic demand) remains entirely in Acts I-II; the D returns only if gen19
lands. The title-versus-delivery tension is manageable exactly as before (deliberate refinement,
evidenced by the negative campaign), but the conclusions chapter must say it in one plain
sentence rather than let the examiner say it first. If gen19 passes, the sentence improves to
"within-episode dynamism restored in the security-game register; stochastic demand remains the
recorded extension (full B1)".

### 4.5 The multi-convoy act's honest strength

gen18 settles it: the fleet result is a LEADER-MIX with structural stacking, and emergent
coordination under independent exploration is an open problem (measured, post-fix, favourable
instance, trained follow_w). Write the act at exactly that strength: the Obj-5 ladder is
unaffected (the deployable object is the stacked mixture), the coordination boundary is a
contribution, and any sentence implying the fleet "learned to coordinate" would be false. The
sceptical-examiner bank should gain the entry: "why is stacking optimal at all?" (answer: the
mission objective plus shared-edge structure; the oracle's stacked-optimum-equals-equilibrium
check was part of instance screening: 0.206 = 0.206 on 35-159).

### 4.6 City-family homogeneity (scope condition, not a flaw)

All four cities pass through the same arterial-filter + 30 m-consolidation pipeline; gen16's
transfer claim is therefore "across cities within one graph-construction family". The A2 row
(transfer to the differently-constructed kaliningrad_original: 1.90 vs random 2.43) partially
covers the pipeline axis and should be cited for exactly that; a leave-one-city-out rotation
(§5.2) covers the city axis. State the scope; do not let a reviewer discover it.

### 4.7 The compression problem (now the largest single risk to the mark)

Nineteen generations, three critiques, two headline games, a transfer arc, a four-part SBO stack,
a dynamics study and a negative campaign, against 12,000 words, a 10% abstract, 10% conclusions
and a 20% structure/presentation weighting. The thesis needs ONE spine, and everything not on it
must compress to a sentence plus a ledger citation. Recommended spine (one line per act): when
does adversarial deep RL buy unexploitable logistics policies? (I) The negative: with a
congestion adversary, never: mechanism chain, four preconditions (one results section, heavily
compressed). (II) The positive: in the interdiction security game, with instance asymmetry and
smooth-FP discipline: two ladders against computable equilibria. (III) The payoff: the policy
transfers zero-shot across ODs and cities, and prices a three-tier design stack no exact method
can (ZST + SBO). (IV) The boundaries: transients, structural stacking, identity-vs-semantics,
bounded-adversary dynamism (gen19), each one paragraph. The gen09 STAB arc, gen11 mechanics,
B2-P/P2 bracket and campaign rung details belong in appendices or citations to the repo. Writing
this selection down early (a figure-and-table shortlist per act) is worth more than any remaining
experiment except gen19 and F2.

### 4.8 Related work (unchanged urgency, now with more to position against)

The assessed review contains no Stackelberg/security-game section; the thesis's contribution now
sits squarely inside that literature and must position against it: the deployed security-games
lineage (ARMOR/PROTECT/IRIS; Tambe's book), network interdiction in OR (Wood; Washburn and Wood;
the Smith and Song survey), learning in games (double oracle: McMahan et al.; PSRO: Lanctot et
al.; fictitious play: the review's own Heinrich and Silver citation), deep solvers for security
games (DeepFP-style work), quantal response and behavioural attackers (QRE; SUQR in green
security games: the literature home of gen19's adversary), adversarial patrolling, and GNN
generalisation for combinatorial optimisation (the ZST act's ML context). Every reference must be
verified against the original before citation (the guidance PDF treats fabricated references as
plagiarism); the names above are search targets, not citations.

---

## 5. Methodological oversights (beyond §4)

1. **The ZST act lacks a trained non-adversarial control.** gen15/16 compare the generalist to
   random-init, shortest-path and each OD's loss_det, but not to a VANILLA generalist (same
   multi-instance training, travel objective, no adversary). Obj-5's named control exists at
   single-instance level only. Without it, "adversarial training is what makes transfer work" is
   an inference, not a measurement (a cost-trained generalist would presumably concentrate on
   cheap routes and transfer badly, but that is exactly the row to show). One seed, ~1-1.5 h,
   eval-only harness already exists. Highest-value cheap addition to the ZST act.
2. **Single held-out city.** A leave-one-city-out rotation (hold out each of the four cities in
   turn, 1 seed each) turns "transfers to Gdansk" into "transfers to whichever city is held out"
   at ~3-4 h total. Cheap insurance against the "you picked the easy hold-out" reading.
3. **The generalist is trained and evaluated at N=3, K=1 only.** A zero-shot K=2 / N=5 evaluation
   row (no retraining; the policy conditions on the map, not on K) would show whether the learned
   hedge survives adversary-budget shift: either outcome is informative, and it is eval-only.
4. **gen19 needs its worst-case row and (w, tau) curves** (§4.3) added to the pre-registration
   before launch, not after.
5. **Statistical reporting**: the gen14 n=10 discipline (t-CIs, per-seed lists) should be the
   template for every number that reaches the thesis; gen15/16/17/18 are n=3 with population std
   and should say so in their table captions. No new significance language anywhere.
6. **Calendar**: the Final Activities Report and presentation (due 30 July, "bulk of the research
   done" expectation) should be drafted from `SACRED_PROGRESS.md`; the chronicle gap (§1)
   therefore blocks a deliverable, not just hygiene.

---

## 6. Codebase audit: bugs

### 6.1 What was checked and found sound

The load-bearing post-fix paths were re-read end-to-end this session: `node_index_map` used at
every consumer (select_action, both update passes, hop_probs, menu_route_node_idx, the
generalist's eval path); the per-transition menu/feature plumbing in `sac.py` (per-sample menus
correctly override net attributes for both current and next states, in the right order); the
exact fleet-route evaluator (one forward mapped onto stacked occupancies); the vectorised
mission-objective matrix (clamped log-survival matmul, equivalence-tested); the greedy submodular
BR (verified against exact at K <= 2); the smooth-FP helper; the LP constructions (row-minimiser,
cost-constrained frontier); the analytic expected-mission-failure reward; the ALNS
reaches-loss_det verification; the gen19 trainer's game logic (reward uses the pre-action window,
matching the screen's MDP; the no-window control zeroes only the history column). The 2026-07-09
and 07-10 audit items remain fixed where they were fixed. **No new critical defect found.**

### 6.2 Findings (none invalidates a banked number)

1. **Selection-on-test in gen15/gen16** (§4.2): measured, survivable, fix the reporting. The
   fix-forward for future generations: select on the train-set mean, report held-out at that
   checkpoint.
2. **`follow_w` trains at the base lr in `train_generalist.py`** (no `head_term_lr` on its param
   groups, unlike `route_feat_w`): inconsequential for the deployable object (fleet-route leader
   has `taken = 0`), but inconsistent with the gen18 lesson; tidy or annotate.
3. **The eager objective matrix still stands** (`MultiConvoyInterdictionEnv.__init__` builds
   `[#occ x #iset]` unconditionally; `CRITIQUE_INTERDICTION` §5.4, third critique in a row): with
   the vectorised build this is now a RAM wall rather than a time wall, and it is the single
   blocker for the K >= 4 training cells (A4's recorded remaining step). See §7.
4. **Stale-reference hygiene**: `train_multiconvoy.py`'s `--skip-vanilla` fallback still hardcodes
   0.859 (now labelled, still a trap if the ladder moves again); `save_checkpoint`/`load_checkpoint`
   still omit `log_alpha_foll`, its optimiser and the `follow_w`/head-term optimiser state
   (resuming a role-alpha run silently resets them; per-eval actor snapshots are what the
   programme actually relies on, so this stays latent).
5. **`_FEATURIZE_CACHE` grows without bound** across instances (keyed per graph + edge-set
   signature). Harmless at 24 instances; a 900-design D-loop or a large A3 stream over many
   graphs would accumulate; an LRU or explicit clear is a two-line hardening.
6. **gen19 eval noise**: `eval_policy`'s 2000 stochastic sorties give the best-checkpoint min a
   noise floor; smooth (TAP-style) or re-confirm the selected checkpoint with a longer run.
7. **Repo state**: the gen19 trio was untracked (now committed alongside this critique); the
   chronicle gap (§1).

### 6.3 What this audit did not find (for balance)

No indexing regression in the multi-city path (menu indices and `node_index_map` both sort
stringified ids; verified consistent); no cross-instance cache contamination (the featurise cache
key covers the edge-set signature; per-transition menus ride the observations); no leakage of
held-out instances into training (pool split by fixed pool-seed; hold-out city never sampled for
training); no error in the gen19 oracle screen (re-run, matches); the equivalence and regression
test net (161) covers the places earlier bugs lived (node ordering, route head terms, obs
snapshots, vectorised objective, greedy BR).

---

## 7. Performance: what matters for scaling past the oracle

Measured reality: `update()` dominates training; eval is exact and cheap on every headline path;
the oracle LP is milliseconds at K=1 sizes. The items that change what EXPERIMENTS are possible,
in order:

1. **Wire `greedy_br_attacker` into the trainer and the eval** (A4's recorded remaining step).
   Replace `env.obj_matrix` in the smooth-FP refresh with a greedy BR against the trailing-window
   occupancy support (tiny support, so O(E*K*S*R) per refresh is trivial), and gate the eager
   matrix build behind `K <= 3`. This single change makes the K = 4/5 training cells honest (the
   env currently materialises the very matrix the scaling claim says is infeasible) and is the
   only code prerequisite of the "trained where the exact oracle cannot follow" datapoint.
2. **Lazy objective rows with an LRU** for whatever still wants exact values at K <= 3 (the
   defender visits a tiny corner of the occupancy simplex; per-row closed forms already exist).
3. **Vectorise `featurize_state`'s node loop and cache static columns per graph**, patching only
   the dynamic columns (col 14 and edge occupancy): the featurise path is secondary today but
   becomes the constant factor in any B1-style campaign (many decisions per episode) and in
   larger-city training (Istanbul is 1266 nodes and its rows are rebuilt per decision).
4. **Batch the per-sample head loop in `update()`**: the encoder is batched, the heads are a
   Python loop over ~32 samples of tiny tensors; padding to a fixed menu width would vectorise
   actor and critic heads in one shot (~1.3-2x on the measured update-bound profile). Worth it
   only if a large training campaign (B1, bigger generalist pools) is funded.
5. Non-items: MPS (re-confirmed slower, twice), the LP itself (milliseconds), eval (exact).

---

## 8. Outlook: SBO, ZST, scaling (the requested informed view)

**SBO.** The stack is complete for thesis purposes (regression -> acquisition -> hardening tier ->
composite over the trained policy); I would spend at most one more half-day here, on the one
exhibit that composes the whole project: **run the D3 loop on the HELD-OUT city with the
multi-city generalist** (design base placements in Gdansk, priced by a policy that never trained
there; eval-only; all machinery exists). That is the poster image: strategic design in an unseen
theatre, operational tier priced by ZST, no retraining, no LP able to participate. Beyond that:
write it up. The honest scope sentence stays: at K=1 sizes the oracle is cheap and the surrogate's
value is the loop pattern plus the policy-target capability (D3), which becomes compute-necessary
exactly when evaluations get expensive (large K, instance streams, trained-policy targets).

**ZST.** The result is real and the boundary map (OD -> city -> construction-pipeline) is the
right shape. To harden before the freeze, in value order: the vanilla-generalist control (§5.1),
the select-on-train dual-report (§4.2, numbers already computed), leave-one-city-out (§5.2), the
zero-shot K/N rows (§5.3). The deeper questions (what graph variety is sufficient; whether
fine-tuning beats zero-shot at matched budget; scaling the pool) are honest future work. Guard the
framing with §4.1: ZST's worth is deployment structure, scale and D3, never wall-clock at small
sizes.

**Scaling.** The honest chain is now fully evidenced except its last link: exact solvers win small
instances outright (A3, measured); the enumeration wall is real and measured (K >= 4-5, RAM); the
matrix-free greedy BR with a (1 - 1/e) guarantee exists and is verified; what is missing is ONE
trained cell at K = 4 or 5 on 35-159 with the greedy BR as both sparring partner and disclosed
yardstick (§7 items 1-2, then an ordinary run). With it, the scaling section converts from
projection to measurement; without it, the thesis should simply not claim a scaling result beyond
the amortisation frontier framing. Against column generation / double oracle the exact-method
frontier extends far past naive enumeration, and the thesis should keep conceding that in one
sentence: the strongest scaling story remains ZST + D3, not wall-clock.

---

## 9. Ranked programme to the freeze (firm recommendation; every launch is Kilian's go)

Rails: FAR + presentation due 30 July (draft from the chronicle; plan experiments effectively done
by ~27 July); freeze 3 August HARD; thesis + poster 10:00, 28 August (12,000 words). Today is
11 July: about 2.5 real experimental weeks, and §4.7 argues the marginal mark now lives mostly in
writing. The list below is deliberately small.

1. **Chronicle + doc hygiene (hours, immediate).** Append SACRED_PROGRESS entries 19+ for
   gen13-18, the expansion ledgers and gen19; fix the HANDOVER "entries 17-19" pointer. Unblocks
   the FAR.
2. **gen19 / B1-lite-1 (the D restored; in flight as of this writing).** When the 3-seed batch
   lands: add the worst-case-exploitability row and the (w, tau) sensitivity curves (§4.3,
   eval-only) beside the pre-registered primary, and run the no-window causal control if the
   build instance has not already. This is the single biggest thesis-claim upgrade left.
3. **F2: the one clean learned-interdictor attempt (afternoon).** Third critique in a row
   recommending it; closes the Obj-1 "antagonist agent" gap in the positive arc, or produces the
   honest "oracle-bounded" sentence with a measured basis. Evaluation stays oracle-BR
   portfolio-max regardless of outcome.
4. **ZST hardening batch (about a day total).** Vanilla-generalist control row (one seed);
   select-on-train dual-report (numbers in §4.2, eval-only); leave-one-city-out rotation (3 x 1
   seed); zero-shot K=2 / N=5 rows (eval-only).
5. **C1: ERB-from-ALNS ablation (~1 day).** Closes Obj-3's verbatim wording with a pre-registered
   time-to-competence read on 35-159.
6. **A4 large-K cell (1-2 days).** §7 items 1-2, then one K=4 (or K=5) trained cell on 35-159
   against the greedy BR, reported with the (1 - 1/e) caveat. The last scaling link.
7. **D3-on-Gdansk composite exhibit (half day, eval-only).** The poster centrepiece (§8).
8. **Then freeze and write.** FAR by ~27-28 July from the chronicle; the thesis spine of §4.7;
   ledgers as the sole number source; dual-selection and scope sentences from §4-5 folded in.

Drop order if the calendar bites: 6, then 7, then 5, then the rotation half of 4. Items 1-3 and
the control row of 4 should survive any schedule; they are cheap and they are exactly what an
examiner will probe. I would not fund B1-lite-2, full B1, B3, B5 or further coordination/leader
work before the freeze under any outcome: the gates that closed them were bought expensively and
should stay closed.

---

*Artefacts of this critique: this file; the select-on-test re-computation (quoted in §4.2,
reproducible from the gen15/gen16 JSONs); the gen19 screen verification. Suite verified 161
passed this session. No `src/` changes; no training launched.*
