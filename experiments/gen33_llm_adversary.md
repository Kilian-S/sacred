# Generation: gen33_llm_adversary (the LLM as adversary-force generator: a heterogeneous, terrain-grounded red force for the aerial interdiction game)

- **status: PRE-REGISTERED 2026-07-22.** Written BEFORE any LLM call. The bar STRUCTURE is
  pinned here; absolute bar VALUES are anchored from the free oracle screens (marked `[ANCHOR]`)
  and written into this ledger before the first model call. Nothing above the RESULTS section
  moves afterwards.
- **branch:** `gen28-aerial` (worktree `../sacred-aerial`); code additive/new-file where it
  touches shared modules.
- **provenance:** Kilian's supervisor-directed B2 pivot (2026-07-21/22). Supervisor's steer
  (Prof. Angeloudis, verbatim intent): the world must be rich enough that reasoning adds value
  (else a solver or a two-line heuristic matches the model); local models are VRAM-bound and
  quantised, weak at multi-step reasoning but strong at distilling doctrine/text into a few
  decisions that feed the simulator, done in a batch session; the Automated Adversary is the
  interesting route, ideally a population of trait-driven agents whose traits are precalculated
  offline and whose interactions then run algorithmically. Every locked decision below is on
  record in the design conversation.

## Why (the objective repair + the gap)

The banked aerial positive (gen31/gen32) beats a SINGLE hand-tuned anticipatory doctrine enemy,
which was itself found by an automated 48-cell search. The adversary was one designed formula.
This act asks a different question: can an LLM, reading terrain plus a doctrine brief, GENERATE
the red force, a heterogeneous, coordinated, terrain-grounded adversary population, offline, as
structured parameters that feed the existing solver and trainer unchanged?

This repairs the aerial branch's weakest point, its thin fit to the five research objectives:

- **Obj 1** (asymmetric zero-sum game, environment-altering antagonist): the antagonist becomes a
  POPULATION of trait-driven agents, a materially richer instantiation than a single scripted
  enemy.
- **Obj 3** (adversarial training, population coevolution): the LLM force is the adversary
  distribution SACRED co-trains against (the curriculum metric below).
- **Obj 5** (resilience under VARIED levels of network disruption): the LLM-generated force IS the
  varied disruption, terrain-grounded and diverse rather than a single parameterised threat.

The supervisor's steer is load-bearing: this is a text-distillation task (doctrine plus terrain
mapped to a few structured decisions), the register local quantised models are strongest at, not
the multi-step-reasoning register they are weakest at.

## The finding we already hold (Phase 0, oracle-only; folded from `gen33_llm_adversary_static_probe.py`, measured 2026-07-21)

A free oracle probe of exactly this act's central question, on the real Kaliningrad to Gvardeysk
corridor, with Kilian's own heterogeneous laydown (2 long-range SAMs at 2.5 km / 0.90 on
open-field, 3 short-range MANPADS at 1.2 km / 0.85 on any emplaceable terrain; N=3 stacked
drones, mission objective):

- **(A) A static heterogeneous laydown, even reshuffled every episode, TIES the naive frontier**
  (best-naive / equilibrium = 0.99-1.02x, one outlier 1.29x). Reshuffling placement per episode
  does not open a corridor.
- **(B) Deterministic multi-system laydowns are degenerate** on this corridor with stacking plus
  mission: too few systems and the reactive optimum collapses to ~0 (trivially evadable), too
  many and it saturates (equilibrium 0.97). The non-degenerate, moderate contest gen32 achieves
  comes from the enemy's MIXED strategy (the security-game equilibrium distribution) plus the
  degraded effectiveness field, NOT from deterministic placement of K systems.

**BINDING DESIGN CONSTRAINT (pre-registered from this finding):** a purely static red force is
expected to TIE the naive frontier, so the LLM force must be DYNAMIC and MIXED to clear the
baseline. The act's bar therefore lives in the dynamic register; a static red force is measured
only as the pre-declared control. This is the disjoint-baseline lesson arriving before any
training CPU is spent.

## The game and roles

- Aerial interdiction on real OSM vec-theatres (the gen32 substrate). **Defender** = the SACRED
  fleet routing policy (N=3 UAVs, loss-averse mission objective P(>=1 lost)). **Adversary** = the
  LLM-designed red force.
- **Dynamic register** (locked): pattern-of-life doctrine on the gen31/gen32 window machinery.
  The enemy reacts to the defender's realised routes over a window; the defender is history-aware.
- **Physics fixed by terrain** (locked): weapon range and effectiveness are read from the terrain
  class of the emplacement site (the `hazard_sites` contract, already how the code works). The LLM
  sets WHERE and HOW-IT-BEHAVES, never lethality. This is the honesty safeguard: gen28 showed
  firepower compresses the gap and structure widens it, so denying the LLM firepower control is
  what forces any advantage to come from positioning and coordination.
- **Coverage-fraction range scaling** (locked): ranges scale per map so the single-site coverage
  fraction phi = 2Kr/W lands in the screened non-degenerate band; terrain RATIOS stay fixed
  (forest ~0.48x open, the current 1.2/2.5). The ledger reports phi, not kilometres, as the
  coverage axis, so difficulty is comparable across theatres and any residual difference is
  terrain STRUCTURE. Militarily: theatre scale sets the weapon class (tactical SHORAD on small
  corridors, operational SAMs on large ones).
- **Phase 1: K=1** (single agent), scored individually (banked gen32 machinery). **Phase 2: K=3
  coordinated** (joint placement plus team roles). The joint-doctrine semantics of K=3 dynamic
  coordinated agents (how three doctrines share a window and aim jointly) do not yet exist in the
  code and are pinned by an oracle screen during the build, before any phase-2 bar. Both phases
  are mandatory deliverables.

## The LLM output contract (structured; frozen before any call)

One call produces the WHOLE force as structured JSON (guided decoding). Per agent:

- `archetype` (enum): sniper_overwatch, ambusher, anticipator, blocker, forward_picket
- `emplacement_zone`: terrain class + corridor region (near_base / mid_corridor /
  near_target_standoff / chokepoint)
- `doctrine`: {punish_pattern, anticipate_flight, hold_static} on the simplex (the gen31
  q_rep / q_flee / q_eq)
- `decisiveness`: aim-temperature bin (the gen31 tau); `memory`: window w in {1, 2, 3}
- `terrain_preference`: concealment (forest, short range) vs reach (open, long range)
- `risk`: forward vs deep posture
- `team_id`, `team_role` (bait / block / cover / anchor) [phase 2 only]
- `rationale`: free text (the transcript record and the terrain-use check)

The physics table IS shown to the LLM (locked: the designer knows their weapons; the test is
composition, not guessing ranges). Placement zones resolve to actual candidate sites on the
emplaceable grid; doctrine parameters feed `stepdmg`; the assembled force is handed to the solver
and trainer unchanged.

## Archetypes (the heterogeneity targets, and the hand-written baseline)

sniper_overwatch (open ground, long sightline, commits), ambusher (forest chokepoint, punishes
the pattern, concealed), anticipator (sits at the obvious escape route, the gen31 winning
doctrine), blocker (the terminal funnel or pinch), forward_picket (aggressive, near the base).
The two-line DOCTRINE HEURISTIC baseline places agents on the highest-value terrain the rulebook
implies with a single fixed archetype; the LLM force must beat it on hardness AND on diversity.

## Metrics and bars (structure pinned; `[ANCHOR]` values set from the free screens, before any LLM call)

> **Metric 1, HARDER WORLD (oracle-only, freeze-safe):**
> - *Intrinsic (GATED):* the LLM force's induced game value against a best-responding defender,
>   placed between random placement (floor) and the oracle-optimised force (ceiling).
>   **BAR:** the LLM force beats the two-line doctrine heuristic on >= 2/3 of the three core
>   theatres (kgd, ukraine, narva; all screen-confirmed usable) AND pooled, for BOTH phase 1
>   (K=1) and phase 2 (K=3), per model.
> - *Deployed (REPORTED, ungated):* the LLM force's interception against the fixed gen32 policy;
>   per model, per theatre, no pooling.
>
> **Metric 2, CURRICULUM TRANSFER (the headline; needs the w05 training slot):**
> Three arms, defender trained against: (i) the LLM force population, (ii) random placement,
> (iii) the single gen32 hand-tuned doctrine. Three seeds each (a 3x3 batch, run concurrently on
> w05). Held-out test = the oracle-optimised force plus the gen32 doctrine on the held-out
> theatres (Ukraine, Narva).
> **BAR:** the LLM-curriculum defender's zero-shot held-out value is below BOTH control
> defenders' on `[ANCHOR >= 4/6]` held-out cells AND pooled, on >= 2/3 seeds, at the
> validation-selected checkpoint. **STRONG:** `[ANCHOR]`. Untrained-defender context row and
> select-on-val discipline reported as standing.
>
> **Metric 3, DOCTRINAL FIDELITY (oracle-only):**
> Does the force match doctrine and terrain? Scored by placement value vs the oracle placement and
> by archetype-terrain appropriateness. **CONTROLS (binding):** a scrambled-terrain field
> (relabel the classes, preserve the statistics) must DEGRADE the force materially (proof it reads
> terrain, not surface statistics, the gen28 permuted-field control); a renamed-map field (strip
> place names) must NOT change it (no memorisation of real geography).
>
> **DIVERSITY (reported beside metric 1; gates nothing but is the population's whole point):**
> trait-space spread (entropy over archetypes and doctrine simplex) and placement-space spread
> (spatial dispersion of emplacements) across the generated agents. A force that collapses to one
> archetype fails the population premise and is reported as such.

## Baseline family (pre-registered, make-or-break; the gen26 / disjoint-baseline dogma)

Every metric-1 ladder carries ALL of: random placement (the floor); the two-line DOCTRINE
HEURISTIC (best-terrain, single archetype, the beat-me baseline); the oracle-optimised force (the
ceiling); and the tabular-FP mixed enemy where relevant. Baselines receive the SAME budget K as
the LLM force (fairness). BOTH models (llama-3.3-70b, qwen3-27b), reported per-model and
per-instance, NO pooling across models.

## Theatres

Core (one terrain vocabulary, all land): **Kaliningrad to Gvardeysk** (primary and training base,
gen32 machinery proven), **Ukraine to Zaporizhzhia** (held-out, open-field character), **Narva**
(held-out, forest plus the river chokepoint). **Karelia** in reserve if Narva screens degenerate.
Coverage-fraction range scaling per map. Maritime and alpine theatres (Hormuz, Singapore, Hong
Kong, Taiwan, Fulda, Alps) are a SEPARATE showcase, NOT in the scored core: different terrain
family, unvalidated mechanics.

## Models and harness

Local only (the department box): llama-3.3-70b + qwen3-27b, quantised, no tools, structured output
via vLLM guided JSON (fallback: few-shot plus parse plus retry). Every request and response
auto-logged (the transcript record). Training on w05 (CPU; MPS was slower for these small graphs),
three arms concurrent; the professor has been given a heads-up per the shared-box rule; thread
pools capped per process; run artefacts pulled back into the repo.

## Fail branches (all writable, all are results)

- **LLM force ties the naive frontier** (matches Phase 0): the finding is that even LLM-composed
  heterogeneous forces need the mixed/dynamic register, static composition adds nothing, a
  measured boundary consistent with the whole programme.
- **Diversity collapses to one archetype:** the LLM is a doctrine-lookup, not a population
  generator; reported.
- **Curriculum no better than random or single-doctrine:** the diversity-drives-transfer
  hypothesis is falsified; the boundary is the result.
- **Doctrinal fidelity fails the scrambled-terrain control:** the LLM pattern-matches rather than
  reasons from terrain; the claim re-scopes to amortised placement with no map-conditioning
  content.

## Engineering queue (before any screen or CPU; the ledger is written, the bars anchor after)

1. **Loader adapter:** `load_vec_theatre` reads the new `poly`/`line` format alongside the old
   `classes`; the `sea` class maps to non-emplaceable (like `water`).
2. **Terrain-table close-out:** add `sea`; encode the coverage-fraction range-scaling rule.
   Alpine no-fly and coastal/island long ranges stay parked (showcase-only).
3. **Free non-degeneracy screens** on Ukraine and Narva (this also measures the compute envelope
   on the large Narva grid, which sets the phase-2 K=3 budget). Karelia if Narva degenerates.
4. **Map-to-text serialiser** + the frozen trait schema above.
5. **The generation harness** (guided JSON, both models) + the baseline family.

Bars are anchored from steps 3 to 5 into this ledger BEFORE the first LLM call.

## RESULTS (appended per step; nothing above changes)

### NON-DEGENERACY SCREEN (2026-07-22, oracle-only, `scratch/gen33_screen.py`; suite 214 green incl. 5 new loader/scaling tests)

Loader extended to the new `poly` fetch format with `sea` non-emplaceable; terrain table gains
`sea` and `alpine`; weapon range scales per map by lateral-width ratio (kgd = reference,
lat_w 28 km). Two gates run per core theatre:

**Static feasibility** (uniform field; confirms plumbing + the K=3 envelope). All three build, all
K=3 exact matrices are feasible (< 60M cells), coverage fraction held comparable by the scaling:

| theatre | scale | R | sites | phi | eq | best-naive/eq | K=3 matrix |
|---|---|---|---|---|---|---|---|
| kgd (ref) | 1.00 | 25 | 185 | 0.16 | 0.373 | 1.17x | 26M |
| ukraine | 2.04 | 26 | 195 | 0.17 | 0.253 | 1.18x | 32M |
| narva | 2.27 | 26 | 227 | 0.12 | 0.294 | 1.13x | 50M |

The static uniform-field game is near-symmetric (leader entropy ~1.0) and the naive stack ties
within ~1.15x, exactly the Phase 0 picture: static placement is not where the contest lives.

**Dynamic doctrine gate** (the trainability screen; random degraded field + the pinned gen31/gen32
operating point q=(0.7 rep, 0.3 flee), tau=0.10, w=2; mean over 3 field seeds). The degraded field
breaks the symmetry and the doctrine opens a wide corridor on ALL THREE:

| theatre | eq_static | leader entropy | G1 (static cap / opt) | G2 (blind rules / opt) | verdict |
|---|---|---|---|---|---|
| kgd | 0.261 | 0.55 | 7.39 | 3.40 | DOCTRINE CONTEST |
| ukraine | 0.206 | 0.67 | 8.56 | 4.23 | DOCTRINE CONTEST |
| narva | 0.195 | 0.63 | 7.00 | 3.57 | DOCTRINE CONTEST |

**Verdict: all three core theatres are usable in the dynamic register** (leader entropy 0.55-0.67
< 0.95, static play capped 7-8.5x the dynamic optimum, blind rules 3.4-4.2x). **Narva does NOT
degenerate, so Karelia is not needed in reserve.** Ukraine and Narva carry slightly wider corridors
than Kaliningrad. These anchor the bar objects: the LLM force is scored inside this doctrine game
against the random (floor), two-line doctrine-heuristic (beat-me) and oracle-optimised (ceiling)
adversary rows, which the generation harness computes per generated force.

### BUILD RECORD (2026-07-22; the w05-ready generation half; suite 214 -> 218 green)

Engineering queue steps 1-5 landed, all local, no training, no live model:

1-2. **Loader + terrain table** (`src/envs/aerial_theatre_vec.py`): the loader reads the new
`poly`/`line` fetch format alongside the old `classes`; `sea` and `alpine` classes added
(non-emplaceable; sea no-LOS, alpine LOS-blocking); `range_scale` hook on `hazard_sites` /
`build_theatre_game` plus a `lateral_width` helper for coverage-fraction scaling. Default
`range_scale=1.0` keeps kgd and all existing games byte-identical. Regression tests
`tests/test_theatre_vec_scaling.py` (5).

3. **Screens** (`scratch/gen33_screen.py`): the table two sections above. All three core theatres
usable in the dynamic register; Karelia dropped.

4-5. **The LLM I/O contract** (`src/redforce.py` + `scripts/gen33_generate_force.py`): the frozen
guided-JSON `FORCE_SCHEMA`, the terrain-to-brief serialiser (physics table shown, per decision 4),
a synthetic `dry_force`, and the resolver mapping an emitted force onto sites + normalised doctrine.
The harness serialises -> calls (dry or OpenAI-compatible `guided_json`) -> validates -> resolves
-> saves. **Dry-run passes end-to-end on all three theatres, both phases** (single K=1, coordinated
K=3: 3 sites/force, 3 archetypes). Regression tests `tests/test_redforce.py` (4). **The generation
half is w05-ready:** flip `--provider openai --base <tunnel> --model llama-3.3-70b` to go live.

**Enemy-semantics finding (the one open design item, surfaced while wiring the resolver).** The
first-cut resolver maps a force to K HARD sites. Phase 0 (B) already shows why that cannot be the
dynamic-register semantics: with too few fixed sites the enemy is trivially evadable (reactive
optimum ~0). So the enemy the LLM force parameterises must **mix over a broad site support**, with
the agents' zones shaping WHERE the mass concentrates (a soft weighting over the grid) and the
doctrine setting the temporal behaviour, exactly the gen31 enemy but with an LLM-shaped site prior.
This is the phase-2 joint-doctrine semantics the pre-registration reserves for an oracle screen; it
is now pinned as: *force -> a soft site-weighting (summed agent concentrations) + doctrine; the
enemy plays gen31 over the full grid under that prior; K = the number of concentrations.* The
scorer and the random/heuristic/oracle baseline forces are built against THAT semantics (next step,
oracle-only, local), and confirmed before any live scoring is trusted. The generation half above is
unaffected (it emits and resolves forces regardless of how they are scored).

### PHASE 0 (folded 2026-07-22 from `gen33_llm_adversary_static_probe.py`, oracle-only, Kaliningrad to Gvardeysk)

The static heterogeneous laydown finding, moved here from a commit message per the house rule
(numbers live in ledgers):

- **(A) static heterogeneous, reshuffled per episode, TIES the naive frontier:** best-naive / eq
  = 0.99-1.02x (outlier 1.29x). Static placement does not open a corridor.
- **(B) deterministic multi-system laydowns are degenerate:** too few systems, reactive optimum
  ~0; too many, saturated eq 0.97. Non-degeneracy comes from the mixed strategy plus the degraded
  effectiveness field, not from deterministic K placement.
- **Consequence carried into the design:** the red force must be dynamic and mixed to clear the
  naive frontier (see the binding design constraint above). Artefact
  `models/runs/gen33_llm_adversary_static_probe.json`, regenerable oracle-only in seconds.

### RUN RECORD (2026-07-22, the concurrent live generation on the workbench; suite 218 -> 220 green)

**Harness concurrency (additive).** `scripts/gen33_generate_force.py` now fans the
(model x theatre x phase x force-index) tasks over a capped thread pool (`--workers`, default 12;
the calls are HTTP-bound so threads are correct), validating + resolving each force as it returns;
`--models a,b` holds both models in flight simultaneously. New regression tests
`tests/test_gen33_harness.py` (2: fallback JSON extraction, concurrent dry fan-out). Suite tail:
`220 passed, 4471 warnings in 44.07s`.

**Gateway structured-output finding (mechanics, not results).** The workbench gateway ACCEPTS but
silently IGNORES vLLM's `guided_json` extra (HTTP 200 + free prose from llama, off-schema JSON
from qwen; measured with single diagnostic calls). It HONOURS OpenAI
`response_format: {type: json_schema}`, which is now the harness's guided mode. Failure mode
observed on llama under schema enforcement: whitespace degeneration until `finish=length`
(thousands of near-empty lines); handled by the pre-registered retry ladder (guided resample,
then the few-shot + parse + retry fallback), with the mode that produced each force recorded in
the artefact.

**SMOKE (both models in flight, 12 concurrent tasks: 2 models x 3 theatres x 2 phases x n=1).
PASS.** 12/12 schema-valid, all in guided mode, correct site counts (1 or 3), normalised
doctrines, team fields present in every coordinated force, zero resolve-fallbacks, zero endpoint
or timeout errors. Per-call latency 11-46 s (one llama task 136 s = one length-capped guided
attempt + successful resample, trail recorded); total wall-clock 145 s.

**FINAL RUN (2 models x 3 theatres x 2 phases x N=8 = 96 concurrent tasks, 12 workers).
96/96 valid, wall-clock 553 s.** Modes: 89 guided, 7 few-shot fallback (ALL seven = llama on the
single phase, the whitespace-degeneration cells; qwen 48/48 guided). Resolve-fallbacks 0
everywhere. Artefacts `models/runs/gen33_forces/force_<model>_<theatre>_<phase>.json` (one per
cell, with per-force mode, latency and retry trail; these are LLM samples, NOT regenerable from
seeds). Per-cell (archetypes = unique archetypes across the cell's 8 forces; latency is
task-level and includes retries):

| model | theatre | phase | valid | archetypes | modes | lat mean/max s |
|---|---|---|---|---|---|---|
| llama-3.3-70b | kgd | single | 8/8 | 2 | 6 guided, 2 fallback | 201.2/426.0 |
| llama-3.3-70b | kgd | coordinated | 8/8 | 2 | 8 guided | 50.6/243.8 |
| llama-3.3-70b | ukraine | single | 8/8 | 2 | 6 guided, 2 fallback | 105.6/377.5 |
| llama-3.3-70b | ukraine | coordinated | 8/8 | 2 | 8 guided | 22.7/38.6 |
| llama-3.3-70b | narva | single | 8/8 | 3 | 5 guided, 3 fallback | 148.1/318.1 |
| llama-3.3-70b | narva | coordinated | 8/8 | 3 | 8 guided | 42.4/180.8 |
| qwen3-27b | kgd | single | 8/8 | 2 | 8 guided | 12.3/16.6 |
| qwen3-27b | kgd | coordinated | 8/8 | 4 | 8 guided | 30.3/40.6 |
| qwen3-27b | ukraine | single | 8/8 | 1 | 8 guided | 11.9/14.6 |
| qwen3-27b | ukraine | coordinated | 8/8 | 1 | 8 guided | 26.7/30.2 |
| qwen3-27b | narva | single | 8/8 | 2 | 8 guided | 13.1/20.1 |
| qwen3-27b | narva | coordinated | 8/8 | 2 | 8 guided | 25.3/30.0 |

**Population signatures (descriptive, per model, no pooling across models; the DIVERSITY metric
proper is scored later beside metric 1).** Agent-level archetype counts over each model's 48
forces: llama = blocker 46, sniper_overwatch 40, ambusher 9, anticipator 1; qwen = ambusher 75,
blocker 11, anticipator 9, sniper_overwatch 1. Neither model EVER emits forward_picket (0/192
agents). qwen's ukraine cells collapse to a single archetype per cell; flagged for the diversity
report. No bar moved; no training started; enemy-scoring semantics untouched.

### SCORER SEMANTICS + CONFIRMATION SCREEN (2026-07-22 night; pinned BEFORE any LLM force is scored; suite 220 -> 224 green)

**The pinned semantics, implemented** (`src/redforce_score.py`; regression-tested to reproduce
the gen32 `DynTheatre` exactly in the flat-prior single-agent case). A force of K agents induces
ONE dynamic enemy on the gen31/gen32 window machinery: per agent, an RBF site-concentration
prior over the FULL candidate grid centred at the agent's resolved site (scale sigma, the soft
site-weighting); the agent's own doctrine simplex (punish_pattern = q_rep on the recent window,
anticipate_flight = q_flee at the myopic escape from its own prior-shaped repeat aim,
hold_static = q_eq on the site value against the static equilibrium mix), softmaxed at its own
tau, using its own memory w; the joint enemy = the equal MIXTURE of the K agents' aim
distributions. Induced game value = the best-responding history-aware defender's stationary
damage (relative value iteration; the gen32 yardstick; HIGHER = harder world). Heterogeneous
memories share one lifted window chain (w_max = the force's largest w).

**Screen** (`scratch/gen33_score_screen.py`, decision rules pre-written in its header;
artefact `models/runs/gen33_score_screen.json`):

- **CONSISTENCY PASS:** flat-prior single-agent (0.7, 0.3, 0) tau 0.10 w2 reproduces the gen32
  DynTheatre history_opt to 6 decimals on all 3 field seeds (0.020505 / 0.021355 / 0.032148).
- **SIGMA RULE -> sigma0 = 8 km** (kgd scale; scaled per theatre like ranges). Grid (2, 4, 8):
  every value passes S1 (non-collapse) + S2 (placement sensitivity >= 0.05) on both phases, so
  the pre-written rule takes the largest. Shape: at sigma 2 a badly-placed force is trivially
  evaded (random 0.0045, the Phase 0 B behaviour); at sigma 8 the doctrine contest survives
  everywhere and placement still swings the value by 2.6-2.8x (oracle vs random).
- **Baseline correction made pre-scoring:** the random/oracle constructors initially forbade
  site stacking; the schema allows it (and the resolver produces it), so the ceiling searched a
  smaller space than the LLM can emit (visible as narva oracle-K3 < oracle-K1). Fixed to allow
  stacking before any anchor was banked.
- **ANCHOR LADDER (the metric-1 rows; mean over field seeds 5100-5102; random = 16 draws
  mean+/-sd; oracle = disclosed-budget search on seed 5100, winner re-scored on the 3 seeds):**

| theatre | phase | random floor | heuristic (beat-me) | oracle ceiling | eq_static |
|---|---|---|---|---|---|
| kgd | single K=1 | 0.0309 +/- 0.0179 | 0.0313 | 0.0927 | 0.261 |
| kgd | coordinated K=3 | 0.0419 +/- 0.0118 | 0.0336 | 0.0963 | 0.261 |
| ukraine | single K=1 | 0.0234 +/- 0.0119 | 0.0178 | 0.0994 | 0.206 |
| ukraine | coordinated K=3 | 0.0366 +/- 0.0075 | 0.0183 | 0.0973 | 0.206 |
| narva | single K=1 | 0.0127 +/- 0.0079 | 0.0193 | 0.0848 | 0.195 |
| narva | coordinated K=3 | 0.0236 +/- 0.0056 | 0.0194 | 0.0779 | 0.195 |

**Honest flag, recorded before scoring:** the two-line heuristic sits BELOW the random-population
mean in 4/6 cells under these semantics (its top-exposure stacked placement is predictable), so
the pre-registered beat-the-heuristic bar is a WEAK bar here; results will also report each
population's position against the random mean and the oracle ceiling, and the bar verdict is
stated on the pre-registered object only.

### METRIC 1 RESULTS + DIVERSITY (2026-07-22 night; `scratch/gen33_score_forces.py`; artefact `models/runs/gen33_force_scores.json`)

Each banked force scored as mean best-response damage over the 3 pinned seeds at sigma0 = 8 km
(theatre-scaled); population mean +/- sd vs the anchor rows; per model, NO pooling across models.

| model | phase | kgd | ukraine | narva | theatres beat-heur | pooled LLM vs heur (rand) | BAR |
|---|---|---|---|---|---|---|---|
| llama | single | 0.0284 (below) | 0.0215 (BEATS) | 0.0187 (below) | 1/3 | 0.0228 vs 0.0228 (0.0223) | **NOT MET** |
| llama | coordinated | 0.0435 (BEATS) | 0.0300 (BEATS) | 0.0349 (BEATS) | 3/3 | 0.0361 vs 0.0238 (0.0341) | **MET** |
| qwen | single | 0.0249 (below) | 0.0145 (below) | 0.0210 (BEATS) | 1/3 | 0.0201 vs 0.0228 (0.0223) | **NOT MET** |
| qwen | coordinated | 0.0271 (below) | 0.0253 (BEATS) | 0.0277 (BEATS) | 2/3 | 0.0267 vs 0.0238 (0.0341) | **MET** |

**VERDICT (pre-registered object, both phases required): metric 1 intrinsic NOT MET for either
model.** Phase 1 fails for both (the single-agent populations are statistically ~random
placement); phase 2 PASSES for both (llama 3/3 theatres, qwen 2/3 + pooled). The mechanism
reading: composition/coordination is where the LLM adds hardness. llama's coordinated pooled
mean (0.0361) also clears the random-population mean (0.0341); qwen's does not (0.0267 < 0.0341).
Both remain FAR below the oracle ceiling (0.078-0.099): the LLM reaches 33-45% of oracle
hardness at K=3.

### METRIC 3 RESULTS: FIDELITY + BOTH CONTROLS (2026-07-23 night; `scratch/gen33_controls_generate.py` + `scratch/gen33_controls_score.py`; artefacts `models/runs/gen33_forces_controls/` + `models/runs/gen33_controls_scores.json`)

Controls generated live (kgd, both phases, both models, n=8 per cell per control; 64/64 valid;
scrambled = terrain-label 3-cycle in the brief with statistics preserved; renamed = neutral
codename), resolved and scored on the TRUE map with the pinned protocol.

| model | phase | scrambled rel-shift | renamed rel-shift | arch overlap (scr/ren) |
|---|---|---|---|---|
| llama | single | **+46%** | +25% | 0.88 / 0.75 |
| llama | coordinated | -3% | -3% | 0.25 / 0.62 |
| qwen | single | -8% | -18% | 0.88 / 0.88 |
| qwen | coordinated | +10% | +30% | 0.79 / 0.71 |

**VERDICT: the binding scrambled-terrain control FAILS (the pre-written fail branch fires).**
Scrambling every terrain property in the brief does NOT materially degrade the induced hardness
anywhere (one cell even improves), and the shifts are the same magnitude as the renamed-control
shifts, i.e. resampling noise (population sd/mean is 25-99% at n=8). Per the pre-registration,
the terrain-reading claim is NOT licensed: the claim re-scopes to **amortised placement with no
demonstrated map-conditioning content**. Supporting numbers: archetype-terrain fidelity rubric
llama 0.37 / qwen 0.60 (many agents sit off their archetype's canonical ground); placement value
= 0.33x (llama) / 0.26x (qwen) of the oracle ceiling. This coheres with metric 1: the models'
distinct archetype signatures are model priors, not terrain inference.

### METRIC 2 PRE-TRAINING PIN (2026-07-23 ~01:00, written BEFORE any training sortie; anchors set)

- **Arms** (`scripts/train_aerial_dyn33.py`; SAC settings = the gen32 trainer's verbatim; equal
  budget 8000 sorties; seeds 0/1/2): **llm** = the llama-3.3-70b kgd population (16 forces, both
  phases; choice rule = the model with the stronger metric-1 phase-2 result, disclosed);
  **random** = fixed random forces (K mix 1/3 matching the population's phase mix); **single** =
  the screened gen32 operating point (0.7, 0.3, 0) tau 0.10 w2, flat prior. Enemy-to-instance
  assignment fixed across arms and seeds.
- **Selection:** the checkpoint with the LOWEST validation value (4 kgd fields 3000-3003, the
  arm's OWN curriculum). Held-out cells are NEVER evaluated during training (no select-on-test
  surface).
- **Held-out cells (6, fixed):** {ukraine, narva} x {oracle-K1 force, oracle-K3 force (the
  banked screen winners), gen32-doctrine flat}; each cell = mean exact chain value over PRISTINE
  field seeds 4100-4102 (all gen33 screens used 5100-5102; these are untouched).
- **BAR `[ANCHOR pinned]`:** the llm-arm defender's held-out value is BELOW BOTH control arms'
  on **>= 4/6** cells AND pooled, on **>= 2/3 seeds**, at the val-selected checkpoint.
  **STRONG:** 6/6 AND pooled on 3/3 seeds. Untrained-defender context row reported as standing.

### METRIC 1 DEPLOYED ROW (reported, ungated; `scratch/gen33_deployed_row.py`; artefact `models/runs/gen33_deployed_row.json`)

The banked kgd populations against the FIXED gen32 trained policy (gen32_confirm seed10,
val-selected actor_ep15000), field 5100, duplicates deduplicated, per model per phase:

| model | phase | policy-suffered | best-response | premium |
|---|---|---|---|---|
| llama | single | 0.0534 +/- 0.0168 | 0.0284 | 1.88x |
| llama | coordinated | 0.0680 +/- 0.0083 | 0.0435 | 1.56x |
| qwen | single | 0.0401 +/- 0.0289 | 0.0249 | 1.61x |
| qwen | coordinated | 0.0457 +/- 0.0130 | 0.0271 | 1.69x |

The LLM forces are 1.56-1.88x harder for the deployed gen32 policy than for a best-responding
defender: zero-shot enemy transfer costs the trained policy roughly what the gen27/gen32
worst-case premiums cost, and llama's coordinated population is the hardest object (0.0680).

### METRIC 2 RESULTS, PRELIMINARY AT REDUCED BUDGET (2026-07-23 ~01:40; artefacts `models/runs/gen33_curriculum/`)

**What happened to the batch (disclosed plainly):** the 9-run batch (3 arms x 3 seeds, pinned
8000 sorties) was launched locally at ~00:05 (location deviation from the pinned w05 disclosed
to Kilian in-session and ratified). At sortie 1000 every run had written its eval + checkpoint;
the batch's wrapper process was then killed by an unattributed external signal (~00:40-01:09;
not by this agent; the harness task reports "stopped"). The true pace (42 min / 1000 sorties at
9-way contention) also means the quoted ETA was wrong (full batch ~5.6 h). All nine runs died
at the SAME 1000-sortie budget with one checkpoint each, so the held-out read below is an
EQUAL-BUDGET PRELIMINARY comparison, NOT the pre-registered 8000-sortie read. The pre-registered
metric-2 verdict remains OPEN pending a full-budget re-run (staged; one command).

**Held-out cells at the 1000-sortie checkpoint** (`scripts/eval_aerial_dyn33.py`, 10 parallel
evaluations; fold `models/runs/gen33_curriculum/heldout_fold.json`): every arm learns
(untrained 0.150-0.190 -> trained 0.038-0.126; the gen32doc cells improve most, ~4x), but the
ARMS ARE INDISTINGUISHABLE at this budget: differences are in the third decimal. Per seed, the
llm arm is below BOTH controls on 6/6 cells + pooled (seed 2) but 0/6 (seeds 0, 1): **bar NOT
MET at the reduced budget (1/3 seeds), verdict = inconclusive-by-budget** (curricula have not
differentiated at 1/8 of the pinned budget; this is NOT evidence against the hypothesis, and it
is not the pre-registered read).

**Diversity (the population premise; reported, gates nothing).** Archetype entropy is LOW
everywhere (0.21-0.63 of max; qwen-ukraine literally single-archetype = 0.00). Placement spread:
in 4/6 single-phase cells the population's resolved placements COLLAPSE to one site
(pairwise distance 0.0 km); mechanism = the RESOLVER (same terrain+region zone -> same
best-exposure site), i.e. the LLM's zone choices are homogeneous per theatre and the resolver is
deterministic. Coordinated forces spread properly (6.4-23.6 km pairwise vs 14-43 random).
Doctrine spread is real (mean pairwise L1 0.27-0.77). The pre-written "diversity collapses"
fail-branch fires PARTIALLY: trait diversity narrow, placement diversity resolver-limited at
K=1, healthy at K=3.

### ACT VERDICT (2026-07-23 ~02:00; what gen33 established and what stays open)

- **The generation contract WORKS at population scale:** 96/96 + 64/64 control forces valid and
  resolvable from two local quantised models, concurrent, schema-enforced (via response_format;
  the act's engineering deliverable).
- **Metric 1 intrinsic: NOT MET overall** (phase-1 fails both models; phase-2 PASSES both:
  llama 3/3 theatres + above the random mean pooled, qwen 2/3). The surviving positive claim is
  SCOPED: *LLM-composed COORDINATED forces induce harder worlds than the two-line doctrine
  heuristic; single-agent placements are ~random.* Deployed row: the forces cost a fixed trained
  policy 1.56-1.88x its best-response value.
- **Metric 3: the binding scrambled-terrain control FAILS,** so the composition skill is NOT
  demonstrably terrain-grounded (amortised placement re-scope; the renamed control's shifts are
  the same size as sampling noise, so no memorisation claim either way).
- **Metric 2: OPEN.** Preliminary equal-budget read (1000/8000 sorties, batch externally killed)
  is inconclusive-by-budget (1/3 seeds favour the LLM curriculum). The pre-registered read needs
  the staged full-budget re-run:
  `for arm in llm random single; do for s in 0 1 2; do PYTHONPATH=. OMP_NUM_THREADS=1 nice
  .venv-or-sacred-venv/bin/python scripts/train_aerial_dyn33.py --arm $arm --seed $s --sorties
  8000 --json-out models/runs/gen33_curriculum/${arm}_s${s}.json --ckpt-dir
  models/runs/gen33_curriculum/${arm}_s${s}_ckpts ... & done; done` (~5.6 h at 9-way local, or
  w05 once the repo exists there), then `scripts/eval_aerial_dyn33.py` x10 + the fold.

### TERRAIN-MISMATCH APPENDIX (2026-07-25; code audit, no re-run; binding for how metric 3 reads)

**Found while designing gen39.** This act's terrain table declares `forest: los=True`, but
`route_survival` masks line of sight with the URBAN union only. The forest flag is read in exactly
one place in the repository: `src/redforce.py::serialise_theatre`, the prose brief handed to the
model, which tells it that forest "conceals you (blocks line of sight)" and that forest is "cover
for the drones". **Both statements were false of the simulator.** Concealment also had no
mechanical meaning at all in this act, because the defender never observes enemy positions, so
there was nothing to be concealed from. Net: under the gen33 table, sitting in forest bought a
shorter range (1.2 km against 2.5 km, about a fifth of the covered area) for +0.02 lethality and
nothing else.

**What this does and does not change.** It does NOT touch metric 1, the deployed row, the
diversity rows or any gen31/gen32 number: the physics was identical across every arm, so all
comparisons here remain internally valid. What it changes is the READING of metric 3. The
scrambled-terrain control failed, and the ledger attributed that to the model not reasoning from
terrain. There is now a second candidate cause: the model was briefed on a mechanic the world does
not implement, and its own rationales show it acting on that brief (forest choices justified as
"concealment"). **Binding wording:** the terrain-grounding claim stays unlicensed, as recorded,
but the failure may no longer be attributed solely to the model. The honest sentence is that the
control cannot separate "the model does not read terrain" from "our terrain did not do what we
told the model it did", and that gen39 repairs the environment so the question can be asked
properly.

gen39 (`experiments/gen39_concealment.md`) implements forest LOS, makes urban emplaceable, and
gives concealment a meaning (a concealed site that engages does not reveal itself), then re-runs
the composer comparison with the model asked for doctrine and roles only.
