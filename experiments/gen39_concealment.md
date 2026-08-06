# Generation: gen39_concealment (the terrain redesign: concealment buys persistence, and the LLM composes doctrine)

> **READER'S SUMMARY (2026-07-28; the act is COMPLETE through step 5 + zero-shot rows).**
> Results are appended in chronological order and superseded blocks are left visible with their
> reasons (there are nine). If you want the state rather than the history, read
> `../HANDOVER_AERIAL_28-07-26.md`, then these blocks in this order:
> **STEP 1 RESULTS, THIRD RUN** (the mechanic + operating point) -> **STEP 2 RESULTS** (the LLM
> composition positive + the terrain control) -> **STEP 3 RESULTS** (the curriculum negative) ->
> **PHASE 1 / 1C / 1D / 1E / 1F** (why it failed, and where an LLM does earn its place) ->
> **STEP 5 PRE-REGISTRATION / RESULTS / n=3 / ZERO-SHOT** (the negative, fixed and transferred).
> Everything above the first RESULTS line is the original pre-registration and is unchanged.

**status: PRE-REGISTERED 2026-07-25.** Mandate: Kilian's 2026-07-24/25 direction, in-conversation.
The act has two halves that share one environment change: (i) a terrain table in which the enemy
faces a real reach-versus-cover trade, and (ii) a re-aimed gen33 in which the language model
composes DOCTRINE and ROLES only, at a screened team budget, while an algorithm does the
placement. Ledger committed BEFORE any code exists; results are appended below the RESULTS line
and nothing above it changes. **Training launches require Kilian's explicit go; the oracle screen
is free under the standing rule; the generation step hits the shared GPU box and needs his go.**

**git SHA:** the commit landing this ledger; each step pins its own SHA at its record.

---

## Why

Two measured facts drive this act.

1. **The surviving positives all sit in one corner.** Across the whole programme the only cells
   where a trained policy beat every simple rule were dynamic ones in which the defender's cheap
   escape was closed: roads at K=3 on an m=6 instance (gen35, 3/3 seeds, -8.6% against the best
   two-line rule, a tie at K=2), and the aerial acts against a doctrine that punishes both
   repetition and naive avoidance (gen31/gen32). The unifying variable is not budget as such; it
   is whether a two-line avoidance rule still has somewhere safe to go.
2. **gen33's own result points at the same corner, and its terrain control failed for a reason
   that is partly ours.** A single LLM-placed team was statistically indistinguishable from random
   placement; three COORDINATED teams beat the hand-written doctrine heuristic for both models.
   Composition, not placement, is where the model added value. Meanwhile the binding
   scrambled-terrain control failed, and a code audit (2026-07-25) found that the terrain table
   declares `forest: los=True` while `route_survival` masks with the URBAN union only. The forest
   flag is read in exactly one place in the repository: `src/redforce.py`, the prose brief handed
   to the model, which tells it that forest "conceals you (blocks line of sight)". So the model
   was briefed on a mechanic the simulator does not implement, and its rationales show it
   reasoning about that mechanic ("concealment"). The gen33 control failure therefore has two
   candidate causes, not one, and the honest fix is to implement the world we described.

More deeply: concealment currently buys nothing at all, because the defender never observes enemy
positions, so there is nothing to be concealed FROM. Forest is a strictly worse place to sit
(1.2 km reach against 2.5 km, i.e. about a fifth of the covered area, for +0.02 lethality). This
act gives concealment a meaning that costs one observation channel and makes the reach-versus-cover
choice strategic.

**Nearest prior evidence for the new mechanic (gen34, roads, disclosed):** the defender was given
the realised interdiction set after every sortie as two observation columns. It did NOT convert
that into inference of a hidden enemy TYPE (0/18 held-out cells, pooled 1.373x the type-blind cap).
But the channel was causally useful short of inference: the head weights on those columns trained
strongly negative (dodge-recent-attacks), and the no-intel control sat far worse when it was
stopped early. gen39 asks for exactly the behaviour gen34 demonstrated (avoid what you have seen),
not the behaviour it failed at (infer what you cannot see). The corollary, taken seriously below:
"avoid the revealed sites" is a strong TWO-LINE RULE and must be in the baseline family from the
start.

---

## The environment change (pinned)

### Terrain table v2

Ranges are RELATIVE to the standing anchor (v1 open = 2.5 km at kgd scale); the existing
`range_scale` multiplier applies on top and is a screen axis, so the absolute difficulty is chosen
by measurement, not by hand. Firepower stays terrain-set, never adversary-set (the standing
honesty safeguard).

| class | emplace | r (rel) | r @ kgd | p_max | reveals on engagement | blocks LOS |
|---|---|---|---|---|---|---|
| open | yes | 1.4 | 3.5 km | 0.90 | **yes** | no |
| field | yes | 1.0 | 2.5 km | 0.85 | **yes** | no |
| forest | yes | 0.6 | 1.5 km | **0.55** | **no** | **yes** |
| urban | **yes** | 0.4 | 1.0 km | **0.45** | **no** | yes |
| water | no | - | - | - | - | no |
| sea | no | - | - | - | - | no |
| alpine | no | - | - | - | - | yes |

Rationale, one line each: open ground has clear sightlines and the best sensor horizon (longest
reach, full lethality); farmland carries clutter and hedgerows (same class of system, slightly
degraded); forest restricts arcs and masks the site's own sensor, implying shoulder-launched
systems (short reach, much lower kill probability, concealed); urban is the extreme of the same
trade (very short sightlines between buildings, shortest engagement window, best cover).

### Three mechanics

1. **Forest blocks line of sight** (the mismatch above, repaired). The LOS blocker becomes the
   union of every class with `los=True`, which under v2 is urban plus forest.
2. **Urban is emplaceable**, with the self-masking fix: a site is never masked by the polygon it
   stands in. Without this, a site inside a blocker starts its own sightline inside the blocker
   and can never engage anything.
3. **Concealment buys persistence.** A site on REVEALING ground (open, field) that engages the
   flight becomes visible to the defender from the next serial. A site on concealed ground
   (forest, urban) never does. "Engages" is defined as EXPOSURE, not a kill: the site is revealed
   if the flight passed inside its ring on that serial with line of sight, whether or not the
   interception succeeded (the flight would know it had been engaged). Memory is the gen34 form:
   a decaying average over the episode (decay 0.8), reset per episode.

### Back-compatibility (binding)

Every change is additive and defaults to v1, so gen31, gen32 and gen33 reproduce byte-identically
with the new table switched off. v1 stays the module default. Suite green with raw output pasted
at the build record.

### What this changes about the game (prediction, recorded before measurement)

The enemy gains a portfolio decision: buy visible coverage that works now and gets routed around
from the next serial, or buy concealed coverage that is weak but never stops working. A pure
open-ground force should decay across serials; a pure hidden force should be flat and low; the
best force is expected to be a mixture in which visible teams push the flight off the good lanes
and hidden teams wait where a pushed flight goes. That is exactly the bait/block/anchor structure
the gen33 model already emits unprompted, which is the principal reason to add the mechanic.

---

## Baseline family (PRE-REGISTERED, non-negotiable, all arms get the SAME observations)

The revealed-site channel is given to EVERY arm, not only to SACRED. Handing the policy an
information advantage the rules do not have would manufacture the result; this project has been
bitten by that recursion twice (the disjoint-route finding; the aerial full-menu rows).

- **avoid-revealed** (two lines: fly the menu route minimising damage from the currently-known
  sites);
- **avoid-revealed + anti-repeat** (the composed rule; the gen27 pattern);
- **anti-repeat / rotation over lanes and the full menu** (payoff-blind, the standing family);
- **static equilibrium mixture** and the multi-start **static local optimum** (the caps every
  static object is bounded by);
- **doctrine-fitted rules** (oracle-fitted, disclosed as caps);
- **exact best possible adaptive play** (Karp minimum-mean-cycle over the window MDP), wherever
  the enemy's rule is fixed.

---

## Step 1: the screen (ORACLE-ONLY, FREE, runs without a go)

Sweep, on the committed Kaliningrad->Gvardeysk vec theatre, over >= 6 threat fields:

- **number of teams** K in {1, 2, 3, 4, 6};
- **global range multiplier** in {0.7, 1.0, 1.3};
- **hidden lethality ratio** (a multiplier on forest/urban `p_max`) in {0.4, 0.6, 0.8, 1.0}, i.e.
  how much of the visible classes' lethality the concealed classes retain;
- **forest LOS** on/off (the sensitivity row).

Per cell, compute exactly: the static cap, the best payoff-blind dynamic rule, the best
avoid-revealed rule, the best doctrine-fitted rule, and the exact optimum.

> **DECISION RULE (fixed before looking).** The operating point is the cell with the largest
> **G2 = best simple rule / exact optimum**, subject to: G2 >= 1.25; G1 = static cap / optimum
> >= 2.0 (static play genuinely capped); values non-degenerate (inside 0.02-0.9) and not
> saturated (the naive rule not collapsing onto the deterministic value); defender leader entropy
> materially below uniform. **If NO cell clears G2 >= 1.25, the act stops here** and the finding is
> reported: with concealment implemented, avoiding what you have seen is near-optimal at every
> coverage on real terrain. That is a clean measured boundary and it costs a day of free compute.
>
> **Deliverable regardless of outcome:** the curve of G2 against team count and against hidden
> lethality ratio. That is the aerial twin of the roads K-boundary map and is thesis material on
> its own.

**Recorded prediction (may be wrong):** the interesting band sits where concealed lethality is
roughly half to two thirds of visible lethality; below that the avoid-revealed rule wins because
hidden teams cannot hurt; above it, hiding dominates and the game collapses to one terrain class.

---

## Step 2: three composers at the screened point (needs Kilian's go: shared GPU box)

The model is asked for DOCTRINE and ROLES only: per team, the behaviour mix (punish pattern /
anticipate flight / hold static), decisiveness, memory, posture (concealment vs reach; forward vs
deep), team role, and a rationale. **It is not asked for coordinates.** An algorithm then places
each team on the best site consistent with its stated posture.

Three arms, identical budget, terrain, weapons table and placer, so the only difference is the
composition:

- **llm** (llama-3.3-70b and qwen3-27b; per-model reporting, never pooled);
- **random** (doctrine mixes drawn uniformly on the simplex, roles random);
- **heuristic** (the pinned gen32 doctrine replicated across the teams).

> **BAR:** the model's forces induce more damage against a best-responding defender than the
> heuristic on >= 2/3 fields AND pooled, AND above the random arm's population mean, per model.
> **REPORTED:** the random floor and the oracle-searched best force as the ceiling; **implied
> coverage** per force (total engageable area after LOS shadows) as the fairness column, with a
> matched-coverage row if the arms differ materially; diversity (archetype and doctrine spread).
> **CONTROL (binding, moved from placement to doctrine):** relabelling the terrain in the brief
> must materially change the doctrine the model writes; if it does not, the terrain-reasoning
> claim is not licensed and the act re-scopes to "composition without terrain content", exactly as
> gen33 did.
> **Fail branch:** llm ~ random = "structure matters, the model is not needed to produce it", the
> gen37 shape, reported plainly.

---

## Step 3: the defender (needs Kilian's explicit go)

Three training arms at the screened operating point, identical budgets and seeds: SACRED trained
against the llm population, against the random population, and against the single hand-tuned
doctrine. Held-out test = enemies never trained against, drawn from all three families plus the
oracle-searched best force, on pristine threat fields.

> **PRIMARY (draft; pinned at the launch record): the llm-trained defender is below BOTH controls
> on >= 4/6 held-out cells AND pooled, on >= 2/3 seeds, at the validation-selected checkpoint.**
> **REPORTED as standing:** the untrained-network context row; a blinded arm (revealed-site channel
> zeroed) to prove the gain is the conditioning; the worst-case row against an enemy that simply
> commits; per-cell values with no averaging-away.
> **Fail branches, all writable:** arms indistinguishable = "curriculum composition does not
> transfer at this scale", which finally answers gen33's open metric 2; llm worse = reported
> plainly.

---

## Step 4 (OPTIONAL, only if 1-3 land): the enemy commander

Every 10 serials the model reads a short after-action report (lanes used, what was engaged, what
was missed) and re-issues the force's doctrine and roles; all per-sortie aiming stays algorithmic,
so this is one call per ten serials rather than one per decision. **Disclosed cost: once the
enemy's rule changes, the exact optimum stops existing, so this runs with per-phase yardsticks and
arm-versus-arm comparisons, as a demonstration, never as a bar-carrying headline.**

---

## Design decisions ledgered (one line each)

1. Ranges pinned as RATIOS with one swept multiplier: the relative capabilities come from physics
   reasoning, the absolute difficulty from the screen, so no range was chosen to produce a result.
2. Reveal on EXPOSURE not on kill: a flight knows it was engaged; tying reveal to a successful
   interception would make the channel vanish exactly where the enemy is most effective.
3. Reveal memory = the gen34 decaying average, reset per episode: a proven form, not a new one.
4. Concealed classes never reveal, binary rather than probabilistic: the cleanest version of the
   mechanic to explain and to control; a probabilistic variant is recorded as future work.
5. The revealed-site channel is given to every baseline arm (see the family above).
6. Urban emplaceable requires the self-polygon LOS exemption; without it urban sites are dead.
7. v1 remains the default table; gen31/32/33 reproduce byte-identically with v2 off.
8. New numbers under v2 may NEVER be mixed with gen31/32/33 numbers: v2 is a different game and
   re-anchors its own baseline family (the standing never-compare-across-game-states rule).
9. Theatre convention inherited from gen33: kgd primary, ukraine and narva held out.
10. Numbers live only in this ledger and its JSONs; thread pools capped on any multi-process launch.

## Compute envelope

Step 1: oracle only, seconds to a few minutes per cell, hours total, no model calls, no training.
Step 2: ~50-100 model calls (tens of minutes on the workbench) plus cheap oracle scoring.
Step 3: 9 runs (3 arms x 3 seeds) at the gen32/33 scale, roughly one night at 9-way parallelism,
launched DETACHED (the gen33 metric-2 batch was killed by an external signal at 1/8 budget).
Step 4: optional, one further batch.

---

## RESULTS (appended per step; nothing above changes after launch)

### BUILD RECORD (2026-07-25; env only, no screen, no model calls, no training)

**Landed** in `src/envs/aerial_theatre_vec.py`, all additive: `TERRAIN_V2` + `terrain_v2(hidden_leth,
forest_los)`; `blocker_union` (union of every class with `los=True`, cached per blocking-class set);
`containing_blockers` (the self-polygon LOS exemption); `reveal_flags`; `route_survival` gains
`terrain`, `own_polys`, `return_exposed`; `hazard_sites`, `build_terrain_menu`, `_threat_field`,
`engagement_footprint`, `build_theatre_game` gain a `terrain` pass-through (+ `return_cls`).

**A back-compatibility trap the tests caught, recorded because it is the sharp end of the gen39
finding.** The first implementation derived the LOS blocker from the terrain table. That silently
changed v1, because the v1 table DECLARES `forest: los=True` while the v1 code masked with the
urban union only. Repaired: `terrain=None` reproduces the implemented v1 behaviour (urban only)
and honouring the declared flags is opt-in. `tests/test_gen39_terrain.py` guards this with a
verbatim copy of the pre-gen39 `route_survival` as a byte-identity oracle, plus an explicit test
that documents the declared-versus-implemented gap rather than papering over it.

**Suite: 235 passed** (`PYTHONPATH=. pytest tests/`, 47 s; 224 before, +11 new gen39 tests).

**Plumbing check (oracle-only, K=1 static, kgd, spacing 2.5 km, standoff 4 km, R=18, ~1 s each):**

| table | sites | by class | eq | det | revealing |
|---|---|---|---|---|---|
| v1 (banked) | 116 | open 55, field 34, forest 27 | 0.215 | 0.784 | 116/116 |
| v2 | 126 | open 55, field 34, forest 27, **urban 10** | 0.193 | 0.609 | **89/126** |
| v2, hidden_leth 0.6 | 126 | same | 0.193 | 0.609 | 89/126 |

**Two readings, both expected and both worth pinning before the screen runs.** (a) Urban
emplacement adds 10 sites and the concealed classes are 37 of 126, so the reveal split is live.
(b) **The hidden-lethality knob does nothing to the static game**, and it should not: in a
one-shot solve nothing is ever revealed, so a concealed site is simply a weaker site and the
attacker's best single emplacement stays on open ground. The concealment mechanic is invisible by
construction in the static register and can only bite ACROSS serials. The screen is therefore run
in the dynamic register throughout, as the gates already specify; the static rows above are a
plumbing check, not an aiming metric. Recorded here so that no later reader mistakes the flat
static row for a null result.

**Not yet built (next tranche):** the reveal channel in the dynamic game (exposure -> decaying
observation, gen34 form), the avoid-revealed rule and the rest of the baseline family, and the
step-1 sweep script.

### STEP-3 LAUNCH LOCATION AND SEED COUNT, PINNED BEFORE LAUNCH (2026-07-25, Kilian's call)

**Step 3 runs LOCALLY on the M4 (10 cores), 3 seeds x 3 arms, 8000 sorties, launched DETACHED.**
Pinned here before any training starts, because gen33's batch was launched in a place its ledger
did not pin and the deviation had to be disclosed afterwards.

The alternative considered and declined was w05 (2x RTX A6000, 251 GB RAM, shared). The estimate
that informed the call, stated so a reader can check it: w05 is **not** materially faster for this
batch. Nine runs on ten local cores already give each run about one core, so the wall-clock is set
by SINGLE-RUN speed, not by core count, and a server core is not quicker than an M4 performance
core for these small graphs (the GPUs are irrelevant: CPU already beat the Mac's GPU by 2.4-4x on
this workload, and the A6000s are serving the two language models). Measured local rate, from the
gen33 attempt: 42 min / 1000 sorties at 9-way contention, so **~5.6 h for the full batch**;
w05 was estimated at 4-6 h. The genuine w05 advantages were an uninterruptible host and the
headroom for 5 seeds at the same wall-clock; **3 seeds is therefore the pinned n, with the thinness
disclosed** (gen33's preliminary read failed on exactly this: 1/3 seeds met the bar). Any later
seed extension is a NEW pinned decision, never a response to the first three.

Step 2 (the three composers) needs no SSH and no tunnel: contrary to the HANDOVER note, the
workbench gateway on port 8080 is now reachable directly from the Mac (verified 2026-07-25,
both `llama-3.3-70b` and `qwen3-27b` answering `/v1/models`). Kilian's go for w05 is on record.

### YARDSTICK CORRECTION + STEP-3 SIZE, PINNED BEFORE LAUNCH (2026-07-25, Kilian's call)

**Yardstick (no environment change, an analysis re-cut only).** The screen used ONE denominator,
the omniscient optimum, for two different questions, and that silently made the concealment
mechanic unmeasurable: concealment's whole value is not being located, and the reference defender
was handed every emplacement before the run. Measured consequence, from the step-1b grid: hidden
laydowns scored 0.00-0.50 of open laydowns against the omniscient optimum, and the hidden-lethality
axis was inert (G2 3.39 vs 3.45 across 0.4-1.0). Two questions, two comparisons, from here on:

- **"Is there room for a learned policy?"** denominator = the omniscient optimum. UNCHANGED and
  unaffected by this correction: simple rules leave 3-4x (G1 ~4.0, G2 ~3.4) across maps, team
  counts, reach and lethality.
- **"Does concealment do anything, and should the enemy use it?"** compare the SAME defender with
  and without the exposure channel, and score the enemy's laydown choice against a defender that
  must observe. Measured: the channel is worth **1.24-1.70x** to the defender on open laydowns,
  **1.38-1.70x** on mixed, and **exactly 1.00x on hidden** (the mechanic, with an internal
  control: same map, same rules, only the ground changes). Under that scoring hiding rises from
  ~0 to **0.59-0.80 of open ground**: competitive, not yet preferred.

**Wording rule (binding).** Concealment is "competitive, not preferred" at this defender skill.
It pays only insofar as being seen costs the enemy, which depends on how well the defender
exploits a sighting; our observing defender is the best SIMPLE rule, so these are a FLOOR on
concealment's value, never a ceiling. The crossover is a SACRED-dependent quantity and is not
claimed from oracle rows.

**Rejected alternative, recorded with its reason.** A belief-MDP yardstick (enemy draws from a
shortlist of laydowns, defender holds the consistent subset) was proposed and rejected by Kilian:
it is an oracle construct, SACRED does not reason over beliefs, so it would have changed the
scorekeeper and not the agent. The yardstick re-cut above achieves the measurement with no code
change.

**Step 3 size: TWELVE runs (was nine).** 3 curricula x 3 seeds, PLUS a blinded arm (exposure
channel zeroed) at 3 seeds on the arm carrying the primary claim. The blinded arm was already a
standing reported row; it now also carries the concealment claim, because sighted-vs-blind IS the
measurement of what the channel is worth to a TRAINED policy, and therefore of whether hiding
becomes the enemy's better choice against a good defender. Not blinded across all three curricula
(18 runs): that would answer a second-order question we have no reason to ask. Local, detached,
8000 sorties; measured local rate 42 min/1000 sorties at 9-way => **~7.5 h at 12-way**.

### STEP 1 RESULTS (2026-07-25; `models/runs/gen39_screen2.json`, 8640 cells, oracle-only, free)

Grid: 4 maps x 3 range multipliers x 3 concealed reaches x 5 team counts x 4 laydown archetypes x
4 hidden-lethality settings x 3 fields. Every cell scored under BOTH defender memories in one
build (`scratch/gen39_screen2.py`, block-parallel, 165 min on 8 workers). Read-outs:
`scratch/gen39_read_{screen,scoping,operating_point}.py`.

**VALIDATION.** The forgetful column reproduces the independent serial screen (`gen39_screen.py`,
run to completion separately) on all 8640 overlapping cells: **0 value mismatches, largest absolute
difference 0.00e+00**. The `episodic` hoist is byte-identical to the pre-hoist loop across 3
laydowns x 20 rule variants. Suite 235 green.

**G1. Room for a learned policy (vs the omniscient optimum) - PASSES, and is unaffected by the
yardstick correction.**

| map | cells | a real game | G1 cap/opt | G2 rules/opt |
|---|---|---|---|---|
| kgd_gvardeysk | 2160 | 89% | 4.59 | 3.47 |
| ukraine | 2160 | 84% | 3.52 | 3.37 |
| narva | 2160 | 32% | 1.65 | 2.07 |
| fulda | 2160 | **0%** | - | - |

Both pre-registered gates (G1>=2.0, G2>=1.25) pass on **87% of non-degenerate cells**.

**G2. Concealment denies the information channel - the mechanic works, with an internal control.**
Same defender, same map, same rules; only the ground the enemy sits on changes:

| enemy sits in | flying blind | using what it saw | sight is worth |
|---|---|---|---|
| open | 0.1334 | 0.1059 | **1.61x** |
| mixed | 0.1281 | 0.0936 | **1.62x** |
| random | 0.1418 | 0.1158 | 1.12x |
| concealed | 0.0842 | 0.0842 | **1.00x (nothing)** |

**G3. The crossover: hiding is the better force design against a defender that must observe.**
Median hidden-laydown damage as a share of open-laydown damage, all cells (degenerate INCLUDED:
"a knowing defender can walk around it" is a real property of a short-ranged concealed force, not
a missing datum). The split by map is material and is NOT pooled:

| map | concealed reach | vs an omniscient defender | vs a defender that must observe |
|---|---|---|---|
| kgd_gvardeysk | 0.43 / 0.65 / 0.85 | 0.07 / 0.66 / 0.72 | 0.93 / **1.10** / **1.23** |
| ukraine | 0.43 / 0.65 / 0.85 | 0.20 / 0.32 / 0.44 | 0.50 / 0.54 / 0.53 |

**Memory (the 2026-07-25 fix) is faithful but is NOT where the value is.** Paired, same cell:
optimum 1.00x, best blind rule 1.00x, best observing rule 1.08x, G2 1.03x. Recorded plainly: the
whole-mission form is kept because it is the correct model of the mechanic, not because it moved
the numbers. The earlier single-cell reading (G2 2.67 -> 3.85) did not survive the full grid.

**SCOPING NEGATIVE (binding on every later claim).** **Fulda is not a game at any swept setting:
100% of its 2160 cells are degenerate (median optimum 0.00024), flat in team count (K=1 0.00032 ->
K=6 0.00059).** Narva is degenerate in 68% of cells and its concealed laydowns survive in 2%. The
coverage fraction phi is ~0.51-0.55 on ALL FOUR maps, so **phi does not capture playability**: on
the long, wide corridors the route menu spans more ground than <=6 concentrations can threaten and
the defender always finds a free lane. **The scored theatres for gen39 are kgd_gvardeysk (primary)
and ukraine (held out).** Narva and Fulda are reported as a measured scoping boundary with this
mechanism, never as failed runs, and no gen39 claim is made on them.

**OPERATING POINT (pinned for steps 2-4): kgd_gvardeysk, K=3 teams, concealed reach 0.85,
range multiplier 1.3, hidden lethality 0.8.** At that cell: 100% of laydowns are a real game,
G1 4.44, G2 3.93, sight worth 1.54x, and - the reason for this cell over its neighbours - **the two
force designs are exactly equally dangerous to an omniscient defender (hidden/open = 1.00) while
hiding is 1.65x better against one that must observe.** Raw firepower is matched by construction,
so everything that separates the two designs at this point IS the information channel. Chosen on
the pre-registered rule that ranges are pinned as ratios with the absolute difficulty set by the
screen. Alternative recorded, for a stricter physical story on reach (forest at 0.65 of open):
cr 0.65 / rm 1.3 / hl 0.6, omniscient 0.93 and observing 1.47.

### CORRECTION TO THE STEP-1 SCOPING NEGATIVE (2026-07-25, same day, before any step-2 launch)

**The mechanism stated above for narva/fulda ("the route menu spans more ground than <=6
concentrations can threaten and the defender always finds a free lane") is WRONG.** It was inferred
from the coverage fraction, not measured. Measured: on fulda only **1 of 25 routes** is genuinely
free, so the defender is not walking around the force.

**The real cause is the forest line-of-sight rule interacting with how wooded the theatre is.**
Table v2 makes forest opaque (Kilian's 2026-07-25 call, "for forest, implement it"). Same map, same
laydowns, forest opaque vs transparent, best optimum over 6 laydowns (`scratch/gen39_why_dead.py`):

| map | concealable ground | forest opaque | forest transparent | ratio | sight lines blocked |
|---|---|---|---|---|---|
| kgd_gvardeysk | 26% | 0.07033 | 0.15144 | 2.2x | - |
| ukraine | 12% | 0.04430 | 0.04627 | **1.0x** | 31% |
| narva | 64% | 0.00898 | 0.02836 | 3.2x | 47% |
| fulda | 66% | 0.00173 | 0.02311 | **13.3x** | **79%** |

Ukraine (12% wooded) does not move; fulda (66% wooded) moves 13-fold. On fulda the single most
dangerous route in the theatre costs 6% of the fleet where kaliningrad's costs 53%: there is
nothing at stake, which is why the team-count row is flat (K=1 0.00032 -> K=6 0.00059). **Deadness
tracks wooded fraction, not map size or team budget.** The playability boundary sits between 26%
(plays) and 64% (does not).

**Open design question for Kilian, not decided here.** The v2 rule blocks sight SYMMETRICALLY:
a team in woodland is invisible AND blind. The physical asymmetry is that canopy hides a ground
team from an aircraft looking down but does not hide an aircraft above the treeline from a team
looking up. An asymmetric forest rule (hides the team, does not blind it; urban unchanged, since
buildings are true vertical obstacles) would revive narva and fulda, make concealment purely an
information-channel mechanic (which is what this act claims), and stop penalising cover twice.
It is a DIFFERENT GAME: nothing measured under symmetric forest may be mixed with it (rule 8).
Until Kilian decides, the scored theatres remain kgd (primary) + ukraine (held out) and the
operating point above stands.

### DESIGN CHANGE: FOREST HIDES WITHOUT BLINDING (2026-07-25, Kilian's call; SHA at the commit)

> "With modern radar equipment the team in the forest should only hide it, not blind it."

Table v2 blocked sight SYMMETRICALLY, so a team in woodland was invisible AND blind. That is not
the physics (canopy conceals a ground team from an aircraft looking down; it does not stop a
radar-cued crew engaging an aircraft above the treeline) and it is what killed two theatres.
**`TERRAIN_V2["forest"]["los"]` True -> False; `reveal` stays False. Urban keeps both, because
buildings are genuine vertical obstacles.** `terrain_v2(forest_los=True)` restores the symmetric
rule as the disclosed sensitivity row.

**Rule 8 applies with full force: the symmetric and asymmetric tables are DIFFERENT GAMES.** Every
number in "STEP 1 RESULTS" above was measured under the symmetric rule and is superseded, not
amended. Those artefacts are archived as `models/runs/gen39_screen{,2}_symforest*` and keep their
own cross-check (0 mismatches on 8640 cells). **The pinned operating point above is VOID** and is
re-picked from the re-run. Nothing from the two tables may appear in one figure or one ladder.

**The LLM brief is repaired in the same commit,** because it carried the same conflation and is the
likelier of the two candidate causes of gen33's failed terrain control: `_physics_table_text`
derived "it conceals you (blocks line of sight)" from the single `los` flag, so the model was told
woodland concealed it (which the simulator did not implement) via a flag that actually means
sight-masking. Hiding and sight-blocking are now stated separately, and `serialise_theatre` takes
the table in force (defaulting to v1). **CORRECTION 2026-07-25 (Kilian's call: remove the claim):**
the v1-default brief text is NOT byte-identical to the brief gen33 actually sent (the wording
changed in this same commit, and the new v1 default describes the reveal mechanic, which v1 does
not implement); gen33's reproducibility record is its stored transcripts, and any regeneration run
would use the new text. Three contract tests pin the asymmetry, the default blocker set and the
brief wording; suite 235 -> 238 green.

**Re-run launched** (same grid, same harness, ~2.5 h). Prediction to be checked against it, on
record before the numbers land: narva and fulda should become playable, since the mechanism was
measured at 13.3x on fulda and 3.2x on narva, while ukraine (12% wooded, 1.0x) should barely move.

### STEP 1 RESULTS, RE-RUN UNDER THE ASYMMETRIC FOREST RULE (2026-07-25; supersedes the block above)

Same grid, same harness, 8640 cells, 139 min on 8 workers. `models/runs/gen39_screen2.json`.
**The prediction recorded before launch was borne out**, which is the reason to trust the
mechanism rather than the story: narva and fulda revive, ukraine barely moves.

| map | wooded | a real game, symmetric -> asymmetric | G1 | G2 |
|---|---|---|---|---|
| kgd_gvardeysk | 26% | 89% -> **91%** | 4.21 | 2.84 |
| ukraine | 12% | 84% -> **91%** | 5.07 | 3.88 |
| narva | 64% | 32% -> **66%** | 3.77 | 3.33 |
| fulda | 66% | **0% -> 81%** | 1.85 | 2.18 |
| ALL | | 51% -> **82%** | 3.95 | 3.26 |

Gates pass on 78% of real cells, i.e. **64% of the whole grid** (was 45%). All four theatres are
scored; the symmetric-forest scoping negative is retired as a property of that table, not of the
maps, and is kept as the disclosed sensitivity row.

**The mechanic, unchanged by the repair and still with its internal control** (same map, same
rules, only the ground the enemy sits on changes): sight is worth **1.58x** to the defender against
an open force, **1.59x** against a mixed one, and **exactly 1.00x** against a concealed one.

**The crossover is now general rather than marginal.** Hidden-laydown damage as a share of
open-laydown damage, all cells (degenerate included):

| concealed reach | vs an omniscient defender | vs a defender that must observe |
|---|---|---|
| 0.43 | 0.23 | **1.02** |
| 0.65 | 0.54 | **1.10** |
| 0.85 | 0.71 | **1.13** |

Per map (honest form, all cells): kgd 1.42/1.66/1.41 and narva 1.42/1.37/1.37 cross at every
reach; **ukraine does not (0.65/0.72/0.73)** and it is the least wooded theatre at 12%, so the
scarcity of cover is itself the explanation. The split is reported, never pooled away.

**Memory** (paired, same cell): optimum 0.99x, blind rule 1.02x, observing rule 1.08x, G2 **1.09x**.
Slightly larger than under the symmetric table but still small; the whole-mission form is kept
because it is the correct model, not because it moved the numbers.

**OPERATING POINT (pinned for steps 2-4, replacing the voided one):
`kgd_gvardeysk, K=3 teams, concealed reach 0.85, range multiplier 1.0, hidden lethality 0.4`.**
Detail in `scratch/gen39_pinned_cell.py`; selected by `scratch/gen39_pick_operating_point.py` from
720 candidate cells on: >=90% of laydowns a real game, G1>=2, G2>=2, sight >=1.4x, then ranked by
closeness of the omniscient hidden/open ratio to 1.00 (matched firepower) against the size of the
observing ratio (the information gap).

| | value |
|---|---|
| laydowns that are a real game | 100% (12/12) |
| G1 (static cap / optimum) | 4.53 |
| G2 (best simple rule / optimum) | 3.58 |
| sight worth to the defender | 1.74x |
| hidden vs open, omniscient defender | **1.02** |
| hidden vs open, observing defender | **1.61** |

The two force designs are within 2% of equally dangerous to a defender that already knows where
they are, and hiding is 61% better against one that must find out. **Firepower is matched by
construction, so what separates them at this point IS the information channel and nothing else.**
Range multiplier 1.0 (no range inflation was needed, unlike the voided symmetric pick at 1.3).
Physical reading: a concealed team engages nearly as far (2.98 km vs 3.5 km) at roughly a quarter
the kill probability (0.22 vs 0.90), and those two effects cancel against a knowing defender.

### THE CROSSOVER WAS AN ARTEFACT: the concentration leaked across terrain (2026-07-25, Kilian's catch)

Kilian asked whether a team told to sit in woodland could end up delivering its effect from
neighbouring grassland. It could, and it dominated the result.

The engagement concentration (sigma = 1.5 x the team's own reach, i.e. ~4.5 km on kaliningrad
against a 2 km site grid) was spread over EVERY nearby candidate site regardless of ground, while
`reveal` reads only the team's OWN site. Measured share of a team's effect delivered from its own
ground class (`scratch/gen39_leak_probe.py`):

| team on | kgd own-ground share | ukraine |
|---|---|---|
| open | 55% | 60% |
| field | 42% | 54% |
| **forest** | **20%** | **7%** |
| **urban** | **16%** | **21%** |

**A "concealed" team drew 60-65% of its reach and lethality from OPEN ground while keeping
woodland's invisibility.** That diluted the price of concealment roughly fivefold and inflated
every hide-vs-open number in the block above.

**With the leak removed (`same_class=True`, now the DEFAULT), the crossover disappears entirely.**
At the (now void) pinned point, K=3, 3 fields:

| map | | hidden/open, omniscient | hidden/open, observing |
|---|---|---|---|
| kgd | leaked -> masked | 1.16 -> **0.00** | 1.62 -> **0.13** |
| ukraine | leaked -> masked | 0.77 -> **0.02** | 0.81 -> **0.12** |

**And no setting rescues it** (`scratch/gen39_conceal_rescue_probe.py`, kgd, K in {3,6},
concealed reach up to **1.00** i.e. equal to open, lethality multiplier up to 1.0): the omniscient
optimum against an all-concealed force is **0.0000 in all 12 cells**, and the observing ratio peaks
at **0.87**, still below parity, at the most generous setting we can write down.

**The mechanism, and it is a real finding rather than a bug.** Woodland occupies 38 of 200
candidate positions on kaliningrad and is patchy: **cover does not span the corridor.** A force
confined to cover therefore always leaves a lane, and a defender that finds the lane takes zero
damage. Concealment cannot buy interdiction on real terrain when the cover is fragmented, however
generous its weapons. This bounds the mechanic by geography, not by the lethality/reach trade we
had been sweeping.

**Consequences, stated before the re-run lands.** Everything in "STEP 1 RESULTS, RE-RUN UNDER THE
ASYMMETRIC FOREST RULE" was measured with the leak and is superseded; the operating point pinned
there is VOID for the second time. Leaked artefacts archived as `*_leaked*`. The claims most at
risk are the crossover (expected to die outright) and any hide-vs-open row; the claims expected to
survive are the room-for-a-learned-policy gates and the information-channel control (sight worth
~1.6x against a revealing force and exactly 1.00x against a concealed one), since neither depends
on the concealed force being competitive. **The better-posed question the re-run must answer is not
all-open vs all-concealed but the MIX**: whether a force that blocks the corridor with open teams
while keeping concealed teams for persistence beats both pure designs. That is also precisely the
composition judgement step 2 asks the model to make. Suite 235 -> 240 green; re-run launched.

### VERIFICATION WITH ALL THREE REPAIRS (2026-07-25; `scratch/gen39_verify_fixes.py`, no screen re-run)

Repairs in force: (1) the engagement concentration stays on the team's own ground; (2) candidate
class shares match the theatre's composition on a fixed 200-point budget (Kilian's quota scheme,
whole-map shares); (3) the force is the best COMBINATION of K positions, with the old
K-best-points picker kept as a competing candidate and both scored exactly.

| map | K | force | vs perfect play | blind rule | observing rule | sight worth |
|---|---|---|---|---|---|---|
| kgd | 3 | open | 0.0968 | 0.5015 | 0.4564 | 1.10x |
| kgd | 3 | concealed | 0.0123 | 0.0622 | 0.0622 | **1.00x** |
| kgd | 6 | open | 0.0980 | 0.3763 | 0.2340 | 1.61x |
| kgd | 6 | concealed | 0.0080 | 0.0975 | 0.0975 | **1.00x** |
| ukraine | 3 | open | 0.0627 | 0.2315 | 0.1785 | 1.30x |
| ukraine | 3 | concealed | 0.0027 | 0.0741 | 0.0741 | **1.00x** |
| ukraine | 6 | open | 0.0626 | 0.2394 | 0.1730 | 1.38x |
| ukraine | 6 | concealed | 0.0024 | 0.0747 | 0.0747 | **1.00x** |

**THE CONCEALMENT HEADLINE IS DEAD, and now on defensible methodology.** Concealed forces reach
0.04-0.14 of an open force against perfect play and 0.14-0.43 against a defender that must observe.
There is no crossover at any team count on either scored theatre. Decisively: a force search given
a FREE choice over all four terrain classes ("mixed" rows) returns an all-open force on every map
and every K, matching the open row to four decimals. **Given the choice, the best force our search
can find never uses cover at all.**

**What survives, and it is the mechanism rather than the headline.** The information-channel
control is unaffected and remains exact: sight is worth **1.10-1.61x** to the defender against a
force on revealing ground and **exactly 1.00x** against a concealed one, on every map and team
count. Concealment does precisely what it was designed to do (it shuts the channel completely); it
simply is not worth its price in reach and lethality on real terrain.

**Disclosed limitation, because it cuts the other way.** Under proportional sampling both classes
get the same points-per-km2, which is fair, but cover is a small share of the ground, so a
concealed force chooses from ~34 options on kgd where an open force chooses from ~76. A
higher-density sensitivity run (stratified top-up, ~146 forest sites) moved the K=6 observing ratio
from 0.42 to 0.87, i.e. **the concealed force's quality is sampling-limited in a way the open
force's is not, and the gap narrows with density without closing.** Any write-up of this negative
must carry that row.

**RETRACTED with it:** the "cover does not span the corridor / ukraine's cover is dust" mechanism.
It compared patch AREA against weapon FOOTPRINT, which are unrelated quantities: a MANPADS needs
somewhere to stand, not somewhere to fit its engagement circle (Kilian). Patch area plays no part
in the current sampler and no part in this result.

### SAMPLER BUG (scan-order ties) + WHAT CONCEALMENT COSTS (2026-07-25; VOIDS the block above)

**Bug, found by inspecting the site figures Kilian asked for.** In `quota_sites` every anchor
standing INSIDE a patch ties at distance zero, and the tie broke on anchor index, i.e. raster scan
order. Result: **every kaliningrad candidate sat in the left 27 km of a 45 km theatre** (open
x-span 1.0-13.2, forest 7.1-26.7, nothing beyond x=27). Fixed by farthest-point selection among
each class's eligible anchors; all four maps now span their full extent. The anchor grid also
sizes itself from the budget: fulda's 11.6 km grid gave only 87 anchors against a 200-point quota,
so OPEN GROUND received ZERO points despite being 23% of that theatre.

**The verification block above is VOID: it was measured on a candidate set confined to half the
map.** With the repair the concealment verdict is a near-miss, not a death.
`scratch/gen39_conceal_cost.py`; full output `results/gen39_conceal_cost.txt`; figures
`assets/gen39_sites_<map>.png` (`scratch/gen39_site_map.py`).

Share of what an OPEN force achieves, against a defender that must observe:

| concealed force given back | KGD K=3 | KGD K=6 | UKR K=3 | UKR K=6 |
|---|---|---|---|---|
| nothing (table as pinned) | **58%** | **65%** | 28% | 37% |
| open REACH | 68% | 75% | 29% | 59% |
| open LETHALITY | **87%** | **95%** | 45% | 68% |
| both | 98% | 112% | 55% | 89% |
| *control: OPEN force, same few positions* | *80%* | *84%* | *87%* | *84%* |

**On kgd the ~40% cost splits into two separable halves.** (a) WEAPONS ~22 points, within which
**lethality is the expensive charge and reach the cheap one**: paying reach back alone buys 10
points, lethality alone buys 29. (b) CHOICE ~20 points, isolated by the control row: an OPEN force
cut to 53 positions loses 20 points by itself, so half the penalty is that there is less cover to
choose from and has nothing to do with cover being worse. Ukraine splits the other way (control
84-87%, so weapons dominate); it holds 26 concealed positions across 46 x 90 km.

**Break-even sweep (share of the open force, observing defender), kgd K=3:**

| concealed reach | leth 0.55 | 0.70 | 0.90 |
|---|---|---|---|
| 0.85 of open | 58% | 73% | 87% |
| 1.00 | 68% | 84% | **98%** |
| 1.20 | 84% | 93% | **104%** |
| 1.50 | **100%** | 126% | 147% |

**RECOMMENDATION, recorded but NOT applied (awaiting Kilian): raise concealed LETHALITY, keep the
reach penalty.** Equal reach at lethality 0.70 gives 84% (K=3) and 94% (K=6): near parity, so the
enemy's choice stays genuinely open. Handing back both (98-112%) makes hiding weakly dominant and
deletes the decision the act exists to study. The physical story fits that shape: an unseen team
shoots first with an unspoiled firing solution, so surprise buys KILL PROBABILITY while clutter
costs ENGAGEMENT ENVELOPE. Ukraine breaks even at no swept setting (best 80% at K=3), which is a
map-dependent finding rather than a knob.

### CODE REPAIRS BEFORE THE STEP-1 RE-RUN (2026-07-25 late, critic session; no screen re-run, no training)

An audit of the committed gen39 code (fresh critic instance, Kilian's blanket approval for the
bug fixes, in-conversation) found and repaired the following. Every prior number stands where its
script was sound; the cost table above gains three disclosed caveats and is RE-MEASURED before the
lethality decision.

1. **`gen39_screen2.py` could not run and finding 6 was never wired in.** `cell()` recorded an
   undefined `picker` (NameError on the first cell of every block: the planned re-run would have
   failed 36/36), and the force was still chosen by the OLD top-K picker alone, the exact fault
   finding 6 retired. Fixed: `choose_force` (now in `src/envs/aerial_conceal.py`, with
   `pick_laydown` moved beside it) scores BOTH pickers exactly (episodic optimum, T=40) per cell,
   keeps the winner and records which won; both screens use it. Per-cell cost rises (up to 4
   candidate games per cell); re-estimate the wall-clock from the first block before trusting the
   ~2.5 h figure.
2. **`gen39_screen.py` (the independent cross-check) was three games stale**: symmetric forest
   (`forest_los=True`), the raster sampler (no `n_sites`), the old picker. Aligned to the current
   game (asymmetric default, 200-site quota, `choose_force`); its symmetric-era artefacts remain
   archived apart (rule 8).
3. **`gen39_conceal_cost.py` (the decision table's source) carried three distortions,** so the
   table above is honest-but-caveated: (a) the terminal standoff did not scale with the map (its
   UKRAINE columns were measured at 4 km where every other gen39 artefact uses ~8.2 km: a
   different game; kgd unaffected at scale 1.0); (b) forces were selected by perfect-play damage
   but reported in the observing-defender matchup (now selected by the observing score, the
   reported column); (c) the lethality axis rebuilt the terrain table and therefore the route
   MENU per cell (now one base per reach, lethality applied via the score-time knob, menu
   frozen: the screen's convention). `gen39_verify_fixes.py` selection aligned likewise (its
   verdict was already void: the sampler bug).
4. **The "gen33 briefs reproduce verbatim" sentence is removed** (correction block above).
5. **Step-2 readiness:** `redforce.force_schema(terrain)` makes URBAN choosable under v2 (the
   frozen module constant, computed from v1 at import, excluded it); the v1 default is untouched.
6. **The `w=2` comment corrected:** it claimed the "gen32 pinned operating point" but gen32
   pinned w=3; w=2 is a deliberate cost choice, and the defender's memory of DISCOVERED teams is
   whole-mission in the persistent arm regardless (Kilian's requirement, already satisfied).
7. **SPOTTING FOLLOWS THE FIRE (decided by Kilian 2026-07-25, in-conversation, and implemented
   in the same session).** The old trigger read only the team's NOMINAL site while its fire is
   delivered from its whole same-class zone (it relocates between serials), so a team on
   revealing ground could engage a flight whose track never entered the nominal ring and stay
   unspotted: the within-class remnant of the fault-4 leak, free invisibility on open ground,
   biasing every hide-vs-open number AGAINST concealment. New rule (Kilian's, stated in his own
   words: static within a serial, relocates between serials, engages only from where it sits,
   spotted exactly when it can engage, and only THAT team is spotted): a team is revealed when
   the flight comes within range of any position carrying >= 5% of its peak concentration
   weight; spotting reveals the team's operating ZONE (its `dmg_j` threat), so relocation within
   the patch does not stale the intel. Concealed ground still never reveals. Contract test:
   the new trigger is a superset of the old one and hidden teams stay dark.
   **RULE 8: spot-at-nominal-site and spot-where-it-fires are DIFFERENT GAMES; every number
   above this line was measured under the old trigger and may not be mixed with what follows.**

Suite 240 -> 243 green (choose_force never-worse; schema-follows-table; spotting-follows-fire).
The cost table + break-even sweep re-measure (repaired probe, new trigger) writes
`results/gen39_conceal_cost_v2.txt`, named apart from the void v1 numbers; the lethality-vs-reach
decision is taken from THAT table.

### COST TABLE v2, ALL FOUR MAPS (2026-07-26 early; repaired probe + spot-where-it-fires;
### `results/gen39_conceal_cost_v2.txt`; pool-parallel `gen39_conceal_cost.py`, serial path
### verified byte-identical on kgd K=3)

**The v1 verdict ("concealment never pays; ukraine breaks even at no swept setting") is
OVERTURNED by the honest probe.** Share of an open force's damage achieved by a concealed force
against an observing defender, at the PINNED table (concealed lethality 0.55, reach 0.85):

| map | cover share | K=3 | K=6 | options control |
|---|---|---|---|---|
| kgd_gvardeysk | 26% | 57% | 52% | 77% / 87% |
| ukraine | 12% | 82% | 69% | 113% / 112% |
| narva | 64% | 80% | 91% | 100% / 100% |
| fulda | 66% | **121%** | 91% | 100% / 100% |

**Concealment's value tracks the cover share of the map**: on fulda (66% wooded) hiding already
BEATS an open force at the pinned table; on narva it is near parity; on kaliningrad (26% cover)
it costs ~45%. The "choice" charge exists only where cover is scarce (kgd control 77-87%; 100%
on the wooded maps). The v1 ukraine columns (28-37%) were dominated by the three probe defects;
honest values are 82/69. Fulda/narva remain omniscient-degenerate for concealed forces in the
earlier sense retired with the symmetric table; against perfect play concealed forces now reach
0.9-1.0x of open on fulda (the woods are simply where the corridor is).

**RECOMMENDATION (firm, awaiting Kilian): concealed lethality 0.55 -> 0.70, reach penalty kept
at 0.85.** At that setting the observing-defender shares become kgd 70/60, ukraine 98/84,
narva 99/114, fulda 159/119: the choice is live on every map, open ground still favoured on the
primary (kgd), hiding favoured on the cover-rich maps, near parity between. Two supporting
arguments: (i) these numbers are a FLOOR (rule 3): a TRAINED defender exploits sightings harder,
which lowers only the open force's value, so a kgd measured at 70-84% vs simple rules can cross
parity vs SACRED, making the crossover a trainable, SACRED-dependent quantity: exactly the act's
question; (ii) the physical story stands (an unseen team's first shot has an unspoiled firing
solution: more kill probability than clutter-degraded 0.55, still below open 0.90 with restricted
arcs). The 0.90 alternative overshoots: hiding becomes dominant on three of four maps.
Keeping 0.55 is defensible too (the map-gradient already gives a live choice across theatres,
just not ON the primary); recorded as the fallback.

**Screen re-run re-timed on the repaired code (2026-07-26 early):** one full kgd block (240
cells, with the honest two-picker force selection) = 8.4 min serial, run to a SCRATCH folder
(no head start on the real outdir). Full 36-block grid at 10 workers: **roughly 30-50 min**
(was 139-165 min at 8 workers; the paired-memory hoists and cheap builds pay for the costlier
picker). The re-run itself still waits on the lethality decision.

### STEP-1 THIRD-RUN LAUNCH RECORD (2026-07-26, Kilian: "rerun step 1"; pinned BEFORE launch)

Code state: the repaired screen (`choose_force`, spot-where-it-fires, quota sampler,
asymmetric forest), SHA at this commit. **One disclosed grid change, decided before launch:
the hidden-lethality axis gains two points, 1.27 and 1.64 x the pinned 0.55 (= effective forest
lethality 0.70 and 0.90), so the operating-point pick can settle the raise-lethality question
the v2 cost table opened; the four pre-registered points are unchanged and the decision rule is
unchanged.** Grid: 4 maps x 3 range multipliers x 3 concealed reaches x 5 team counts x 4
archetypes x **6** lethality settings x 3 fields = 12,960 cells, both defender memories per
cell. The stale partial run found in the default outdir (8 blocks, 21:05-21:07 on 25-07, the
mid-session stop on the biased sampler) is archived as
`models/runs/gen39_screen2_stopped_biased_sampler`, never mixed. Launch: `--launch --workers
10`, detached, thread pools capped per process. Read-outs + operating-point pick follow the
standing scripts; the pick script now derives the lethality axis from the data.

### STEP 1 RESULTS, THIRD RUN (2026-07-26; 12,960 cells, 36/36 blocks ok, 220 min at 10
### workers; `models/runs/gen39_screen2.json`; the FIRST run on fully repaired code)

**Timing disclosure:** the pre-launch ~30-50 min estimate was WRONG by ~5x (it extrapolated
from a kgd block; the wooded-map K=6 blocks dominate). Actual: 220 min.

**G1. Room for a learned policy: PASSES on every map** (persistent arm; medians over real
cells): kgd 83% real, G1 3.17, G2 3.21; ukraine 67%, 4.28, 3.25; narva 79%, 3.95, 3.58;
fulda 91%, 2.91, 2.55. Gates (G1>=2, G2>=1.25) pass on 86% of real cells = 69% of the grid.

**G2. The mechanic, with its internal control, on the repaired game:** sight is worth 1.29x
(open laydowns), 1.37x (mixed), 1.26x (random) and **exactly 1.00x (concealed)**.

**Memory now matters (the spot-where-it-fires consequence):** whole-mission memory improves
the observing defender 0.89x vs the forgetful window (was ~1.08x under the old trigger);
optimum and blind rules unmoved (0.99-1.00x). The faithful memory form finally carries value,
as the mechanic intended.

**OPERATING-POINT SHORTLIST (pre-registered filter: >=90% real, G1>=2, G2>=2, sight>=1.4;
ranked by matched-omniscient-firepower vs observing-gap): every qualifying cell is on the
WOODED maps.** Top: **narva, K=3, concealed reach 0.85, range multiplier 0.7, hidden lethality
1.0 x the PINNED table** - real 92%, G1 3.65, G2 4.36, sight 2.08x, hidden/open 1.07 vs an
omniscient defender (firepower matched) and **2.66 vs one that must observe**. No kgd cell
passes the filter: on kgd the hidden force is 0.23-0.40x of open vs perfect play at EVERY
setting (cover is not where the corridor is), while still 1.7-1.8x better vs an observing
defender at raised lethality; kgd's best cell by the same ranking is K=3, cr 0.85, rm 0.7,
hl 1.64 (effective 0.90): real 92%, G1 4.52, G2 4.25, sight 1.61, omni 0.33, obs 1.78.

**THE DECISION THIS PUTS TO KILIAN (not taken autonomously; it moves the theatre convention):**
- **(A) RECOMMENDED: primary moves to NARVA at the pinned table** (K=3, cr 0.85, rm 0.7,
  hl 1.0; kgd + ukraine become held-out). Honours the act's own design principle that no
  range or lethality is chosen to produce a result: the terrain does the work, the lethality
  question dissolves (keep 0.55), firepower is matched by construction and everything that
  separates the two force designs IS the information channel, at the largest measured gap.
- **(B) Keep kgd primary at concealed lethality 0.90** (the surprise-buys-kill-probability
  ceiling): continuity with the gen32/33 Kaliningrad exhibits, but the story changes to
  "weaker-but-invisible vs stronger-but-seen" (omni 0.33), and the table is raised to make
  the primary map work: the knob the design principle warns against.

### STEP-2 IMPLEMENTATION PINS (2026-07-26, BEFORE any live model call; harness
### `scratch/gen39_compose.py`, dry-validated offline; Kilian's launch authority for steps 1-3
### granted in-conversation)

- **Primary scorer** = damage against the best OBSERVING defender (the avoid-revealed rule
  family under persistent memory, T=40): the matchup composition-with-concealment exists for.
  Omniscient optimum, blind value and implied coverage (share of menu routes engageable by any
  team's zone) reported beside it. Oracle ceiling = `choose_force` over the three archetypes
  with the single gen32 doctrine.
- **One placer for every arm**: highest-threat unused site of the team's stated terrain in its
  stated region (thirds along the axis); terrain fallback then any-unused; no site reused.
- **Doctrine map**: punish_pattern -> q_rep, anticipate_flight -> q_flee, hold_static -> q_hold
  (normalised); tau by decisiveness bin (0.05/0.10/0.20); memory clamped to the game's w=2.
  Per-team doctrines via the ConcealDyn extension whose identical-doctrine path reproduces the
  screen's enemy exactly (the tested regression anchor).
- **Heuristic arm** = the gen32 doctrine (0.6/0.2/0.3, tau 0.10, w 2) replicated over 3 teams,
  reach posture (open), regions spread across the three thirds. **Random arm** = 20
  compositions, doctrine simplex-uniform, postures uniform. **LLM arms** = 8 forces per model
  (llama-3.3-70b, qwen3-27b), temperature 0.8, guided JSON (v2 schema: urban choosable), exactly
  K=3 agents enforced.
- **Relabel control (binding)** = 8 more calls per model under a brief whose FOREST and OPEN
  rows swap characteristics (reach, lethality, reveal, LOS) in the text only; placement and
  scoring stay on the true table. The terrain-choice distribution must materially change or the
  terrain-reasoning claim is not licensed (the gen33 re-scope).
- **Fields** = the cell's own 5100-5102; pristine 61xx reserved for step-3 held-out tests.
  Call budget 32; scoring pool-parallel; results to `models/runs/gen39_compose/`.

### STEP 2 RESULTS (2026-07-26; 32/32 live calls valid, generation 1.6 min, scoring
### pool-parallel; `models/runs/gen39_compose/`; bars as pre-registered)

| arm | n | vs OBSERVING (primary) | vs omniscient | coverage |
|---|---|---|---|---|
| oracle-searched ceiling | 3 | **0.0964** | 0.0217 | 1.00 |
| **llm: llama-3.3-70b** | 8 | **0.0747** | 0.0018 | 0.96 |
| **llm: qwen3-27b** | 8 | **0.0613** | 0.0006 | 0.96 |
| heuristic (gen32 doctrine) | 1 | 0.0603 | 0.0191 | 1.00 |
| random | 20 | 0.0123 | 0.0001 | 0.77 |
| relabel control: llama | 8 | **0.0059** | 0.0005 | 0.85 |
| relabel control: qwen | 8 | **0.0057** | 0.0004 | 0.85 |

**BAR, per model:** llama-3.3-70b beats the heuristic on 2/3 fields AND pooled AND sits above
the random mean: **PASS on every clause.** qwen3-27b: pooled PASS (0.0613 vs 0.0603, thin) and
above-random PASS, but 1/3 fields: **the per-field clause FAILS; qwen = partial**, reported
per-model, never pooled.

**The binding relabel control PASSES for both models, decisively.** Under the swapped brief
(forest described as long-reach/revealing, open as short/hidden) both models' compositions
change materially (llama forest share 54% -> 33%, open 13% -> 33%; qwen forest 71% -> 42%) and
their forces, resolved on the TRUE table, collapse to 0.0059/0.0057 (a 10-13x drop): the
composition follows the briefed physics, not the labels. **The terrain-reasoning claim is
LICENSED - the first time in the LLM arc** (gen33's control failed; the repair that made the
brief match the implemented mechanics is the plausible cause, as its ledger predicted).

**Reading.** The models play the information channel deliberately: their forces are
concealment-heavy (54-71% forest posture), nearly worthless against an omniscient defender
(0.0006-0.0018) yet the strongest non-oracle arms against one that must observe - the
bait/block/anchor structure the act's design predicted. Random composition captures none of it
(0.0123): structure needs the model here, unlike gen37's curation register.

### STEP-3 LAUNCH RECORD (2026-07-26; bars PINNED here BEFORE launch; Kilian's launch authority
### for steps 1-3 granted in-conversation, with his monitoring instruction: check at +30 min
### then hourly, REPORT ONLY, the stop decision is his)

**Trainer** `scripts/train_gen39_conceal.py` (gen32 machinery; state = (w=2 track window,
whole-mission seen-mask); head columns [public terrain exposure, recency, spotted-team threat];
`--blind` zeroes the spotted-team column). **12 runs, LOCAL M4, detached, threads=1 each, pools
capped:** {llm, random, heuristic} x seeds {0,1,2} + llm-BLIND x seeds {0,1,2}; 8000 sorties,
eval every 1000, per-eval checkpoints. Shared oracle cache (`--prep`) so every run scores
against byte-identical yardsticks. **llm training population = all 16 valid step-2 forces (both
models pooled for the curriculum, disclosed; per-model claims stay in step 2), 12 train / 4
held out.** Held-out test = 6 pristine fields (6100-6105) x 4 enemies each (held-out llm force,
fresh random force, heuristic-family force, oracle-searched best force); cell value = mean over
the cell's 4 enemies at the validation-selected checkpoint (common validation = heuristic-family
enemies on fields 3000-3003, all arms alike).

> **PRIMARY (as pre-registered above, instantiated):** the llm-trained defender is BELOW both
> control arms (random-trained, heuristic-trained) on >= 4/6 held-out cells AND pooled, on
> >= 2/3 seeds, at the validation-selected checkpoint. **REPORTED (ungated):** untrained
> context row; the blinded llm arm (sighted-vs-blind IS the concealment channel's worth to a
> trained policy, the arm's second duty); per-cell values beside cap/observing-rule/optimum
> refs; final-iterate drift. **Fail branches, all writable:** arms indistinguishable =
> "curriculum composition does not transfer at this scale" (answers gen33's open metric 2);
> llm worse = reported plainly.

### STEP 3 RESULTS (2026-07-27; 12 runs complete, 5000 sorties each, validation-selected
### checkpoints; `models/runs/gen39_step3/*.json`, scorer `scratch/gen39_step3_score.py`)

**PRIMARY FAILS, 0/3 seeds, per the pre-written branch. Reported plainly.**

| arm (curriculum) | seed 0 | seed 1 | seed 2 | pooled |
|---|---|---|---|---|
| heuristic (single gen32 doctrine) | 0.0872 | 0.0985 | 0.0888 | **0.0915** |
| **llm-composed population** | 0.0878 | 0.1166 | 0.1194 | **0.1079** |
| llm-composed, BLINDED | 0.1232 | 0.1135 | 0.1034 | 0.1134 |
| random compositions | 0.1619 | 0.1271 | 0.1672 | **0.1520** |

**What the bar asked and what happened:** the llm-trained defender had to be below BOTH controls
on >=4/6 held-out cells and pooled, on >=2/3 seeds. It beats the RANDOM-composition control
decisively (6/6, 4/6, 5/6 cells; pooled 0.108 vs 0.152, ~29% better, all three seeds) but is
beaten by the HEURISTIC control on 5/6, 6/6, 6/6 cells. **Verdict: 0/3.**

**The measured finding (the fail branch, written before the run):** *what a defender practises
against matters, and structure beats noise: an LLM-composed enemy population produces a ~29%
better defender than random compositions on unseen enemies, on every seed. But a single
well-tuned coherent doctrine, sited by an exact optimiser, remains the better curriculum. The
LLM's value in this act is at COMPOSITION (step 2, where its forces beat the same doctrine),
not as a training curriculum.*

**DISCLOSED CONFOUND (found during analysis, before the numbers were read into any claim): the
arms did not face equally SITED enemies.** The heuristic arm's laydowns come from
`choose_force` (the exact combination optimiser); the llm and random arms' come from the
step-2 rule placer. So part of the heuristic arm's margin is better placement, not better
composition. **The llm-vs-random comparison is CLEAN (identical placer, composition is the only
difference) and carries the finding above; the llm-vs-heuristic comparison is CONFOUNDED and no
sentence may attribute its margin to composition alone.** The matched-placement re-run is
recorded as the follow-up (Phase 2 below), not as a rescue attempt: the primary stands failed.

**REPORTED ROWS.**
- **Blinded arm (the concealment channel to a trained policy):** pooled 0.1079 sighted vs
  0.1134 blind (1.05x). **Broken down BY ENEMY TYPE (the decisive cut, 2026-07-27):**

  | held-out enemy | sighted | blinded | channel |
  |---|---|---|---|
  | oracle(open) = a REVEALING force (the channel can fire) | 0.2412 | 0.2267 | **0.94x** |
  | oracle(hidden) = a CONCEALED force (nothing is ever revealed) | 0.1370 | 0.1779 | 1.30x |
  | llm / random / heuristic forces (mixed ground) | 0.0751 | 0.0810 | 1.08x |

  **Binding reading, and it CORRECTS the first draft of this row: the trained policy converts
  the exposure channel into NOTHING where the channel actually fires (0.94x against revealing
  forces, marginally negative). The 1.30x on concealed forces cannot be the channel (it never
  fires there) and is a difference between two differently-trained policies, not information
  use.** This is the third sighting of the project's channel-does-not-carry pattern (gen29
  coordination, gen34 type, now gen39 exposure) and the sharpest, because here the same channel
  is worth 1.26-1.37x to the SIMPLE observing rules on the same games (step-1 rows). The
  earlier draft sentence ("the channel helps where the enemy is on revealing ground") is
  measured FALSE and is retracted here rather than amended.
  Blind validation ratios are nonetheless much worse (1.48-2.69 vs 0.98-1.17): the blind arm is
  a worse policy overall, it simply is not worse for the reason the channel would predict.
- **Against the oracle rows:** no arm beats the best OBSERVING rule on any cell (0/6 everywhere;
  best arm 1.64x it), and only the heuristic arm dips below the static cap on more than one cell
  (2/6). **Binding: gen39 licenses NO "trained policy beats the simple rules" sentence in this
  register.** The act's positive is step 2 (composition) plus the step-1 mechanism rows.
- Per-cell values, per-seed selection points and the reference rows (cap / observing rule /
  exact optimum) are in the scorer output; nothing is averaged away.

**Pre-registered extension rule NOT triggered:** no arm's validation curve was still improving at
5000 (best checkpoints at 1000-5000 scattered, VAL flat within noise), so the 8000-sortie
extension is not warranted.

### PHASE 1 (2026-07-27, oracle-only, free): WHY the llm curriculum lost, measured

Two probes, run after step 3 closed, to explain the result rather than rescue it.
`scratch/gen39_phase1_confound.py`, `scratch/gen39_phase1b_score.py`;
artefacts `models/runs/gen39_phase1_confound.json`, `gen39_phase1b_scores.json`.

**1a. The disclosed placement confound is REAL BUT SMALL, and it points the OTHER WAY.** Each
arm's actual training opponents scored on one yardstick (median over 8 training fields):

| opponent family | vs PERFECT play | vs the observing rule |
|---|---|---|
| gen32 doctrine + ORACLE placement (the heuristic arm's) | **0.0215** | 0.0688 |
| gen32 doctrine + rule placement | 0.0105 | 0.0608 |
| **LLM doctrine + rule placement (the llm arm's)** | **0.0007** | **0.0805** |
| LLM doctrine + oracle placement (the proposed matched-placement fix) | 0.0004 | 0.0835 |
| random doctrine + rule placement (the random arm's) | 0.0000 | 0.0068 |

Placement is worth 1.13x with doctrine held fixed. **The llm arm trained against opponents that
were 1.17x STRONGER than the heuristic arm's on the observing-rule yardstick and still produced
a worse defender, so the confound does not explain the step-3 result and the matched-placement
re-run is NOT worth its compute** (llm+oracle stays at 0.0004 vs perfect play). Recorded and
dropped.

**THE MECHANISM (the finding this phase earns).** Curriculum value tracks the opponent's threat
against a defender that ALREADY KNOWS where it is - its irreducible danger - not its threat
against the defender you have:

| curriculum | opponent vs PERFECT play | resulting defender (held-out, pooled) |
|---|---|---|
| heuristic | 0.0215 | **0.0915** |
| llm | 0.0007 | 0.1079 |
| random | 0.0000 | 0.1520 |

Monotone across all three arms. **Mechanism: the LLM composes CONCEALMENT-HEAVY forces (54-71%
cover in step 2; 81-84% in the Phase-1b population), and concealed forces are by construction
strong against a defender that must observe and near-harmless against one that knows (step-1
rows: hidden/open 0.23-0.71 omniscient vs 1.02-1.13 observing). A defender trained against them
learns the narrow skill "locate, then walk around", which does not transfer; a defender trained
against irreducibly dangerous forces must learn real play.** The act's own mechanic therefore
explains its own curriculum negative, which is why the two halves belong in one ledger.

**1b. A richer population does not change the picture** (61 valid forces, 32 requested per
model, 3.7 min of calls; scored on both yardsticks):

| model | n | vs PERFECT play (median) | vs observing rule | cover share |
|---|---|---|---|---|
| llama-3.3-70b | 29 | 0.00194 | 0.0707 | 84% |
| qwen3-27b | 32 | 0.00078 | 0.0643 | 81% |

llama composes forces ~2.5x more irreducibly dangerous than qwen (consistent with step 2's
per-model split, where llama passed every clause and qwen was partial), but **the best force in
61 reaches 0.0091 against perfect play, still 2.4x below the heuristic curriculum's 0.0215, and
only 4/61 clear 0.005.** Cover share correlates POSITIVELY with irreducible threat within the
population (+0.38), i.e. the model is not simply over-hiding; the ceiling is the composition
space it explores. **Consequence for any Phase 2: a llama-only or best-of-N curriculum would
still be a materially weaker curriculum than the tuned doctrine, so it cannot overturn the
step-3 verdict. Pre-registered as NOT worth running on those grounds, before any seed was
spent.**

### PHASE 1C (2026-07-27; Kilian's call: "make the LLM stronger"): THREE INTERVENTIONS, ALL
### MEASURED, ALL SHORT OF THE BAR (`scratch/gen39_phase1c.py`, `models/runs/gen39_phase1c.json`;
### 3.4 min of calls + oracle scoring, no training)

Bar fixed before the calls: an arm justifies a Phase-2 training run only by reaching the
heuristic curriculum's **0.0215 median irreducible threat** (damage against a defender that knows
where every team is).

| arm | n | median irreducible | best | % of bar | cover share |
|---|---|---|---|---|---|
| step-2 / 1b population (baseline) | 61 | 0.00115 | 0.0091 | 5% | 83% |
| **robust brief** (one added constraint: stay dangerous when KNOWN) | 16 | 0.00097 | 0.0075 | **5%** | **54%** |
| **iterative** round 0 / 1 / 2 (compose -> exact score -> revise) | 6 | 0.00147 / 0.00063 / 0.00173 | 0.0062 | 7% / 3% / **8%** | 50-56% |
| **curated** (best-of-N over the 61 by irreducible threat) | 12 | **0.00427** | 0.0091 | **20%** | 94% |

**Reading, and it is a capability boundary rather than a briefing failure.** The robustness
constraint DID change what the model does: cover share halves (83% -> 54%), so it understood the
instruction and complied. Its forces did not become more dangerous to a knowing defender. Three
rounds of feedback carrying the model's own two yardstick numbers and the reference bar produced
no trend (7% -> 3% -> 8%). Only ORACLE CURATION lifts the population materially (5% -> 20%), and
even the single best force in 61+16+18 reaches 0.0091, **2.4x below the bar**.

**Mechanism (consistent with the whole LLM arc).** The model's action space is terrain class +
corridor region + doctrine mix. Irreducible danger on this map is GEOMETRIC: it requires the
particular combination of positions that denies every lane, which the exact combination search
finds and a posture statement cannot express. Phase 1a already showed the same: giving LLM
postures oracle placement leaves them at 0.0004. gen33 measured the other end (an LLM choosing
coordinates directly is indistinguishable from random). **So the LLM adds value where the
judgement is verbal-strategic (composition against a searching defender: step 2, where it beats
the tuned doctrine) and cannot supply value where it is combinatorial-geometric (a force that
survives contact with a knowing defender), no matter how it is briefed or how much feedback it
gets.**

**BINDING CONSEQUENCE: Phase 2 (a llama-only / robust / curated curriculum) is NOT RUN.** No LLM
force family reaches a curriculum strength that could overturn step 3, and that is measured
rather than assumed. The one arm that moves the needle, oracle curation, is by construction the
"LLM proposes, algorithm selects" pattern gen38 already banks.

**What Kilian's challenge earned (recorded, since the challenge was right to make):** the
per-yardstick split is now explicit and both halves are true. Against a defender that must
SEARCH, the LLM's forces beat the tuned doctrine (step 2, 0.0747 vs 0.0603, like-for-like
placement) and inflict more damage than it (Phase 1a, 0.0805 vs 0.0688). Against a defender that
KNOWS, and against every TRAINED defender, they are 2.5-7x weaker (step-3 test rows: heuristic
forces 0.1118-0.2445 damage vs LLM forces 0.0177-0.0537). **The concealment gambit beats a
searcher and loses to a learner - the SACRED-dependent crossover the act was built to find, and
the one quantity the standing wording rule says may never be claimed from oracle rows.**

### PHASE 1D (2026-07-27; Kilian's argument, accepted): THE HIGH-FIDELITY FEEDBACK LOOP
### `scratch/gen39_phase1d.py` + `_b3.py`; `models/runs/gen39_phase1d.json`; 6 rounds x 6
### lineages (3 per model), ~6 min of calls + exact scoring; no training

**Kilian's argument, which corrects Phase 1c's design:** a heuristic cannot act on feedback and
the optimiser behind it only searches blindly; READING A REPORT AND REASONING ABOUT IT is the one
capability a language model has that the alternatives do not, and Phase 1c never tested it - it
handed the model a grade (two scalars), not an account of the battle. Phase 1d gives a real
after-action report each round: the full 26-route table (cost, defender usage, which teams
threaten it), the FREE-LANE list and the cost of the flight's safest option, per-team damage /
exposure / coverage / overlap, the mission decay curve, and its own history. **Diagnosis only,
never prescription** (no "move team 2 to X and damage rises to Y": that would be the optimiser
solving it and the model transcribing). **Grounding check (Kilian's addition):** each force must
declare INTENDED_ROUTES, scored against the truth, to separate misreading from hard reasoning.

| round | irreducible | % of bar | vs searcher | free lanes | grounding |
|---|---|---|---|---|---|
| 0 | 0.00208 | 10% | 0.0361 | 1.8 | 12% |
| 1 | 0.00180 | 8% | 0.0658 | 0.7 | 38% |
| 2 | 0.00249 | 12% | 0.0695 | 1.2 | 23% |
| 3 | 0.00323 | 15% | 0.0573 | 1.2 | 40% |
| 4 | 0.00154 | 7% | 0.0499 | 1.7 | 12% |
| 5 | 0.00298 | 14% | 0.0613 | **0.5** | 34% |

**B1 (free lanes fall): PARTIAL** - 1.8 -> 0.5 over the run, but not monotone (noise band 0.5-1.8).
**B2 (irreducible threat -> 0.0215): FAIL** - 7-15% of bar, no trend; best single force 0.0076 (35%).
**B3 (beat the heuristic force against a TRAINED defender, all 6 held-out fields, 9 checkpoints):
FAIL** - pooled 0.0900 vs 0.1650 (0.55x), better on 2/6 fields. Feedback did NOT change B3:
round-0 and evolved forces are indistinguishable (0.0902 vs 0.0900).

**WHAT THE GROUNDING CHECK BOUGHT (the finding of this phase).** Grounding sits at **12-40%**:
the model's declared INTENDED_ROUTES overlap the routes its force actually threatens by only a
third at best. **It cannot predict the geometric consequences of its own posture choices**, so it
is steering blind: no amount of strategic reasoning downstream can compensate. This REPLACES
Phase 1c's over-reached "capability boundary in combinatorial-geometric reasoning" with a
sharper and better-supported claim: **the failure is GROUNDING - mapping a verbal choice
(terrain class + corridor region) onto the geometry it produces - not strategy.** The model
demonstrably acts on the report where the signal is verbal and direct (free lanes fall 1.8 ->
0.5; damage against a SEARCHING defender nearly doubles, 0.0361 -> 0.061-0.070); it fails where
acting requires predicting geometry.

**DISCLOSED ERROR, corrected in-session (the honest record).** The first B3 run scored ONE field
(6100) and reported PASS at 3.92x. Per-field inspection showed the heuristic force is unusually
weak on exactly that field, so the result was a field-selection artefact. B3 above is the
all-six-field rerun. The same inspection also sharpens the step-3 rows: the heuristic force is
NOT uniformly the more dangerous one - **it loses to the LLM force on 2 of 6 held-out fields
(6100, 6101) and wins on 4** - so every "the heuristic force is 2.5-7x more dangerous" sentence
must carry "on average, with the LLM force ahead on a third of fields".

### PHASE 1E (2026-07-27): A GROUNDED ACTION SPACE + A FAIR CEILING - the grounding problem is
### FIXED, and what remains is measured (`scratch/gen39_phase1e.py`, `models/runs/gen39_phase1e.json`)

Phase 1d diagnosed grounding at 12-40%: the model chose in a vocabulary (class + region) whose
geometric consequences it had never been shown. Phase 1e removes the guess with a SLOT
CATALOGUE - every (class, region) slot that exists on the map, each annotated with the routes a
team there would threaten, its reach, its lethality and whether it reveals itself. Descriptive
only: the catalogue never says which combination to pick. **The fair ceiling is computed
exhaustively over all 165 three-slot combinations of the SAME catalogue**, so the model is no
longer scored against a search with a hundred times more freedom.

| | value |
|---|---|
| restricted ceiling (best of 165 slot combinations) | **0.0278** |
| median slot combination (= choosing at random) | 0.0055 |
| the heuristic curriculum's `choose_force` laydown (context) | 0.0215 |

| arm | n | median | % of ceiling | best | % | grounding | free lanes |
|---|---|---|---|---|---|---|---|
| qwen3-27b | 8 | 0.0071 | **26%** | **0.0113** | **41%** | 92% | **0.5** |
| llama-3.3-70b | 8 | 0.0007 | 3% | 0.0061 | 22% | 90% | 4.5 |
| ALL | 16 | 0.0038 | 14% | 0.0113 | 41% | **91%** | 2.5 |
| round 0 -> round 1 (one feedback round) | | 9% -> **20%** of ceiling | | | | | 3.2 -> 1.8 |

**G (grounding >= 80%): PASS - 12-40% -> 91%.** The interface fix worked exactly as predicted:
given a catalogue it can read, the model knows what its own force covers.
**C2 (beat a random slot choice): PASS** - best 0.0113 vs 0.0055, so it is genuinely choosing.
**C1 (>= 60% of ceiling): FAIL at 14% median / 41% best.**

**What this settles, and it is the sharpest statement of the LLM arc.** The failure was NEVER
briefing (Phase 1c), and it is no longer grounding (fixed here). With a readable action space and
full diagnostic feedback the model still recovers only ~2x random and ~40% of the best available
choice: **what remains is combinatorial SEARCH over its own options.** The gap is now attributable
to one named capability rather than to prompt quality, and the fix is architectural (let the model
propose, let a search select: the gen38 pattern) rather than linguistic.

**Two things worth banking beside it.** (i) **A model reversal:** qwen3-27b is far better here
(26% vs 3% median; 0.5 vs 4.5 free lanes), the opposite of step 2's ordering where llama led -
the per-model rule earns its keep, and no "LLMs" sentence is licensed. (ii) **Feedback works
where the model can act:** one round of the catalogue-grounded report doubled the median
(9% -> 20% of ceiling) and halved the free lanes, replicating Phase 1d's pattern with the
grounding confound removed.

**Methodological catch, disclosed:** the restricted ceiling (0.0278) EXCEEDS the heuristic
curriculum's own laydown (0.0215), i.e. `choose_force`'s surrogate search is itself suboptimal
and a better enemy than the one we trained against exists inside a 165-option menu. The step-3
heuristic control is therefore a strong-but-not-optimal opponent, and its label in every table
should read "the tuned-doctrine curriculum", never "the best possible curriculum".

### PHASE 1F (2026-07-27): SAMPLE EFFICIENCY IN AN UNENUMERABLE SPACE - the honest place for an
### LLM to earn its keep (`scratch/gen39_phase1f.py`, `models/runs/gen39_phase1f.json`)

Phase 1e showed the residual gap on a 165-option menu is combinatorial search, where brute force
is free and no model can earn its place. Phase 1f moves to the space that actually exists:
**200 sites = 1,313,400 three-team forces**, nobody enumerates that. Every arm gets the SAME
budget of 96 EXACT EVALUATIONS (the operational currency: one full mission solve each), the SAME
fields, and the SAME fixed doctrine, so only the SEARCH differs. Model calls are reported but not
charged to the budget: the question is "fewer simulations", not "fewer tokens".

**Best force found vs evaluation budget:**

| arm | @8 | @16 | @32 | @48 | final (96) |
|---|---|---|---|---|---|
| random triples | 0.0110 | 0.0267 | 0.0385 | 0.0385 | 0.0385 |
| greedy top-K by site threat | 0.0374 | 0.0394 | 0.0394 | 0.0394 | 0.0404 |
| local search (seed + steepest-descent swaps) | 0.0345 | 0.0345 | 0.0410 | 0.0410 | **0.0485** |
| **llm: llama-3.3-70b** | **0.0394** | **0.0408** | 0.0408 | 0.0408 | 0.0433 |
| **llm: qwen3-27b** | 0.0309 | 0.0316 | **0.0426** | **0.0426** | 0.0426 |

Context lines: the tuned-doctrine curriculum sits at **0.0215**, the 165-slot restricted ceiling
at **0.0278**. **Every search arm in the full space beats both**, which is the first thing to say
plainly: the step-3 heuristic control was never near the best available enemy.

**S1 (beat random): PASS** (0.0433 vs 0.0385). **S3 (reach the tuned doctrine's 0.0215): PASS,
2.0x it.** **S2 (beat local search): FAIL** - local reaches 0.0485, ~12% above the best LLM arm.

**The honest reading, and it is a genuine LLM positive with a named boundary.** At SMALL budgets
the LLM arms lead everything: at 8 evaluations llama is at 0.0394 where local search is at 0.0345
and random at 0.0110, i.e. **reading the map in words gets a good force in a handful of
simulations**, which is exactly the operational regime (an intelligence cell that can afford a
few full assessments, not a hundred). By 96 evaluations local search overtakes: **given enough
simulations, blind hill-climbing wins.** The crossover sits between ~32 and ~96 evaluations on
this map. So the claim the act licenses is a SAMPLE-EFFICIENCY one, bounded: *language-guided
proposal dominates blind search in the low-budget regime and is overtaken once the budget grows*
- never "the LLM finds better forces".

**Also measured:** both models plateau early (llama flat from evaluation 23 to 61, qwen from 34),
i.e. they propose a strong first neighbourhood and then stop improving on their own leaderboard,
while local search keeps grinding. That is the same shape as Phase 1d/1e (good first move, weak
refinement) and it points at the obvious hybrid, recorded as future work and NOT run here: LLM
proposes the neighbourhood, local search refines inside it.

### STEP 5 PRE-REGISTRATION (2026-07-27, BEFORE any code that touches it; Kilian's go for
### 2 seeds): DOES A STRONG CURRICULUM FIX THE STEP-3 NEGATIVE?

**Why this is a test and not a rescue.** Phase 1 measured a mechanism: curriculum value tracks
the enemy's IRREDUCIBLE THREAT (damage against a defender that knows the laydown), monotone
across the three step-3 arms. Step 3's llm arm failed with 0.0007-threat enemies against the
tuned doctrine's 0.0215. Phase 1f/free-gate then produced LLM-proposed enemies at 0.030-0.049
with only 16 exact evaluations. **The mechanism therefore PREDICTS the llm arm should now win.**
That prediction is falsifiable, is written here before the runs, and the fail branches below are
written with it.

**FROZEN DESIGN (all arms identical except the curriculum's source).**
- Theatre narva, the pinned cell (K=3, cr 0.85, rm 0.7, table lethality); 5000 sorties;
  validation-selected checkpoint; the step-3 trainer unchanged.
- **DOCTRINE FROZEN to gen32 (0.6/0.2/0.3) in EVERY arm.** Free-gate Part A measured LLM-written
  doctrine on the same positions at 0.53-0.75x the tuned recipe on all four maps, so an
  LLM-doctrine arm is excluded on evidence, not preference; the LLM's contribution under test is
  POSITIONS ONLY.
- **Matched budget: every search curriculum gets 16 exact evaluations per field.** Without this,
  "llm wins" is indistinguishable from "any search wins".

**ARMS (4 x 2 seeds = 8 runs, ~6 h local).** `llm16` (llama-proposed, 16 evals), `local16`
(steepest-descent swaps, 16 evals), `random16` (uniform triples, 16 evals), `tuned` (the step-3
control, unchanged: `choose_force` + gen32).

**TEST SET (binding, and different from step 3's).** Six pristine fields x FOUR enemy families,
all built STRONG by the same 16-eval recipe: llm16, local16, random16, tuned, plus the
oracle-searched force as the ceiling row. Step 3's test set was mostly WEAK enemies (0.02-0.05
damage), where a free-lane shortcut suffices and arms are indistinguishable; that is disclosed
there and repaired here.

> **PRIMARY: the llm16-trained defender is below the tuned-trained defender on >= 4/6 held-out
> cells AND pooled, on >= 2/2 seeds.**
> **SECONDARY (the control that decides what the claim may say): llm16 vs local16 and vs
> random16 on the same clauses.** If llm16 beats `tuned` but ties the search controls, the
> licensed sentence is "a STRONG curriculum fixes the step-3 negative; the LLM is one
> sample-efficient way to author one", NOT "the LLM curriculum is best".
> **STAGED SEEDS (pinned now, so it can never be a rescue): n=2 first. A THIRD seed is added if
> and only if the arm ordering is AMBIGUOUS - any pair of arms within 10% pooled, or a 1-1 seed
> split - never because a particular arm lost.**
> **Fail branches, all writable:** (a) all strong arms tie the tuned arm = the curriculum-strength
> curve SATURATES above ~0.02, which locates its knee and is a real finding; (b) llm16 below the
> controls = the step-3 negative was not curriculum strength and the mechanism is wrong, stated
> plainly; (c) llm16 wins only where free-gate said it should (narva/ukraine/fulda, not kgd) =
> the map-dependence becomes the result.
> **DISCLOSED IN ADVANCE:** on kgd_gvardeysk the free gate measured RANDOM search beating the
> LLM (0.0647 vs 0.0494); kgd is therefore a pre-declared negative cell for the LLM edge, and the
> zero-shot map rows must report it.

### STEP 5 RESULTS (2026-07-28; 8 runs, 4 arms x 2 seeds, 5000 sorties, validation-selected;
### `models/runs/gen39_step5/*.json`, scorer `scratch/gen39_step5_score.py`)

**PRIMARY PASSES 2/2 SEEDS. The step-3 negative is fixed, and the mechanism's prediction held.**

| arm (how the enemy's POSITIONS were chosen; doctrine frozen to gen32 everywhere) | seed 0 | seed 1 | pooled |
|---|---|---|---|
| local16 (hill-climbing, 16 evals) | 0.1298 | 0.1198 | **0.1248** |
| **llm16 (llama-proposed, 16 evals)** | 0.1302 | 0.1417 | **0.1359** |
| random16 (uniform triples, 16 evals) | 0.1597 | 0.1844 | 0.1720 |
| tuned (`choose_force` optimiser: the step-3 control) | 0.1717 | 0.1824 | 0.1771 |

> **PRIMARY (llm16 below the tuned control on >=4/6 cells AND pooled, 2/2 seeds): PASS** -
> beats it 5/6 and 5/6, pooled 0.1359 vs 0.1771 (**23% better**), both seeds.
> **SECONDARY vs random16: PASS 2/2** (6/6 cells both seeds). **SECONDARY vs local16: FAIL 0/2**
> (beats it 3/6 and 1/6; local16 pooled 0.1248 is 8% BETTER than llm16).

**THE LICENSED SENTENCE, and it is exactly the one the pre-registration reserved:** *a STRONG
curriculum fixes the step-3 negative - training against enemies with high irreducible threat
produces a defender ~23% better on unseen strong enemies than the tuned-doctrine control, on
every seed - and the LLM is ONE sample-efficient way to author such a curriculum, not the best
one. A 16-evaluation hill-climb authors a slightly better curriculum still.* No sentence claiming
the LLM curriculum is superior is licensed.

**The mechanism is confirmed quantitatively.** Curriculum irreducible threat -> resulting defender:
llm16 0.0393 -> 0.1359, local16 0.0222 -> 0.1248, random16 0.0286 -> 0.1720, tuned 0.0278 ->
0.1771. The two arms whose curricula were built by a DIRECTED search (llm16, local16) produce the
two best defenders; the two built by undirected or surrogate search do not. **Threat alone does
not order the arms** (llm16 has the strongest curriculum but is second): what the strong arms
share is that their enemies were selected by iterated exact evaluation, i.e. the curriculum's
DIVERSITY-plus-strength matters, not strength alone. Recorded as a refinement of the Phase-1
mechanism, not a contradiction of it: at 0.0007 (step 3) the arm failed; at 0.022-0.039 all
directed arms succeed; the curve saturates in between.

### STEP 5 AT n=3 (2026-07-28; the third seed the ambiguity trigger called for; scorer re-run
### over all 12 runs)

| arm | seed 0 | seed 1 | seed 2 | pooled +/- sd |
|---|---|---|---|---|
| **llm16 (llama-proposed, 16 evals)** | 0.1302 | 0.1417 | **0.1145** | **0.1288 +/- 0.0111** |
| local16 (hill-climb, 16 evals) | 0.1298 | **0.1198** | 0.1564 | 0.1353 +/- 0.0155 |
| random16 | 0.1597 | 0.1844 | 0.1421 | 0.1621 +/- 0.0173 |
| tuned (step-3 control) | 0.1717 | 0.1824 | 0.1490 | 0.1677 +/- 0.0139 |

> **PRIMARY (llm16 below the tuned control, >=4/6 cells + pooled): PASS on 3/3 seeds.**
> Paired difference **-0.0389 +/- 0.0031**, i.e. 23% better with a spread an order of magnitude
> smaller than the effect: the fix of the step-3 negative is unambiguous.
> **vs random16: PASS 3/3 seeds** (6/6, 6/6, 4/6 cells). **vs local16: 1/3 seeds**; the third
> seed REVERSED the ordering (llm16 ahead 0.1145 vs 0.1564) and the paired difference is
> **-0.0066 +/- 0.0265** - the spread is 4x the difference. **llm16 and local16 are STATISTICALLY
> INDISTINGUISHABLE at n=3 and the ledger says so; no ordering between them is claimed.**
> The pinned rule permits no fourth seed, and none is run.

**FINAL LICENSED SENTENCE FOR THE ACT:** *training SACRED against enemies authored by a directed
16-evaluation search - whether the proposals come from a language model or from a hill-climb -
produces a defender 23% better on unseen strong enemies than the tuned-doctrine control, on every
seed. The LLM and the hill-climb are indistinguishable as curriculum authors (paired difference
0.0066 +/- 0.0265); what separates both from the controls is that the curriculum was SELECTED by
iterated exact evaluation, and what separates the LLM from the hill-climb is only cost (a handful
of model calls versus an evaluator in the loop throughout, the Phase-1f low-budget result).*

**Oracle rows at n=3 (pooled):** llm16 and local16 beat the static cap on 5/6 cells (0.82x,
0.86x); random16 and tuned on 4/6 (1.04x, 1.09x). **No arm beats the best observing rule on any
cell (0/6): the standing gen39 boundary is unchanged.**

**AMBIGUITY TRIGGER FIRED at n=2 (pinned pre-launch):** llm16~local16 differ by 8.2% pooled (<10%) and
random16~tuned by 2.8%. **A third seed was warranted by the pinned rule** - because the ordering was
ambiguous, NOT because a particular arm lost. RUN on Kilian's go 2026-07-28; results in the n=3
block above (the ordering between llm16 and local16 reversed, confirming it was noise).

**Against the oracle rows (pooled):** llm16 and local16 beat the static cap on 5/6 cells
(mean 0.87x and 0.79x the cap); random16 and tuned on 3/6 (1.10x, 1.16x). **No arm beats the best
OBSERVING RULE on any cell (0/6 everywhere)** - the standing gen39 boundary is unchanged and no
"trained policy beats the simple rules" sentence is licensed by this act.

**Disclosed limitations.** (i) The validation set is inherited from step 3 and is built from
TUNED-family enemies, so the tuned arm's validation numbers (0.78-0.79) are not comparable to the
strong arms' (1.14-1.54); every arm is selected by the same rule, but the strong arms may be
losing their best checkpoints to a mismatched validation family. Fixing this is a rebuild of the
validation cache, recorded as future work. (ii) `tuned` is a poor label - it means "positions from
our own optimiser, tuned doctrine"; all four arms share the gen32 doctrine.

### STEP 5 ZERO-SHOT ROWS (2026-07-28; `scratch/gen39_zeroshot.py`,
### `models/runs/gen39_zeroshot.json`): the Narva-trained defenders on three unseen theatres

All twelve validation-selected checkpoints scored on kgd_gvardeysk, ukraine and fulda. Each map
gets its OWN strong test set, built exactly as Narva's: four enemy families each authored by its
own 16-evaluation search at the matched budget, plus the oracle ceiling; doctrine frozen to gen32.
Nothing retrained. **HARNESS SELF-CHECK: rebuilding Narva's test set through this harness
reproduces the step-5 run log to 0.00000 on all six cells**, so the map rows rest on a verified
pipeline.

| map (unseen) | llm16 | local16 | random16 | tuned | llm16 beats tuned | beats local16 |
|---|---|---|---|---|---|---|
| kgd_gvardeysk | **0.3522** | 0.3688 | 0.3720 | 0.4117 | **3/3 seeds** | **3/3 seeds** |
| ukraine | **0.2192** | 0.2390 | 0.2351 | 0.2654 | **3/3 seeds** | **3/3 seeds** |
| fulda | **0.1042** | 0.1108 | 0.1160 | 0.1099 | **3/3 seeds** | 2/3 seeds |

**The step-5 result TRANSFERS, and on the unseen maps it is cleaner than at home.** llm16 beats
the tuned control on 3/3 seeds on all three theatres (paired -0.0595 +/- 0.0082 on kgd,
-0.0461 +/- 0.0160 on ukraine, -0.0056 +/- 0.0042 on fulda: on kgd and ukraine the effect is
5-7x its own spread). llm16 is also ahead of local16 on every map (3/3, 3/3, 2/3 seeds), which
Narva at n=3 could not separate: **the ordering that was statistically indistinguishable in
distribution becomes consistent out of distribution.** No sentence claims a large llm16-local16
margin - the paired differences (-0.017, -0.020, -0.007) remain within a spread of similar size -
but the SIGN is now the same on nine of nine map-seed pairs, which the Narva rows alone did not
support.

**THE PRE-DECLARED KGD PREDICTION FAILED, and that is disclosed prominently.** The free gate
measured RANDOM search beating the LLM proposer at authoring kgd forces (0.0647 vs 0.0494), and
the step-5 pre-registration therefore pre-declared kgd a negative cell for the llm16 arm. In the
event, kgd is the arm's STRONGEST map (paired -0.0595 +/- 0.0082, the largest margin of the
three). **Reading, stated plainly: curriculum-authoring strength on a map does not predict the
transferred defender's quality on that map - the two quantities came apart, and the free gate's
per-map ordering does not license a per-map prediction about the trained policy.** The prediction
is recorded as made and as wrong rather than quietly dropped.

**Oracle rows (pooled): llm16 beats the static cap on 6/6 cells on ALL THREE unseen maps** (cap
0.441 / 0.429 / 0.155 vs llm16 0.352 / 0.219 / 0.104). **No arm beats the best observing rule on
any cell on any map (0/6 everywhere)**: the standing gen39 boundary survives transfer unchanged,
and no "trained policy beats the simple rules" sentence is licensed anywhere in this act.

### STEP-3 AMENDMENT (2026-07-26 afternoon, BEFORE any result is read; Kilian's instruction:
### "add resume and restart at 5000")

The first three launch attempts never produced a checkpoint (thread-pool thrash at 12-way; the
151k-forwards eval defect, repaired above; then the `nice` band confining runs to efficiency
cores at ~7.6 s/sortie measured vs the historical 2.5). Changes, all pinned before results:

1. **Budget 8000 -> 5000 sorties**, with a PRE-REGISTERED extension rule: any arm whose
   VALIDATION curve is still improving at 5000 is extended to 8000 by LOSSLESS RESUME; the
   extension decision reads the validation curve only, never the test cells.
2. **Resume machinery** (`--resume` / final full-state save: nets, optimisers, alpha,
   replay buffer, all four RNG states; featurization caches stripped, they rebuild
   deterministically). **Equivalence smoke result, disclosed plainly: the split run does NOT
   reproduce the straight run bitwise, because the trainer itself is not run-reproducible -
   two byte-identical invocations (same seed, same PYTHONHASHSEED, same everything) diverge
   (VAL 3.79 vs 4.67 at sortie 160). This is inherited from the gen32-class machinery and was
   never previously tested; gen31/32's claims rest on multi-seed confirmations, not per-seed
   reproduction.** The resume is therefore STATE-COMPLETE (everything defining the process is
   restored; the continuation is a stochastic realisation of the same training process, exactly
   as an unbroken run's tail is). The extension rule stands unchanged: validation-curve-only
   decision, extension = more training of the same restored state.
3. **Launch at default priority** (no nice: it cost ~3x on the P/E-core split), waves of six,
   caps + passive waiting kept, stagger kept. The only learning-relevant signal so far, from the
   crawled first checkpoints at sortie 1000 (llm arm, discarded run): rw[known-threat] trained
   to -6.4/-6.9 and held-out damage fell ~25-35% below untrained: the reveal channel drives
   behaviour, as designed. Those numbers are context only; the restarted runs are the record.

### STEP-3 THROUGHPUT REPAIR, MEASURED (2026-07-26 evening; solo smoke, machine otherwise clear)

The four launch attempts that produced no result were all one defect: **every stored transition
memoised its OWN copy of the field's featurized graph**, so a run's footprint grew ~1 GB as the
buffer filled, several runs pinned the machine at the memory-compression threshold, and the
paging showed up as 40-67% SYSTEM time (misdiagnosed twice as thread-pool spin and as the
scheduler's P/E-core split; both fixes were real but not the cause). Repaired by attaching ONE
shared per-field graph at push time; tensor-exactness pinned by `tests/test_gen39_trainer_eval.py`.

| flight | cumulative | segment rate | RSS |
|---|---|---|---|
| 200 | 368 s | 1.84 s/flight | 3.1 GB |
| 400 | 731 s | 1.82 s/flight | 3.1 GB |
| 600 | 1098 s | 1.84 s/flight | 3.1 GB |

**Flat**, where the pre-fix code went 2.07 s/flight solo -> 7.2 s/flight by flight 1000 at 6-way.
Footprint 3.1 GB/run sets the concurrency: **four runs per wave** (12.4 GB, clear of the
threshold), three waves. Also disclosed: an hour of this session's diagnosis was invalid because
`pkill -f <pattern>` matched the very shell issuing it, so "batch stopped" was reported while
four trainers still ran; kills now exclude self and are verified.

### OPERATING POINT PINNED FOR STEPS 2-4 (2026-07-26, Kilian's decision: option A, all four maps)

**Primary theatre: NARVA. Cell: K=3 teams, concealed reach 0.85, range multiplier 0.7, hidden
lethality 1.0 x the PINNED table (no weapons knob turned).** At this cell: 92% of laydowns a
real game, G1 3.65, G2 4.36, sight worth 2.08x, hidden/open 1.07 vs an omniscient defender
(firepower matched by construction) and 2.66 vs a defender that must observe: everything that
separates the two force designs IS the information channel. Absolute weapon characteristics on
narva (range_scale = 2.27 x 0.7): open 5.6 km / 0.90; field 4.0 km / 0.85; forest 4.7 km /
0.55 (hidden); urban 3.3 km / 0.45 (hidden). **HELD OUT: kgd_gvardeysk, ukraine AND fulda
(Kilian: all four maps in the act).** The lethality-raise question DISSOLVES: the pinned table
stands (0.70/0.90 rows remain measured context in the cost table and the screen grid). The two
earlier voided kgd pins remain visible above with their void reasons.

### 2026-08-06 SESSION: MODEL IDENTITY, THE UNRECORDED THINKING PROBE, THE v1-BRIEF DEFECT
### IN PHASES 1C/1D/1E, AND THE PRE-REGISTERED REPAIR RE-RUN (disclosures first; bars fixed
### BEFORE the re-run fires; nothing banked outside 1c/1d/1e changes regardless of outcome)

**1. Model identity (binding for every per-model table and the thesis AI acknowledgement).**
The model served as `qwen3-27b` is **Qwen3.6-27B**, established on four independent lines on
the box (models.json repo field, live process command line, gateway `/v1/models`, quantiser
`base_model` declaration). Caveat: Qwen3.5-27B and Qwen3.6-27B carry byte-identical
architecture configs, so identity rests on repo metadata. All thesis text names Qwen3.6-27B
(served alias `qwen3-27b`). The pair is also vintage-asymmetric (Llama 3.3 late 2024 vs
Qwen3.6 2026), which sharpens the reversal finding (the older, benchmark-weaker model leads
at composition) and is disclosed beside any cross-model sentence.

**2. Thinking mode (binding qualifier).** The box's audit log records the thinking flag per
call: all 213 banked qwen calls of this arc (22-27 July) ran `off(default)`. Mechanism:
`gen39_compose.py` pins the gateway, the shared caller never sets `chat_template_kwargs`, and
the gateway injects `enable_thinking: false` where the model default is off. Every banked
qwen number in this ledger is therefore NON-THINKING mode; llama has no reasoning mode ("on"
in its audit column means "not overridden"). Wording rule: qwen claims carry "non-thinking
mode" unless a corrected-brief thinking row says otherwise.

**3. The dead endpoint and the false zero (process disclosure + harness repair).** The pinned
raw-IP gateway URL (`gen39_compose.py:57`, imported by eight gen39 scripts) went dead for
Python callers (IPv4 works, default resolution times out), and the first thinking-probe
attempt reported a CLEAN n=0 because `one()` in 1d/1e swallows every exception
(`except Exception: continue`), so a dead endpoint was indistinguishable from a model that
produced nothing usable. REPAIRED 2026-08-06: the pin now uses the MagicDNS name
(`http://cv-iits-w05.tail5b8d80.ts.net:8080/v1`) and the 1d/1e handlers print the exception
class and message before retrying. Process rule earned: an n=0 or None-heavy LLM result is a
TRANSPORT question before it is a capability reading.

**4. The thinking-on probe (2026-08-06, pre-registered in-session BEFORE spend; script
`scratch/gen39_phase1e_thinking.py`; artefacts `models/runs/gen39_phase1e_thinking*.json`).**
Instrument identical to Phase 1e (same 11 slots, same catalogue, ceiling 0.0278), qwen only,
n=8, `chat_template_kwargs.enable_thinking=true`, max_tokens 8000 (forced co-change,
documented in the script). Against the pre-registered bars: T1 materiality (median >= 0.0107)
FAIL at 0.0094 (a 1.31x gain over the banked 0.0071); T2 (>= 60% of ceiling) FAIL at 34%;
T3 grounding (>= 80%) PASS at 100% (from 92%); C2 (best beats random) PASS. Cost 9-10x the
generation budget (median 6,827 completion tokens vs 456-788; ~125 s vs ~24 s per call; all
8 calls finish_reason=stop, longest 7,578 of the 8,000 cap). Structural reading: thinking-on
moves 32% -> 34% across the feedback round while thinking-off moves 19% -> 28%, so
deliberation and feedback act as SUBSTITUTES; an order of magnitude more reasoning did not
close the residual gap, supporting 1e's "what remains is combinatorial search". CAVEAT: the
probe ran on the defective brief of item 5; every probe conclusion is PROVISIONAL until the
repair below re-establishes it.

**5. THE v1-BRIEF DEFECT (the session's largest finding).** `serialise_theatre` defaults
`terrain=None` to the v1 table (its documented gen33-compat behaviour); Phases 1c
(`gen39_phase1c.py:62`), 1d (`gen39_phase1d.py:224`) and 1e (`gen39_phase1e.py:198`) all
called it with `terrain=None` while every scorer uses the v2 table
(`terrain_v2(hidden_leth=1.0, conceal_reach=0.85)`). The brief therefore misstated every
terrain class (forest briefed r=1.2/p=0.92 and position-REVEALING vs the true r=1.5/p=0.55
and CONCEALED; urban briefed non-emplaceable vs truly emplaceable), and because v1 rows carry
no `reveal` key the prose declared every class revealing: the concealment mechanic this
generation exists to study was ABSENT from the prompt. In 1e the defective physics table sat
beside the CORRECT slot catalogue in one prompt (an internal contradiction). Blast radius,
measured from the audit log's distinguishable system prompts: 167 calls (85 llama, 82 qwen),
confined to 1c/1d/1e. CLEAN by construction: step 2 (passes `terrain=table`, relabel control
included), 1f, step 5, freegate and the zero-shot rows (catalogue/digest path), and 1c's
curated arm (re-ranks the clean step-2/1b population). Measured consequences on the banked
diagnostic chain: no model in either thinking mode ever chose an urban slot (0/24 forces)
though urban sits in the ceiling-defining optimal combination; the reported 1e ceiling 0.0278
was reachable only at 0.0236 (85%) by a brief-compliant model; rescored against the reachable
ceiling the probe reads 40% median / 73% best (thinking-on), 30% / 48% (off), llama 3% / 26%;
and the banked thinking-off median 0.0071 sits marginally BELOW an urban-free random draw
(0.0074), so 1e's "genuinely choosing" clause survives on best-of-8 only (1.53x) pending the
repair. SUPERSEDED PENDING REPAIR: 1c's "capability boundary rather than briefing failure",
1d's grounding 12-40%, 1e's C1 14% / 41% and the "what remains is combinatorial search"
attribution. UNTOUCHED: every banked step-2, 1f, step-5 and zero-shot claim.

### PRE-REGISTERED REPAIR RE-RUN (2026-08-06, bars fixed HERE before any call fires; runner
### `scratch/gen39_repair_rerun.sh`; defective originals preserved as `*_v1brief.*`)

**The fix (committed this session):** the three call sites pass the live v2 instance (`terr`
/ `base.terrain` from `narva_base()`); endpoint and fail-loud repairs per item 3. Instrument
otherwise UNCHANGED (same models, counts, temperature, catalogue and ceiling machinery):
like-for-like with the banked runs, qwen thinking OFF, so the brief is the only moved
variable. Corrected artefacts land at the standard paths.

**Arms:** 1c `--robust --iter --curated` (curated re-ranks unchanged clean inputs, a
consistency row); 1d `--rounds 6 --n 3`; 1e `--n 4 --rounds 2`; then the 1e thinking-on
rider (qwen, n=8) unmodified on the corrected brief.

**Reading rules, both branches pre-committed:**
- R1 (1c). Bars unchanged (irreducible-threat bar 0.0215). If the corrected-brief robust and
  iterative arms move materially toward the bar, the banked "briefing was never the problem"
  reading is RETRACTED and rewritten; if they stay short, it survives WITH the defect
  disclosed. Phase 2 stays not-run unless an arm reaches the bar (the original binding
  consequence re-applies).
- R2 (1d). Grounding re-measured; the 12-40% figure is superseded by the corrected number in
  either direction; the B1/B2/B3 bars are unchanged.
- R3 (1e). Same 165-combination ceiling machinery; C1 (60%) and C2 bars unchanged; BOTH
  ceiling bases reported (0.0278 is now compliant-reachable); urban-slot uptake reported
  (0/24 was the defect's fingerprint).
- R4 (rider). T1-T3/C2 re-judged at the same bars; the substitutes reading either reproduces
  on the corrected brief or is downgraded to unconfirmed.
- R5. No claim outside 1c/1d/1e changes regardless; the thesis may cite the diagnostic chain
  (including "what remains is combinatorial search") ONLY from the corrected run.

**Cost estimate:** ~86 nominal thinking-off calls + 8 thinking-on; GPU ~40-70 min summed;
Mac-side exact scoring interleaved (runs beside the gen41 Act-2 batch; transient spawn pools,
thread caps exported by the runner).

### REPAIR RE-RUN RESULTS (2026-08-06 21:43-22:13, 30 min end to end; corrected artefacts at
### the standard paths, defective originals at `*_v1brief.*`; two transient JSON-parse
### failures surfaced by the new loud handler and recovered by the standing retry; verdicts
### judged against reading rules R1-R5 exactly as pre-registered)

**Consistency rows, both PASS:** 1c curated reproduces the banked values exactly (median
0.00427, 20% of bar; clean inputs, so agreement certifies the scoring machinery unchanged),
and the 1e exhaustive ceiling reproduces exactly (0.0278, same optimal combination): the
brief is the only thing that moved.

**The defect's fingerprint flips.** On the corrected brief the models choose urban slots in
8/24 forces (7/16 off, 1/8 on) against 0/24 on the defective brief: the misstated table was
demonstrably steering slot choice away from a load-bearing class.

**R1 (1c) verdict: the banked conclusion SURVIVES, with its wording softened as measured.**
Robust arm median 0.00194 (9% of the 0.0215 bar; defective brief 5%), best single force
0.0126 (59% of bar; defective 0.0091, 42%), cover share 0.67. Iterative rounds 9% -> 13% ->
3%: still no trend. A truthful brief roughly DOUBLES the median and lifts the best force
by ~40%, so "briefing is not the constraint" is retired in favour of: briefing quality
measurably moves force quality but cannot close the gap (every arm stays 5-10x short of the
bar). The BINDING CONSEQUENCE stands: no arm reaches the bar, Phase 2 stays not-run.

**R2 (1d) verdict: the grounding conclusion SHARPENS.** Grounding on the corrected brief:
per-round medians 11-18%, overall median 12% (individuals 0-100%, 4 None). The banked
"12-40%" becomes "~12% median, UNCHANGED by correcting the physics prose": the model's
inability to predict the geometric consequences of its own verbal choices is not a
briefing artefact. Irreducible threat 6% -> 11% of bar over six rounds (no approach); free
lanes fall 6.0 -> 0.0 (B1 pattern reproduced). B3 (vs trained defenders, `_b3.py`) queued
until the Mac frees; not part of this verdict.

**R3 (1e) verdict: C1 FAIL stands in both modes; the 1e model-reversal sentence is
REVISED.** Off: llama median 17% of ceiling / best 36%, qwen 19% / 36%; on (rider): 29% /
42%. C1 (>= 60%) fails everywhere; C2 passes (best 0.0099-0.0115 vs random 0.0055,
1.8-2.1x). The defective brief had hurt llama far more than qwen (banked 3% vs 26%
medians; corrected 17% vs 19%, near parity), so the banked "qwen far better here" 1e
reversal claim is RETIRED; what survives of it is the free-lane gap (llama 5.5 vs qwen
0.0) and step 2's clean, opposite ordering. New honest caveat: at the MEDIAN both models
sit at or below the random draw (0.0047/0.0053 vs 0.0055); the choosing signal lives in
best-of-N.

**R4 (rider) verdict: the probe's conclusions REPRODUCE on the corrected brief.** T1 FAIL
(median 0.0082 vs bar 0.0107; a 1.55x gain over like-for-like off), T2 FAIL (29% vs 60%),
T3 PASS (grounding 100%), C2 PASS. The substitutes structure reproduces cleanly:
thinking-on round 0 already sits at 29%, exactly where thinking-off arrives only AFTER its
feedback round (5% -> 29% for qwen), and feedback then adds nothing to the thinking arm
(29% -> 23%). Deliberation and feedback are substitutes; an order of magnitude more
reasoning still does not close the search gap.

**R5 and the arc sentence.** Nothing outside 1c/1d/1e moves. The diagnostic chain is now
citable FROM THE CORRECTED ARTEFACTS ONLY, and its final form is: briefing moves the
number but cannot close the gap (1c), grounding of verbal choices onto geometry is ~12%
and unimproved by truthful physics (1d), a readable catalogue fixes grounding to 91-100%
yet both models stop at 17-29% of the ceiling with feedback and thinking as substitutes
(1e + rider): **what remains is combinatorial search, and the fix is architectural**,
re-established on honest ground and now carrying its first real stress test (the thinking
probe) with it.
