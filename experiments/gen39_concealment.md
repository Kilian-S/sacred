# Generation: gen39_concealment (the terrain redesign: concealment buys persistence, and the LLM composes doctrine)

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
