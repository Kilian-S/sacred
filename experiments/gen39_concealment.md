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
