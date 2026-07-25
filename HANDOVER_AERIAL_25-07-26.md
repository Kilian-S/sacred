# HANDOVER AERIAL, session of 2026-07-25 (gen39 concealment)

> **ADDENDUM 2026-07-25 late (critic session): read the ledger's "CODE REPAIRS BEFORE THE STEP-1
> RE-RUN" block before acting on this file.** The re-run command below would have crashed
> (`gen39_screen2.py` recorded an undefined `picker`, and the finding-6 force wiring was absent:
> the "wired to choose_force" sentence was aspirational, true only after the repair). Also
> repaired: the stale symmetric-forest cross-check screen; three distortions in
> `gen39_conceal_cost.py` (unscaled ukraine standoff, wrong-matchup force selection, per-cell
> menu rebuild), so THE DECISION TABLE BELOW IS RE-MEASURED before the lethality decision; the
> "gen33 briefs reproduce verbatim" claim (false, removed); urban missing from the step-2 schema.
> One design question now blocks the cost re-measure alongside the lethality call: whether a team
> is spotted where it SHOOTS FROM (its manoeuvre cloud) or only at its nominal point (current).

Assumes you have read `HANDOVER.md` and have your context built. This file covers ONE session and
one act: **gen39, the terrain redesign** (`experiments/gen39_concealment.md` is the authoritative
ledger; this is the map to it). Read the ledger top-to-bottom afterwards: it is written so that
superseded blocks stay visible with the reason they were superseded, and there are four of them.

---

## THE ONE-PARAGRAPH STATE

**Step 1 (the oracle screen) is NOT complete and every screen run so far is void.** Four separate
faults were found and fixed during the session, each of which invalidated the run before it: a
symmetric forest sight-block that killed two theatres, a terrain leak in the engagement
concentration, a candidate raster that systematically missed cover, and a scan-order tie-break that
confined every candidate to half the map. The code is now correct as far as we can tell (suite 240
green) and the screen has NOT been re-run on it. **Step 2 (the three composers) has not started;
it is authorised and unblocked.** **Step 3 (the training batch) has not started and needs Kilian's
explicit go.** One design decision is open and blocks the re-run: whether to raise concealed
lethality (recommendation and numbers below).

---

## WHERE THE THREE STEPS STAND

| step | what it is | state |
|---|---|---|
| **Step 1** | oracle screen: find the operating point where simple rules leave value and the game is not saturated | **code correct, ALL RUNS VOID, needs one re-run after the lethality decision** |
| **Step 2** | three composers (llm / random / heuristic) write DOCTRINE + ROLES only, an algorithm places | **not started; authorised.** w05 gateway reachable DIRECTLY from the Mac on :8080 (no tunnel, no SSH; contrary to `HANDOVER.md`), both `llama-3.3-70b` and `qwen3-27b` live |
| **Step 3** | 12 training runs: 3 curricula x 3 seeds + a blinded arm x 3 seeds | **not started; needs explicit go.** PINNED: LOCAL on the M4, 8000 sorties, detached, ~7.5 h |
| Step 4 | optional LLM enemy commander every 10 serials | untouched |

**Step 3 pinning is deliberate and disclosed** (`ac454d0`, `13ab169`): w05 is not materially faster
because wall-clock is set by single-run speed, not core count, and the GPUs are irrelevant for
these small graphs. 3 seeds is thin and that thinness is on record; the arms share seeds so the
comparison is paired. The blinded arm carries TWO results now: the standard "is the gain the
conditioning" control AND the concealment claim, because sighted-vs-blind IS the measurement of
what the information channel is worth to a trained policy.

---

## THE DECISION THAT BLOCKS THE RE-RUN

Concealment currently costs the enemy about 40% on Kaliningrad. The cost decomposes cleanly
(`results/gen39_conceal_cost.txt`):

| concealed force given back | KGD K=3 | KGD K=6 | UKR K=3 | UKR K=6 |
|---|---|---|---|---|
| nothing (table as pinned) | **58%** | **65%** | 28% | 37% |
| open REACH | 68% | 75% | 29% | 59% |
| open LETHALITY | **87%** | **95%** | 45% | 68% |
| both | 98% | 112% | 55% | 89% |
| *control: OPEN force cut to the same few positions* | *80%* | *84%* | *87%* | *84%* |

Two separable halves on kgd: **weapons ~22 points** (lethality is the expensive charge, reach the
cheap one: reach alone buys 10 points back, lethality alone buys 29) and **choice ~20 points**
(the control row: an OPEN force cut to 53 positions loses 20 points by itself, so half the penalty
is simply that there is less cover to choose from).

**Recommendation on the table, not applied:** raise concealed lethality to ~0.70 and keep the reach
penalty. That gives 84% (K=3) / 94% (K=6): near parity, so the enemy's choice stays open. Giving
both back (98-112%) makes hiding weakly dominant and deletes the decision the act exists to study.
Physical story matches: an unseen team shoots first with an unspoiled firing solution, so surprise
buys KILL PROBABILITY while clutter costs ENGAGEMENT ENVELOPE. **Ukraine breaks even at no swept
setting** (best 80%): map-dependent, not a knob.

---

## WHAT WAS FOUND, IN ORDER (each entry killed the run before it)

1. **The defender forgot.** The reveal set came from the 2-serial track window, so a located team
   was unknown again two serials later and concealment was under-rewarded. Replaced with
   whole-mission memory: state = window x set-of-teams-seen, measure becomes EPISODIC (expected
   damage over a T-serial mission from complete ignorance, exact backward induction). **Verdict:
   faithful but NOT where the value is** (G2 moved 1.03-1.09x). A single-cell reading suggested
   2.67 -> 3.85; it did not survive the grid. Kept because it is the correct model.

2. **The yardstick could not see concealment.** Everything was divided by the omniscient optimum,
   which is handed every emplacement, so the mechanic was unmeasurable by construction and the
   hidden-lethality axis was inert. Split into two comparisons: "room for a learned policy"
   (denominator = omniscient optimum, unchanged, 3-4x) and "does concealment do anything"
   (same defender with vs without the exposure channel). **A belief-MDP yardstick was proposed and
   rejected by Kilian: it is an oracle construct that would change the scorekeeper without touching
   the agent.** He was right.

3. **Forest blinded the team that sat in it.** v2 blocked sight symmetrically. Measured cause of
   two dead theatres: fulda 66% wooded -> 79% of sight lines masked -> 13.3x less at stake;
   ukraine (12% wooded) unaffected at 1.0x. Kilian's call: **canopy hides a team, modern radar-cued
   sights still engage above the treeline.** `TERRAIN_V2["forest"]["los"]` True -> False, `reveal`
   stays False; urban keeps both (buildings are true vertical obstacles). Fulda went 0% -> 81%
   playable, narva 32% -> 66%. **The same commit repaired the LLM brief**, which derived "it
   conceals you (blocks line of sight)" from the single `los` flag: that is a mechanic the
   simulator never implemented and is the likelier of the two candidate causes of gen33's failed
   terrain control.

4. **The engagement concentration leaked across terrain (Kilian's catch).** The RBF bump was ~4.5 km
   wide on a 2 km grid and spread over every nearby site regardless of ground, while `reveal` reads
   only the team's own site. A "forest" team delivered **20%** of its effect from forest and
   **60% from OPEN ground** while keeping woodland's invisibility, diluting the price of cover
   about fivefold. `same_class=True` (now default) masks the concentration to the team's own
   ground. **This retracted the hide-vs-open crossover.**

5. **The candidate raster systematically missed COVER (Kilian's catch).** Cover comes in patches
   smaller than the grid, open ground in blocks. On kgd only 17% of forest patches and 5% of urban
   patches held any candidate; on ukraine the sampled forest patches held **11%** of forest area,
   fulda 14%. Replaced by **Kilian's quota scheme**: fixed 200-point budget, class shares taken
   from the WHOLE-MAP terrain composition, even grid as the spatial skeleton, each point snapped
   INSIDE the polygon whose characteristics it carries. Verified: kgd forest true 17.1% -> sampled
   17.4%, urban 9.4% -> 9.7%. **Both halves are needed together**: snapping every node to nearby
   cover over-represents cover, sampling purely by area clumps lengthwise.
   **RETRACTED here: my "cover does not span the corridor / ukraine's cover is dust" mechanism.**
   It compared patch AREA against weapon FOOTPRINT, which are unrelated: a MANPADS needs somewhere
   to stand, not somewhere to fit its engagement circle. Patch area is used nowhere in the sampler.

6. **The force picker was picking positions, not forces.** Greedy top-K by individual threat selects
   K adjacent points and leaves the corridor open, and it DEGRADES as candidates densify (kgd ratio
   0.28 -> 0.05 as H went 563 -> 2360), i.e. the numbers were reporting the picker. Added
   `ConcealBase.best_laydown`: greedy max-min seed plus steepest-descent swaps on a close-every-lane
   surrogate. The surrogate is not the true objective and can lose to the old picker (+53% ukraine,
   -9% kgd), so the screen scores BOTH exactly and keeps the winner, recording which won.

7. **Scan-order tie-break confined every candidate to half the map.** Found by inspecting the site
   figures Kilian asked for. Anchors standing inside a patch all tie at distance zero and the tie
   broke on index: **every kgd candidate sat in the left 27 km of a 45 km theatre.** Fixed with
   farthest-point selection per class. The anchor grid now also sizes itself from the budget
   (fulda's 11.6 km grid gave 87 anchors against a 200-point quota, so OPEN got ZERO points despite
   23% of the theatre). **This voided the verification that had declared concealment dead.**

---

## WHAT SURVIVES, AND WHAT IS RETRACTED

**Survives (mechanism, exact, with an internal control):** sight is worth **~1.1-1.6x** to the
defender against a force on revealing ground and **exactly 1.00x** against a concealed one, on
every map and team count measured. Same map, same rules, only the ground changes. Concealment does
exactly what it was designed to do: it shuts the information channel completely.

**Survives, pending re-measurement:** the room for a learned policy (simple rules leaving ~3-4x
against the omniscient optimum). It never depended on concealment and held across every screen
variant, but every measurement of it is on void code and must be re-run.

**Retracted this session, all four disclosed in the ledger with their mechanism:**
- the hide-vs-open crossover (terrain leak);
- "narva/fulda die because the route menu is wider than the force can cover" (it was forest LOS);
- "cover does not span the corridor / ukraine's cover is dust" (patch area vs weapon footprint);
- "the concealment headline is dead" (measured on a half-map candidate set).

---

## CODE WRITTEN OR CHANGED

**Environment (`src/`)**
- `src/envs/aerial_theatre_vec.py`: `TERRAIN_V2` forest `los` True->False; `terrain_v2(forest_los=)`
  default False; `quota_sites()` (Kilian's quota sampler, farthest-point selection, self-sizing
  anchor grid, snap-into-polygon); `_snap_into`, `_class_parts`; `hazard_sites(stratified=, seed=)`
  (the earlier area-stratified top-up, superseded by quota but kept); `build_theatre_game(n_sites=,
  stratified=, site_seed=)`.
- `src/envs/aerial_conceal.py`: `episodic()` rule-matrix hoist (byte-identical, 1.6-1.8x) and
  `horizons=` (whole mission-length curve from one sweep); `episodic_rule(horizons=)`;
  `_topm_row`; `ConcealDyn(same_class=True)` (default, no terrain leak);
  `ConcealBase.best_laydown()` (the force picker); `ConcealBase(n_sites=, stratified=, site_seed=)`.
- `src/redforce.py`: `_physics_table_text(terrain=)` states HIDING and SIGHT-BLOCKING separately
  (was conflated off `los`); `serialise_theatre(terrain=)` takes the table in force, defaulting to
  v1. (The v1-default text is NOT byte-identical to gen33's original brief; gen33's record is its
  stored transcripts. Claim corrected 2026-07-25, Kilian's call.)

**Screens and probes (`scratch/`)**
- `gen39_screen2.py` — the paired-memory screen, block-parallel (36 blocks, `--launch --workers N`,
  resumable, `--merge`, `--check` asserts the forgetful arm reproduces step 1a). Wired to
  `n_sites=200` and `choose_force`. **Not run on the current code.**
- `gen39_screen.py` — the original serial screen (step 1a), kept as the independent cross-check.
- `gen39_site_map.py` — the per-map candidate figures.
- `gen39_conceal_cost.py` — the cost decomposition and break-even sweep.
- `gen39_verify_fixes.py` — hide-vs-open with all repairs (its printed verdict is now void; the
  script is correct).
- `gen39_leak_probe.py`, `gen39_conceal_rescue_probe.py`, `gen39_patch_coverage_probe.py`,
  `gen39_sampling_saturation_probe.py`, `gen39_stratified_probe.py`, `gen39_why_dead.py`,
  `gen39_geometry_probe.py` — the diagnostic probes behind findings 3-6.
- `gen39_read_screen.py`, `gen39_read_scoping.py`, `gen39_read_operating_point.py`,
  `gen39_pick_operating_point.py`, `gen39_pinned_cell.py`, `gen39_export_summary.py` — read-outs.
- `gen39_maps.py`, `gen39_retarget_fulda.py`, `theatre_atlas.py` — theatre figures and stats.

**Tests** `tests/test_gen39_terrain.py`, 16 cases, suite **235 -> 240 green**. Pins: v1 untouched
and byte-identical; forest hides without blinding; the default blocker set excludes forest; the
brief states hiding and sight-blocking separately; the concentration stays on the team's own ground
and `same_class` defaults to the non-leaking value.

---

## ASSETS AND ARTEFACTS

- `assets/gen39_sites_{kgd_gvardeysk,ukraine,narva,fulda}.png` — **every candidate emplacement
  coloured by the ground it stands on**, with counts, true ground share and reach rings. These are
  the audit picture for the sampler; finding 7 was caught by looking at them.
- `assets/gen39_theatre_{...}.png` — flight-path menu, emplaceable ground, reach rings per theatre.
- `experiments/theatre_atlas.md` — the four-map stats (size, bounds, class shares).
- `results/gen39_conceal_cost.txt` — the full cost decomposition and break-even sweep.
- `results/gen39_screen2_leaked_summary.csv`, `results/gen39_screen2_symforest_summary.csv` —
  per-cell summaries of the two VOID screen runs, named apart so they can never be mixed
  (`models/` is gitignored, hence the CSV export).
- `models/runs/gen39_screen2_leaked*`, `models/runs/gen39_screen{,2}_symforest*` — the void raw runs.

---

## RULES THAT NOW BIND ANY LATER WORK

1. **Symmetric-forest and asymmetric-forest are DIFFERENT GAMES**; leaked-concentration and
   masked-concentration are DIFFERENT GAMES; grid-sampled and quota-sampled are DIFFERENT GAMES.
   Numbers may never be mixed across them in one figure or one ladder (standing rule 8).
2. **Two questions, two denominators.** "Room for a learned policy" divides by the omniscient
   optimum. "Does concealment do anything" compares the SAME defender with and without the
   exposure channel. Never one denominator for both.
3. **Concealment claims are a FLOOR.** The observing defender is the best SIMPLE rule, not a
   perfect player under uncertainty, so anything it measures understates what concealment is worth
   to a good defender. The crossover point is a SACRED-dependent quantity and is not claimable from
   oracle rows.
4. **Disclose the sampling limitation.** Proportional sampling gives both classes the same points
   per km2 (fair), but cover is a small share of ground, so a concealed force chooses from ~53
   positions on kgd where an open force chooses from ~147. The control row that isolates this
   (an open force cut to the same count) must accompany any hide-vs-open claim.
5. **Look at the figures before trusting the numbers.** Two of the seven faults were found by
   plotting the candidate set, not by reading tables.

---

## IMMEDIATE NEXT ACTIONS

1. **Kilian decides** whether to raise concealed lethality (recommendation: ~0.70 at the current
   reach). This is the only thing blocking the re-run.
2. **Re-run Step 1** once, on the corrected code: `PYTHONPATH=. python scratch/gen39_screen2.py
   --launch --workers 8` (~2.5 h on the M4, 36 blocks, resumable), then `--merge`, then
   `scratch/gen39_read_screen.py` and `scratch/gen39_pick_operating_point.py`. Re-pin the operating
   point in the ledger; the previous two pins are void.
3. **Step 2** at the new operating point: three composers, ~50-100 model calls over HTTP to
   `http://100.88.32.88:8080/v1` (key `iits-local-key`), tens of minutes. Watch the BINDING
   CONTROL: relabel the terrain in the brief and the model's doctrine must materially change, or
   the claim narrows to composition without terrain content.
4. **Step 3** only on Kilian's explicit go. 12 runs, local, detached.

**Do not launch anything without telling him first.** He stopped a run mid-session for exactly this
reason, and he was right to: it was measuring a known-biased candidate set.

---

## SESSION CONVENTIONS WORTH INHERITING

- British English, no em-dashes, plain language in all replies to Kilian.
- Give shell commands as a single `&&`-chained line.
- Firm research-direction recommendations; builds and launches stay consultative.
- Pin decisions in the ledger BEFORE the run, including the launch location, so deviations are
  disclosed rather than discovered.
- When a result is retracted, leave the old block visible with the reason. Four blocks in the gen39
  ledger are superseded this way and that is the intended shape.
