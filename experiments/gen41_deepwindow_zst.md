# Generation: gen41_deepwindow_zst (the deep-window zero-shot act; SCREEN pre-registered, act bars DRAFT)

- **status: SCREEN PRE-REGISTERED 2026-08-05 (Kilian's in-conversation go to select and show
  the OD pools; oracle/eval-only; NOTHING TRAINS before his explicit go on the reviewed
  pools and the finalised bars). Act bars below are DRAFT until that go and are then binding
  verbatim or amended by him.**
- **git SHA at registration: `9bf1eb1`.**

## The act (context)

The gen40 landscape (its ledger, laws 1-3) locates the strongest available headline cell for
a zero-shot dynamic act at m = 3 corridors, w = 6 (window a multiple of m, the deepest
rule-failure band), K = 2, padded menus R ~ 15 (kx = 12). There the two-line rule family is
structurally near-static (anti-repeat has nothing left to avoid; rotation's window signature
is balanced and uninformative), while the corridor-restricted optimum sits far below the
statics. Operating point pinned from gen40: tau = 0.15, N = 3, band (0.15, 0.95), mission
objective. Kilian aligned on (m=3, w=6, K=2, R=15) on 2026-08-05; K = 8 was rejected because
the set-softmax enemy is uncomputable past K ~ 5 (gen40 wall law) and m = 3 saturates.

## DRAFT decision metric for the trained act (binding only at launch, after pool review)

Recipe: gen27 verbatim (three training cities, Gdansk held out entirely, per-instance smooth
FP, select-on-train, per-eval checkpoints, no-window causal control) at the new operating
point; policy evaluation by long seeded rollouts (the exact window chain at R^6 is
infeasible; estimator disclosed), oracle references exact.

> **DRAFT PRIMARY: zero-shot on the held-out city, SACRED's mean ratio-to-static-cap beats
> the BEST two-line rule on the same instances (the rule family now includes the extended
> rotation defined below), on >= 2/3 seeds.** DRAFT STRONG: at or below the
> corridor-restricted exact optimum on >= 3/6 held-out ODs (beating the best possible player
> of the entire corridor-locked class). Causal control: no-window arm lands ~ the cap.
> Reported rows: worst-case committing premium; final-iterate drift; per-OD values.

## SCREEN (binding NOW, before any screening CPU)

- **Cities and pools:** kaliningrad, east_london, istanbul (train), gdansk (hold-out); up to
  ~250 sampled deg>=3 OD pairs per city (rng(0), largest component); SELECT 6 per city.
- **Menu requirements:** base disjoint count = 3 AND built-menu core = 3 at kx = 12;
  R in [13, 15]; one-shot equilibrium value >= 0.05.
- **Operating-point requirements at (w=6, K=2), all exact:** with opt_core = Karp on the
  3^6 = 729 corridor window graph,
  1. best_rule / opt_core >= 1.35, where best_rule = min(best rotation over the corridors
     across <= 20 seeded orders, composed anti-repeat over the corridors, and the EXTENDED
     ROTATION family), and
  2. min_static / opt_core >= 1.5, where min_static = min(uniform-core, inverse-vulnerability
     core, exact equilibrium-mixture stationary value; the last computed exactly by
     count-class enumeration with multinomial weights).
     *(AMENDED 2026-08-05 from 2.0 BEFORE the screen ran: the 2.0 figure was calibrated on
     the w=3 landscape; the machinery smoke on the known m=3 OD measured cap/opt ~ 1.67 at
     the agreed (w=6, K=2) point, consistent with gen40's K=2 trend, so 2.0 is infeasible
     at this operating point by structure, not by instance. Noted for Kilian: at K=1 the
     same cell offers cap/opt ~ 2.4 and rule/opt ~ 2.1, structurally stronger on both
     axes, if he prefers to revisit the K choice.)*
- **The extended rotation (the new mandatory baseline, defined here once):** subsets of
  L in {7, 8} routes built greedily for edge-diversity (seed with the 3 corridors, then
  repeatedly add the route sharing fewest edges with the chosen union, ties by lower cost),
  cycled in the natural order and in 10 seeded shuffles (rng(0)); value exact (deterministic
  cycle against the w=6 quantal responder). Its per-instance value is recorded on EVERY
  screened OD, pass or fail, so the strongest naive rule is in the family before any
  training bar is set.
- **Selection and disclosure:** among passers, top 6 per city by best_rule / opt_core; the
  screen selects favourable instances BY DESIGN and the thesis discloses it (the A8
  pattern); full candidate table kept in the artefact.
- **Deliverable for Kilian's review:** one PNG per city (`assets/gen41_pool/<city>.png`),
  six panels each: full street graph, the three corridors bold in colour, padded routes
  light, origin and destination marked, per-panel R and headroom annotations. Script
  `scratch/gen41_pool_screen.py`; artefact `models/runs/gen41_pool_screen.json`.

## RESULTS (appended per step; nothing above changes after each step runs)

### SCREEN RESULT (2026-08-05, 357 s, oracle-only; artefact `models/runs/gen41_pool_screen.json`;
### contact sheets `assets/gen41_pool/{kaliningrad,east_london,istanbul,gdansk}.png`)

301 candidates screened across the four cities (84/76/76/65), 178 passed both bars, 6
selected per city by rule-headroom rank. Selected pools (all m=3, kx=12, exact rows at
w=6, K=2):

| city | ODs (R; rule/opt; stat/opt; ext-rot/opt) |
|---|---|
| kaliningrad | 23-242 (14; 1.64; 1.69; 2.04) · 33-28 (14; 1.74; 1.88; 2.18) · 53-68 (14; 1.63; 1.78; 2.28) · 130-146 (14; 1.64; 1.76; 1.64) · 158-93 (14; 1.64; 1.65; 1.64) · 49-33 (13; 1.73; 1.88; 2.32) |
| east_london | 182-155 (13; 1.56; 1.78; 1.56) · 93-156 (14; 1.61; 1.76; 2.11) · 42-66 (14; 1.59; 1.74; 2.26) · 147-112 (15; 1.59; 1.63; 2.42) · 130-156 (14; 1.60; 1.76; 2.06) · 512-430 (13; 1.57; 1.85; 1.57) |
| istanbul | 596-82 (14; 1.54; 1.67; 2.29) · 1095-824 (13; 1.69; 1.76; 1.83) · 999-45 (14; 1.58; 1.69; 1.90) · 433-1101 (15; 1.56; 1.70; 1.86) · 885-1116 (15; 1.54; 1.68; 2.23) · 1095-115 (14; 1.54; 1.66; 2.00) |
| gdansk (hold-out) | 70-297 (13; 2.36; 2.00; 3.90) · 75-210 (14; 1.63; 1.69; 1.68) · 194-173 (14; 1.64; 1.78; 2.20) · 209-75 (14; 1.62; 1.66; 1.67) · 70-172 (14; 1.64; 1.77; 1.68) · 193-299 (14; 1.64; 1.76; 1.86) |

**The extended-rotation baseline earned its place before any bar was set: on 13 of 301
candidates it BEATS the corridor-locked optimum outright (ext/opt 0.86-0.99, worst gdansk
209-127 at 0.86), by exploiting padded routes no corridor-locked object can reach.** Those
candidates fail the screen by construction; on every SELECTED instance the extended
rotation sits 1.56-2.42x above the optimum. Consequences, binding: (a) the trained act's
rule family includes the extended rotation (already in the DRAFT primary); (b) the screen
deliberately selects instances unfavourable to it, which the thesis discloses exactly as
the A8 favourable-screen sentence; (c) any wording change of the act must keep clause (a).

**Review flag for Kilian (pending his call):** gdansk 70-297 has the best metrics of the
whole screen but its geometry looks operationally degenerate on the contact sheet (origin
and destination nearly adjacent; three near-collinear long-detour corridors). Recommend
replacing it with the next-ranked gdansk passer; one-line swap in the artefact.

**State: pools await Kilian's PNG review; the K=1-vs-K=2 note above awaits his call;
bars finalise at his go; NOTHING TRAINS until then.**

### PRE-REGISTRATION: the two follow-up screens (2026-08-05, Kilian's direction, BEFORE CPU)

**Tier structure adopted for the rule family (Kilian's like-with-like requirement; final
sign-off travels with the bars).** Tier 0 knows the map only (statics, corridor rotation,
FULL-MENU rotation). Tier 1 is told the enemy's mechanism (composed anti-repeat, the
window-tuned extended rotation). Tier 2 earns its knowledge from outcomes at a matched
interaction budget (EXP3, avoid-where-ambushed, self-tuning composed; simulation-evaluated,
reported rows). DRAFT PRIMARY re-scoped: beat every Tier-0 and Tier-2 member zero-shot;
Tier-1 reported beside with information priced, never dropped (baseline completeness).

**Screen 2a, full-menu rotation on the 24 selected instances (Tier 0 entered the family
the moment it was named).** Value = best over the natural order plus 20 seeded shuffles
(rng(0)) of the all-R cycle, exact, at (w=6, K=2). Bar: full-rot/opt_core >= 1.35 per
instance; any failure is swapped for the next-ranked passer that clears BOTH the original
bars and this one, with disclosure and a re-rendered sheet.

**Screen 2b, the menu-diversity probe (Kilian's would-SACRED-suffer question).** Three
instances (klg 23-242, east_london 182-155, gdansk 194-173), each under two menus of
IDENTICAL R: the standing k-shortest menu, and a DIVERSE menu built by penalised shortest
paths (corridor edges and each accepted route's edges reweighted x5; same corridor core;
same padded count; per-edge p_e identical by the absolute vulnerability norm). Per menu:
overlap anatomy; one-shot value; at w=3 the EXACT full-menu and core optima (the padding
value), rules and statics; at w=6 the core optimum, Tier-0/Tier-1 rules and statics
(count-class enumeration). Pre-committed reading: diversity is predicted to RAISE the
naive rules and statics relative to the optimum (headroom compression, the prop-floor
mechanism) while its effect on the w=3 padding value is measured, not assumed; both
directions reported. Scripts `scratch/gen41_fullrot_screen.py`,
`scratch/gen41_menu_diversity_probe.py`.

### SCREEN 2a RESULT (2026-08-05, 24 s; fields folded into the pool artefact)

**All 24 selected instances PASS with zero swaps.** Full-menu rotation (Tier 0, intel-free)
sits at 1.38-3.55x the corridor-restricted optimum across the pools (median ~2.5x); the
one close call is east_london 147-112 at 1.38. Mechanism: cycling through all routes keeps
the flown route out of the enemy's window but not off its punished SEGMENTS, and forces
regular use of high-vulnerability padded variants. The Tier-0 family is comfortably
beatable everywhere on the selected pools.

### SCREEN 2b RESULT (2026-08-05, 50 s; artefact `models/runs/gen41_menu_diversity.json`)

Three instances, standard k-shortest menu vs penalised-diversity menu at identical R and
identical per-edge threat values. Diversity was achieved (median padded own-edge share
27-40% -> 66-80%; candidate edges 68-90 -> 219-317).

| quantity (per instance: klg / e-london / gdansk) | STANDARD | DIVERSE |
|---|---|---|
| one-shot equilibrium value (K=2) | 0.64 / 0.66 / 0.65 | 0.49 / 0.42 / 0.46 |
| w=3 exact full-menu optimum | 0.152 / 0.152 / 0.180 | 0.065 / 0.057 / 0.080 |
| w=3 padding value (core vs full) | 18% / 14% / 12% | 38% / 21% / 26% |
| w=6 ext-rotation / core-optimum | 2.04 / 1.56 / 2.20 | **0.73 / 1.12 / 0.86** |
| w=6 full-rotation / core-optimum | 2.44 / 1.89 / 2.67 | **0.89 / 1.24 / 1.18** |

**Reading (the pre-committed prediction fires, and harder than predicted).** Diverse
padding makes everyone safer in absolute terms (optima improve 2-3x) and DESTROYS the
act's structure: out-of-window cycling over near-independent routes becomes near-optimal
or better than ANY corridor-locked object (ext-rotation at 0.73-1.12x and full-rotation at
0.89-1.24x the corridor-restricted optimum), the corridor-restricted reference stops
bounding the rule family, the true w=6 optimum is incomputable, and the rule-failure
headroom the headline needs disappears. This is the prop-floor mechanism at work: grant
naive spreading enough independence and it is near-optimal. **Binding consequences:**
(i) the act keeps the standing k-shortest menus; near-duplicate padding is not a defect
but the structural condition under which learning has anything to buy; (ii) the 13
screened-out candidates (where ext-rotation beat the core optimum) are now explained:
pockets of incidental menu diversity; (iii) honest thesis caveat wherever menus are
discussed: the mission objective carries no travel cost, so long detours are free in-game;
a latency-priced variant would penalise the diverse menus and is recorded as future work,
not run.

### REVIEW ROUND 1 (2026-08-05, Kilian's first pass on the sheets)

1. **Swap APPLIED:** gdansk 70-297 (degenerate geometry) replaced by the next-ranked passer
   303-15 (R=14; rule/opt 1.62; stat/opt 1.73; ext-rot/opt 1.66); recorded in the artefact's
   `selection_note`.
2. **"Not enough padded routes" resolved as a RENDERING artefact, verified numerically:**
   every selected instance has its full menu (R = 13-15 = 3 corridors + 10-12 padded;
   R < 15 where the k-shortest generator's near-duplicates dedup). The padded routes share
   59-100% of their edges with the corridor union (median own-edge share 15-50% per
   instance; a few padded routes are pure RECOMBINATIONS of corridor segments with zero own
   edges, still distinct paths), so they draw underneath the bold corridors. Renderer v2
   (`scratch/gen41_render.py`, first version had an edge-key-format defect fixed and
   disclosed) gives each padded route its own colour, draws its non-corridor detour edges
   thick, and annotates each panel with the route count and the median own-edge share.
   This anatomy is BY CONSTRUCTION (k-shortest padding) and is exactly where the gen40
   padding value lives: short detours around punished corridor edges.
3. **Taxonomy clarification (Kilian's question):** the extended rotation is NEITHER an
   avoid-where-ambushed rule NOR a self-tuning rule. It is a MAP-ONLY told rule
   (deterministic cycle over 7-8 edge-diverse routes; consumes the menu and map, no
   outcomes, no tuning, no payoff knowledge), i.e. the same information class as rotation
   and the composed anti-repeat. The adaptive fair-heuristic tier (EXP3 over corridors and
   over the menu, avoid-where-ambushed, and the self-tuning composed rule at a matched
   interaction budget) is a SEPARATE family, deliberately not part of the screen; DRAFT
   addition to the act: these run as REPORTED rows (not gating), evaluated by seeded
   simulation, pending Kilian's sign-off with the bars.
