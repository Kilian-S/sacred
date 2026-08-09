# Generation: gen45_unified_corridor (the Act-4 real-corridor positive, rebuilt on the unified game)

- **status: PRE-REGISTERED 2026-08-09, BEFORE any hunt call, any code build, or any training
  (Kilian's in-conversation go: unify the real-corridor games V2-V4 onto the gen39 substrate;
  keep the pattern-of-life enemy; the full 7-run gen32 ceremony). Training launches are
  Kilian's, per the standing workflow.**
- **git SHA at registration: this commit (aerial).**

## Question

Does the gen32 real-corridor dynamic positive reproduce when its game is rebuilt on the exact
gen39 substrate, so the thesis's Act 4 and Act 5 share ONE game? The consolidation removes the
three version differences Kilian named (grid emplacements, terrain-sets-reach-only with a
replacement effectiveness field, and the ambiguous "searchlight" presentation of the enemy) by
adopting the gen39 forms wholesale. The enemy stays pattern-of-life. Its full-map per-serial
relocation is retained deliberately (Kilian's decision): the gen39 enemy machinery in its
flat, unrestricted-prior limit IS the gen31/32 searchlight (the regression anchor recorded in
`src/envs/aerial_conceal.py`), so Act 4 and Act 5 become one enemy model with one dial, how
far a team may relocate between serials (whole map here; its own ground in gen39).

## The unified game (pinned; substrate knobs FROZEN to gen39's values)

- **Theatre:** kgd_gvardeysk (the Act-4 corridor), `data/maps/theatre_kgd_gvardeysk_vec.json`.
- **Terrain table:** v2 via `terrain_v2(hidden_leth=1.0, conceal_reach=0.85)`, the gen39
  pinned table. Terrain sets reach AND lethality (open 3.5/0.90, field 2.5/0.85, forest
  concealed-reach 0.85 x open at 0.55, urban emplaceable 0.45); forest hides without blinding
  (asymmetric default); urban blocks line of sight with the self-polygon exemption.
- **Range scale:** 0.7 (gen39's RM at the kgd reference, whose lateral-width factor is 1.0).
- **Sites:** the quota sampler, `n_sites=200`, spacing 2.0, standoff 4.0, non-grid points
  whose class shares match the terrain composition (the gen39 form; no raster anywhere).
- **Hidden field:** gen39's `resample_field` MULTIPLIER, band (0.55, 1.0). Lethality =
  terrain class x field draw, never a replacement. The field is what varies per instance, so
  the zero-shot axis and the anti-memorisation control survive.
- **Menu:** as `ConcealBase` builds it on this substrate (14 geometric lanes + terrain-aware
  cover routes screened against the v2 field), pinned by construction at build time.
- **Enemy:** the DOC32 doctrine components q = (0.6 punish-the-window, 0.2 pre-aim-the-escape,
  0.3 anti-repeat-anticipation), softmax tau 0.10, aiming over ALL candidate sites with a
  UNIFORM prior (full-map relocation each serial). This is the gen39 concentration machinery
  in its flat limit and must reproduce the gen32-form dynamics on this substrate; the flat
  limit is regression-tested before any number is read. No reveal channel in this act
  (nothing is emplaced for a whole mission, so there is nothing to reveal); the concealment
  information game remains Act 5's mechanic (gen39).
- **Defender:** fleet N=3, 40-serial episodes, mission damage; head columns per route =
  [exposure, recency (window frequency), doctrine column], the gen32 information channel.
- **Scoring:** exact stationary damage of the policy-induced window chain; exact optimum by
  damped RVI (the repaired form); static CAP = min(iid_eq, static_opt) per field.
- **Window w:** preferred 2 (the gen39 value). Phase 0 may move to 3 (gen32 measured that a
  small safe support makes blind rotation optimal at w=2 on kgd). **Pre-declared freedom is
  ENEMY-BEHAVIOURAL ONLY: q weights, tau, w in {2,3}. The substrate (terrain table, cr, range
  scale, site budget, field band, menu, map) is FROZEN; map or substrate edits are barred.**
- **Field seed ranges (disjoint from every earlier act):** hunt 45001-45012 (burned);
  dev-test/validation 45101-45102 (burned by training diagnostics); GATED pristine
  45200-45205 (confirmation only, never touched by any probe).

## Phase 0: the corridor hunt (oracle-only, free; gates fixed here, before any call)

Over the 12 hunt fields at the candidate operating point(s):

> **G1 (there is a corridor):** static CAP / exact optimum >= 2.0 minimum across fields.
> **G2 (rules cannot collect it):** best payoff-blind dynamic rule / optimum >= 1.25 on
> >= 10/12 fields.
> **G3 (reported, never gating):** the fitted doctrine-informed rules (disclosed oracle caps).

**Fail branch, pre-written.** If no operating point inside the pre-declared enemy freedom
passes G1+G2, the act STOPS with the negative recorded here; gen32 stands as Act 4's banked
result with the game-version difference disclosed in the thesis, and no training is spent.

**Sequencing.** The hunt runs only after the current gen39 step-5d/5e waves clear the Mac, or
with a worker pool capped low enough not to contend with running trainings.

## Training protocol (the full gen32 ceremony, SEVEN runs, Kilian-launched)

- **Build:** a gen45 trainer variant of `train_aerial_dyn32.py` wired to the unified substrate
  (ConcealBase + multiplier field + flat-limit enemy), flag-gated and additive; the flat-limit
  regression test and a green suite are required BEFORE any run; a timing smoke re-measures
  s/sortie before projecting wall clock (the state count depends on the pinned w).
- **Attempt (3 runs):** seeds 0/1/2, 16,000 sorties, validation-selected checkpoints
  (validation on 45101-45102), iteration diagnostics on dev-test only.
- **Confirmation (4 runs, the citable tier):** fresh seeds 10/11/12 plus the BLINDED control
  (seed 10, recency + doctrine head columns zeroed), each evaluated ONCE on the gated set
  45200-45205 via the `--eval-gated` pattern.

> **PRIMARY: zero-shot per-field stationary damage < that field's static CAP on >= 4/6 gated
> fields, on >= 2/3 confirmation seeds, and pooled below the pooled cap.**
> **STRONG: pooled <= 2.5x the exact dynamic optimum.**
> **CAUSAL: the blinded control beats the cap on 0/6 fields (lands at ~cap), with its recency
> and doctrine weights pinned at zero.**
> **REPORTED, never gating:** beats-payoff-blind-family count over the 18 seed-field cells;
> the fitted-rule ladder; the worst-case-vs-committing premium; checkpoint drift.

## Binding comparability rules

1. Nothing here pools with gen31/gen32/gen33 numbers (different game). If this act passes,
   the thesis's Act-4 real-corridor claims rebuild from THIS ledger; the banked
   "real terrain came back tighter than the synthetic testbed (1.30x vs 2.06x)" sentence dies
   unless re-derived against the new numbers, and the gen31 synthetic act is unaffected.
2. The banked gen32 result stays banked in its own ledger regardless of outcome; failures are
   reported here with the same prominence a pass would have had.
3. Every training launch, pause, and resume is Kilian's; this ledger carries the pinned
   commands when the hunt passes.

## Cost, estimated (to be re-measured by the smoke)

Hunt ~2-3 h of oracle compute. Training at gen32's measured ~0.6 s/sortie: ~2.7 h/run solo,
~5 h/run at 4-concurrent; 7 runs = two waves, ~10 h wall clock (an overnight).

## RESULTS (appended below this line; nothing above changes after results exist)

### PHASE 0 RESULT (2026-08-09, oracle-only, 17 s wall; `scratch/gen45_corridor_hunt.py`;
### artefact `models/runs/gen45_hunt.json`, log `models/runs/gen45_hunt.log`): GATES PASS
### AT THE PREFERRED PIN

- **Substrate as frozen:** R=26 routes (14 geometric lanes + 12 terrain-aware), H=200 quota
  sites (open 81 / field 66 / forest 34 / urban 19).
- **Flat-limit anchor (the ledger's required regression, run before any gate number):**
  max |stepdmg difference| 3.9e-12, |history_opt difference| 1.4e-13 between `DynTheatre`
  (the gen32 dynamics) and `ConcealDyn` with one team, huge sigma and no class mask (the
  gen39 machinery's flat limit), same base, same field. The two enemies are the same object.
- **Gates over fields 45001-45012 at DOC32, tau 0.10, w=2:** G1 min 3.71 (bar 2.0; range
  3.71-4.31); G2 >= 1.25 on 12/12 (bar 10/12; range 2.01-2.82); G3 (fitted, disclosed caps)
  median 1.11 with a floor of 1.00-1.01 on two fields (the composed fitted rule attains the
  optimum there, reported per the standing disclosure).
- **PINNED: w=2, DOC32 q=(0.6, 0.2, 0.3), tau 0.10.** Full gen39 consistency; the w=3
  fallback was not needed. The w=2 rotation collapse gen32 measured on the v1 substrate does
  not occur on the unified substrate (no field's best payoff-blind rule comes near the
  optimum), so the corridor here is DEEPER than gen32's original hunt (G1 min 2.67, G2 11/12).
- **Interpreter note (ops):** the aerial framework `python3` in this sandbox lacks site
  packages, so every gen45 command (hunt, suite, smoke, and the pinned launches below) runs
  through `../sacred/.venv/bin/python`, one interpreter for the whole act.

### PROTOCOL COMPLETION (2026-08-09, before any training exists; the registration left the
### training field ranges unnamed)

Train fields 45300-45317 (18), validation 45400-45403 (4); dev-test 45101-45102 and the
gated set 45200-45205 as registered; all disjoint from the burned hunt range 45001-45012.
Trainer `scripts/train_gen45_unified.py`, the gen32 trainer with the substrate swap
(`make_base` + `lethality_for` from the hunt module) and the pinned w=2; everything else,
SAC config, head columns, blind-control semantics, eval machinery, byte-for-byte the gen32
form.

### SUITE + SMOKE (2026-08-09; suite then a 600-sortie plumbing/timing smoke, dev fields
### only, the gated set untouched; exit 0)

- **Suite 246 passed** (94.6 s, `PYTHONPATH=. ../sacred/.venv/bin/python -m pytest tests/`).
- **Pool** (24 exact-ref fields) builds in 36 s; dev/val caps 0.17-0.21, best-blind
  0.11-0.13, hist_opt 0.041-0.053 (the hunt's landscape, reproduced by the trainer's own
  refs).
- **Learning signature (diagnostics, burned fields, NOT citable):** beats-CAP 2/2 by sortie
  320; beats-BLIND 2/2 by sortie 600 at ratio-to-iid 0.46; head weights rw = [+0.42
  exposure, -2.95 recency, -6.74 doctrine] (the doctrine channel engages, the gen32
  signature); alpha healthy at 0.20. Plumbing validated end to end.
- **Timing, measured:** ~1.4 s/sortie at threads 1 while three external gen39 trainers
  occupied the machine; 16,000 sorties projects to ~6-7 h per run under that load, likely
  less on a clearer machine (gen32's 0.6 s/sortie was a solo threads-2 measurement). Each
  wave runs its runs in parallel, so attempt and confirmation are one overnight each.

### PINNED LAUNCH COMMANDS (Kilian launches, per the standing workflow; nothing below has
### been run)

1. Attempt wave (3 runs, seeds 0/1/2, dev diagnostics):
   `cd /Users/kilian/Kilian/ICL/Thesis/code/sacred-aerial && bash scratch/gen45_batch.sh attempt`
2. After the attempt diagnostics are read here and pass, the citable wave (fresh seeds
   10/11/12 + the blinded control, all gated):
   `cd /Users/kilian/Kilian/ICL/Thesis/code/sacred-aerial && bash scratch/gen45_batch.sh confirm`

The script caps every thread pool, runs `../sacred/.venv/bin/python`, writes logs, JSONs and
per-eval checkpoints under `models/runs/gen45_unified/`, and spawns its children from a bash
foreground shell so the interactive-zsh background nice(5) trap cannot apply (verify with
`ps -o pid,nice,command | grep gen45`).

### OPS: THE OVERNIGHT CHAIN (2026-08-09 23:0x; Kilian's instruction, "make sure the commands
### fire when they are supposed to ... I don't want the Mac to idle tonight")

- **Armed:** `scratch/gen45_chain.sh`, detached (PPID 1) at **nice 0**, waits on the three
  gen39 step-5e trainer PIDs (27673/27676/27678, qwenthink16 roll 3) and on the absence of any
  `train_gen39_conceal.py` process, settles 60 s, then runs the pinned ATTEMPT command.
  Hourly heartbeats and the per-seed tail land in `models/runs/gen45_unified/chain.log`.
  5e ETA ~02:00-02:20 (the roll-2 runs took 12,347 s for 5000 sorties at 3-concurrent).
- **The nice trap, measured tonight and worth inheriting:** backgrounding from this
  interactive zsh yields **nice 5** (macOS then prefers efficiency cores, the recorded ~3x
  penalty), while backgrounding inside `bash -c '... &'` yields **nice 0**. Every gen45
  process is launched through bash for that reason and its nice value is verified after
  launch, not assumed.
- **THREADS=2 for tonight's waves, disclosed.** The committed batch script still defaults to
  1; the chain passes `THREADS=2` because the wave has the machine to itself (3 runs x 2 = 6
  of 10 cores). Thread count touches no game quantity, and bit-replay does not exist on this
  stack (gen43), so the reproducibility unit is unchanged.
- **The confirmation wave is NOT auto-chained, deliberately.** It spends the pristine gated
  set once, so it fires only after the attempt diagnostics are read. `scratch/gen45_gate.py`
  is the mechanical form of that read (each seed's validation-selected checkpoint must beat
  BOTH the static cap and the whole payoff-blind family on both dev fields, 3/3 seeds; exit 0
  = fire). It was written and tested tonight against the smoke artefact (structure only, no
  verdict) and will be run when the attempt lands, with its table pasted into the results.
