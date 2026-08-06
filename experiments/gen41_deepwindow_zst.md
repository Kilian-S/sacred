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

## BINDING AT LAUNCH (2026-08-05; Kilian's go: K=2 confirmed, tiered primary confirmed,
## padding stays, full-capacity machine use, 3 seeds + control, autonomous execution)

**Config pinned.** Pools = the 24 reviewed ODs (`models/runs/gen41_pool.json`, built from the
screen artefact post-swap). w=6, tau=0.15, K=2, kx=12, N=3, band (0.15, 0.95), mission
objective; 12,000 sorties, episode 40, gamma 0.95, head-term lr 3e-2, batch 32; seeds
{0, 1, 2} + no-window causal control (seed 0); per-eval checkpoints; select-on-train.
References per run via `--fast-refs` (count-class exact iid_eq + corridor-core Karp; the
LP-degeneracy vertex wobble ~1% is disclosed and each run scores against its own refs, the
gen27 amendment-1 convention). Eval cadence 500; PER-ROUND eval sampling trimmed to 250
(train) / 600 (held-out) sorties per instance for wall-clock (selection only; DISCLOSED
deviation from gen27's 400/1000); the FINAL cited numbers come from a separate
high-precision pass (20,000 rollout sorties per held-out instance at the selected
checkpoint). Suite green after the trainer changes: **167 passed** (pytest raw tail pasted
in-session, 2026-08-05; the tests/ tree last changed at gen26's `77fe57f`, so the
HANDOVER's "224" roads figure is a doc slip to fix, the aerial count).

**Smoke gate: PASSED** (240 sorties, pinned command with `--sorties 240`): pool builds with
correct refs, training and eval run end-to-end, and the window-column weight rw[2] trained
to -3.09 within 240 sorties (the gen19/gen27 anti-repeat signature; smokes validate
plumbing, not dynamics).

**PRIMARY (binding).** At the select-on-train checkpoint, on the 6 held-out Gdansk
instances: SACRED's pooled mean ratio-to-cap (i) beats the static cap on >= 4/6 ODs,
(ii) is below EVERY Tier-0 dynamic rule's pooled ratio (corridor rotation, full-menu
rotation; screen values stand), and (iii) is below EVERY Tier-2 adaptive row's pooled
GENEROUS score (defined below), all on >= 2/3 seeds.
**STRONG:** per-sortie loss at or below the corridor-restricted exact optimum on >= 3/6
held-out ODs (high-precision pass).
**CAUSAL CONTROL:** the no-window arm lands ~1.0x cap and beats it nowhere.
**REPORTED (never gating, never dropped):** Tier-1 rows (composed anti-repeat, extended
rotation; screen values), worst-case one-shot exploitability of the marginal mixture,
final-iterate drift, per-OD values, select-on-test dual-report.

**Tier-2 adaptive rows (binding definitions; all seeded, simulation where not exact).**
(i) EXP3 over the full menu and (ii) EXP3 over the corridors: standard EXP3, eta =
sqrt(ln n_arms / (n_arms T)), T = 12,000 sorties learned IN PLACE on each held-out
instance, bandit feedback = the analytic expected loss of the chosen route; scored on the
final-2,000 tail (generous) with the full-horizon mean reported; 5 seeded repetitions.
(iii) Avoid-where-ambushed: Bernoulli interception outcomes sampled from the committed
set; uniform over routes without a realised interception in the last h sorties (fallback
uniform); h in {3, 6, 12}, best taken (generous); 5 repetitions, tail scored.
(iv) Self-tuned composed rule: composed anti-repeat at defender window w' in {2, 4, 6, 8},
each EXACT (stationary chain over the max(w', 6)-window core states), best w' taken (the
self-tuned upper envelope). The gate uses the generous readings throughout.

**Commands (pinned; batch `scratch/gen41_batch.sh`, detached nohup + disown, thread pools
capped per SYSTEM):**
```bash
OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. .venv/bin/python \
  scripts/train_dyn_generalist.py --pool-file models/runs/gen41_pool.json \
  --K 2 --k-extra 12 --window 6 --fast-refs --sorties 12000 --eval-every 500 \
  --eval-n 600 --eval-n-train 250 --seed $S --threads 2 \
  --json-out models/runs/gen41_deepwindow/seed$S.json \
  --ckpt-dir models/runs/gen41_deepwindow/seed${S}_ckpts
# control: identical + --no-window --seed 0, paths suffixed _nowin; all four concurrent
```

## RESULTS (appended per step; nothing above changes after each step runs)

### RUN STATE: PAUSED ON KILIAN'S INSTRUCTION (2026-08-05 ~21:20)

The four training processes (3 seeds + no-window control, PIDs 196-199) are SIGSTOPPED at
sortie ~6,500-7,000 of 12,000 with state in memory; all monitoring watchers cancelled.
Resume = `kill -CONT` on the four PIDs (the batch script's `wait` parent is untouched), at
which point training continues losslessly; the machine must not reboot while paused. At
pause: held-out ratios 1.12-1.20 (beats-cap 1-2/6 per seed), train 0.92-1.15, rw[2] at
-28 to -37 and deepening, alphas 0.19-0.26, control clean (window weight 0.00). Process
note for the record: the processes carry a positive nice value (STAT flag N), applied by
the launching shell's background policy, not by any command of ours; renicing upward needs
privileges, so the pace implication (~30 min per eval cycle) is recorded rather than
repaired. Tier-2 rows are already banked (`tier2_rows.json`): pooled ratios-to-cap EXP3-menu
0.995, EXP3-core 1.042, avoid-where-ambushed 1.327, self-tuned composed 0.932 (the gate's
binding Tier-2 value). Final evaluation runs after Kilian orders the resume and the batch
completes.

## ACT 2 (design revision, same generation per Kilian 2026-08-06): the w=3, K=2
## padding-channel act, GATED. Gate 1 pre-registered here BEFORE any gate CPU.

**Revised operating point:** w=3 (back inside the proven band, where the frequency channel
demonstrably carries), K=2, kx=12, m=3, same 24 reviewed pools, tau 0.15, N=3. Rationale
recorded in-session: at (m=3, w=3, K=2) the corridor-locked class is structurally capped
~1.2x the true optimum (the gen40 padding channel, 14-21%), the true full-menu optimum is
exactly computable (Karp on R^3), and the winning mechanism is frequency-expressible,
unlike the w=6 cell that failed. NOTHING TRAINS unless Gate 1 and then the single-instance
rung (Gate 2) pass.

**GATE 1, the representability certificate (oracle-only, all exact or exact-valued
witnesses).** Per instance at (w=3, K=2): the full-menu exact optimum (Karp), the
corridor-restricted optimum, the exact iid_eq, and the complete w=3 rule family re-measured
(rotation, composed anti-repeat at w' in {1,2,3,4} exact, extended rotation, full-menu
rotation, statics). Then two witnesses whose values are EXACT (deterministic policies,
cycle-walk evaluation, multi-start): (a) the LINEAR-FEATURE witness, argmax policies over
the head's own per-route features [normalised cost, normalised worst-vulnerability,
window-frequency] on a theta grid with local refinement, a certified SUBSET of the trained
architecture's policy class; (b) the COUNT-CLASS witness, coordinate descent over
count-signature-conditioned policies seeded from the exact optimal policy's projection and
from (a), an upper envelope of the frequency-conditioned family. Disclosed scope: (a)
proves expressibility inside the architecture; (b) bounds what any frequency-reading
policy could reach; neither is a training guarantee, both are existence certificates.

> **GATE 1 BARS (binding before results): PASS = the linear-feature witness value beats
> the best composed-rule variant (w' in {1,2,3,4}) on >= 4/6 held-out instances AND >= 12/18
> training instances, AND its pooled held-out ratio-to-cap is below the composed family's
> pooled best. FAIL = the current feature language cannot win at this cell either; the
> recorded consequence is redirect-to-feature-design, no training anywhere.** The
> count-class witness and the full-menu optimum are reported beside, never gating.

Script `scratch/gen41_repr_gate.py`; artefact `models/runs/gen41_repr_gate.json`.

### GATE 1 RESULT (2026-08-06, 66 s, oracle-only): **PASS on every clause, 6/6 held-out
### and 18/18 train**

Pooled held-out ratios-to-cap: full-menu exact optimum **0.439**; count-class witness
**0.441** (attains the optimum to <1% on every instance: at w=3 frequency information
SUFFICES); linear-feature witness **0.478** (within ~9% of the optimum everywhere);
composed-rule family best 0.656; corridor rotation 0.99-1.03. The winning linear weights
are nearly universal, (cost 0, vuln 0, freq -40) on 19/24 instances: pure
frequency-avoidance over the FULL menu, exactly the weight the architecture trains, now
certified to carry the win at this operating point (the precise inversion of the w=6
channel-content failure).

**Consequences, binding.** (i) The win exists inside the architecture's certified feature
subset with a 27% pooled margin over the best told rule; training the act is justified.
(ii) The witness doubles as a NEW two-line rule (in effect a rotation over the corridors
plus the safest padded route, staying outside the w-window; requires knowing w, hence
Tier 1): it enters the act's Tier-1 REPORTED family at its measured values (held-out
pooled 0.478x cap) per baseline completeness, and the act's PRIMARY still gates on Tiers
0 and 2 only (the like-with-like structure Kilian approved). (iii) Tier-2 adaptive rows
must be RE-MEASURED at (w=3, K=2) before the act's verdict (the banked values are w=6).

### GATE 2 (pre-registered; the single-instance rung; Kilian launches per the standing
### workflow)

One training-city instance (kaliningrad 23-242, the reference), 3 seeds, 8,000 sorties,
gen41 trainer flags with a single-instance pool file (`models/runs/gen41_rung_pool.json`,
train = test = that instance), eval cadence 500 at 400/400.
> **BARS: PASS = best-checkpoint ratio-to-cap <= 0.645 (the composed family's value on
> this instance) on >= 2/3 seeds. STRETCH (reported): <= 0.50 (near the witness/class
> ceiling). FAIL = the rung result becomes the recorded boundary; no transfer act.**
Only a PASS unlocks the 24-instance transfer act (whose full bars will be finalised then,
Tier-2 w=3 rows included).

### GATE 2 RESULT (2026-08-06, Kilian-launched, 3 seeds x 8,000 sorties): **PASS 3/3**

Best-checkpoint ratio-to-cap 0.526 / 0.511 / 0.499 (bar 0.645, needed 2/3; the 0.50
STRETCH touched by seed 2 and near-touched by seed 1); beats-cap 1/1 every seed; mild
drift (finals 0.511-0.570); select-on-test agrees (0.496-0.526). The trained weights
reproduce the certified mechanism plus a refinement (freq -23, vuln -4). The rung
collected most of the certified headroom (class ceiling 0.42 on this instance).

### TIER-2 ROWS AT w=3 (2026-08-06, oracle/sim per the binding definitions; artefact
### `tier2_rows_w3.json`): pooled ratios-to-cap EXP3-menu 0.984, EXP3-core 1.032,
### avoid-where-ambushed 1.299, **self-tuned composed 0.822 (the binding Tier-2 gate)**.

## ACT 2 TRANSFER BARS (BINDING AT LAUNCH; both gates passed; Kilian launches per the
## standing workflow)

Config identical to the Act-1 batch except `--window 3` (pool file, K=2, kx=12, 12,000
sorties, eval 500 at 250/600, seeds {0,1,2} + no-window control, select-on-train,
high-precision final pass at 20,000 sorties per held-out instance).

> **PRIMARY: at the select-on-train checkpoint, held-out pooled mean ratio-to-cap
> (i) beats the cap on >= 4/6 ODs, (ii) is below every Tier-0 pooled value (corridor
> rotation 1.006; full-menu rotation per the Gate-1 artefact), and (iii) is below every
> Tier-2 pooled value, binding member the self-tuned composed at 0.822; all on >= 2/3
> seeds. STRONG: pooled below the Tier-1 composed family's 0.656 AND beats it on >= 4/6
> instances (the sentence Act 3 of the thesis currently cannot say). REPORTED, never
> gating: the witness rule at 0.478 (the class ceiling; matching it is the discovery
> claim), the extended rotation, worst-case one-shot row, final-iterate drift.
> CAUSAL CONTROL: the no-window arm lands ~1.0x cap.**

Launch command (Kilian; verify nice values via the trailing check):
```bash
mkdir -p models/runs/gen41_act2 && for S in 0 1 2; do OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. nohup .venv/bin/python scripts/train_dyn_generalist.py --pool-file models/runs/gen41_pool.json --K 2 --k-extra 12 --window 3 --fast-refs --sorties 12000 --eval-every 500 --eval-n 600 --eval-n-train 250 --seed $S --threads 2 --json-out models/runs/gen41_act2/seed$S.json --ckpt-dir models/runs/gen41_act2/seed${S}_ckpts > models/runs/gen41_act2/seed$S.log 2>&1 & done; OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. nohup .venv/bin/python scripts/train_dyn_generalist.py --pool-file models/runs/gen41_pool.json --K 2 --k-extra 12 --window 3 --fast-refs --sorties 12000 --eval-every 500 --eval-n 600 --eval-n-train 250 --no-window --seed 0 --threads 2 --json-out models/runs/gen41_act2/seed0_nowin.json --ckpt-dir models/runs/gen41_act2/seed0_nowin_ckpts > models/runs/gen41_act2/seed0_nowin.log 2>&1 & sleep 5; ps -o pid,nice -p $(pgrep -f train_dyn_generalist | tr '\n' ',' | sed 's/,$//')
```

### FINAL RESULT (2026-08-06; batch completed overnight; high-precision pass 20,000
### rollout sorties per held-out instance at the select-on-train checkpoint;
### artefacts `final_eval_seed{0,1,2,0_nowin}.json`, `tier2_rows.json`)

**PRIMARY: FAIL, 0/3 seeds, on every clause. STRONG: FAIL, 0/6 everywhere.** The verdict
table (pooled ratio-to-cap over the 6 held-out Gdansk instances):

| object | pooled ratio-to-cap |
|---|---|
| corridor-restricted exact optimum | **0.576** |
| self-tuned composed rule (Tier 2, the binding gate) | 0.932 |
| composed anti-repeat w'=6 (Tier 1, reported) | 0.941 |
| EXP3 over the menu (Tier 2) | 0.995 |
| corridor rotation (Tier 0) | 1.006 |
| extended rotation (Tier 1, reported) | 1.030 |
| **SACRED seed 2 / 0 / 1 (select-on-train)** | **1.031 / 1.163 / 1.252** (pooled 1.149) |
| no-window causal control | 1.278 |
| avoid-where-ambushed (Tier 2) | 1.327 |
| full-menu rotation (Tier 0) | 1.363 |

Per-seed: beats the cap on 2/6 ODs each; at-or-below the corridor-locked optimum 0/6
everywhere; worst-case one-shot premiums 1.32-1.40x. The select-on-test optimistic bounds
(1.021-1.115) never approached the gate either, so no selection ambiguity exists. The
window weight trained to -34 to -52 on every sighted seed and 0.00 on the control.

**Reading 1, the causal-control nuance (the sharpest finding).** Unlike gen27 (blind 1.434
vs sighted 0.639, a 0.8-ratio causal gap), the w=6 window channel bought only 0.03-0.25 of
pooled ratio (control 1.278 vs 1.031-1.252), with seed 1 statistically at the control's
level. The recency-frequency column is nearly INERT for collecting deep-window value even
though the policy leans on it maximally: pure anti-repeat behaviour is worth little at
w = 2m (every corridor is always punished), and an aggregate frequency column cannot
express the window-steering cycles the optimum uses. A CHANNEL-CONTENT failure, not a
training failure; the gen34/gen36 conditioning-capacity wall, met from a new direction.

**Reading 2, the register-wide fact.** At w = 2m EVERY practical object sits at 0.93-1.36x
the cap while the exact optimum sits at 0.576x: the deep-window value is real, large, and
uncollected by anything measured, told rules, adaptive no-regret learners, and
recency-conditioned deep RL alike. Combined with gen40 and gen27, the w-axis map is now
complete and citable: at w = m-1 rotation is provably optimal and learning is pointless;
at w = m calibrated history-conditioning wins and transfers (gen27, 0.639x cap zero-shot);
by w = 2m no tested policy class collects the value. **Learning pays in a window BAND
around w ~ m, and the band's far edge is a conditioning boundary, not a game boundary.**

**Binding wording.** No sentence may claim SACRED beats any rule tier at w=6; the licensed
claims are the band map above, the channel-content mechanism (control-attributed), and the
fairness-tier landscape (Kilian's like-with-like framing survives intact: at the deep
window the told-rules' advantage evaporates along with everything else). Future work,
recorded not run: per-lag route-identity window features (sequence, not frequency), or
short-horizon planning over the window MDP the features already determine.

**Process disclosures.** Run at nice 5 until ~sortie 7,000 (wall-clock only); two
externally-reaped watchers (no effect on the runs); the 20,000-sortie rollout estimator
per the pre-registration; suite 167 green at the launch SHA; every bar judged exactly as
pre-registered, no bar moved.

### RUN STATE: RESUMED (2026-08-05 ~21:20, Kilian's instruction) + the nice disclosure

All four processes SIGCONTed and running. **Nice disclosure (binding for any timing
sentence about this run):** the batch has run at nice 5 from launch, applied silently by
the launching shell's background policy, first noticed at the pause; un-nicing to 0
requires root and was handed to Kilian in-session (`sudo renice -n 0 -p 196 197 198 199`),
so the run may be nice-5 up to ~sortie 7,000 and nice-0 after. This affects WALL-CLOCK
ONLY, never results (no timing claim is made anywhere in this act); the SYSTEM
"never nice training runs" dogma gains a corollary recorded here: zsh background launches
can nice silently, so future launches must verify `ps -o nice` immediately after start.
Monitoring policy per Kilian: no periodic watching; ONE completion-only watcher armed so
the final evaluation sequence fires when the batch ends. **Workflow change for ALL FUTURE
RUNS (Kilian, 2026-08-05, recorded in agent memory as binding): Claude prepares and
outputs the pinned launch command; KILIAN launches, pauses, and resumes runs himself and
checks in periodically for Claude to inspect state; Claude arms no watchers of its own.**

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
