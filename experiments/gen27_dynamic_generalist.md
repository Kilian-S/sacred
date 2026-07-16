# Generation: gen27_dynamic_generalist (Block R2: zero-shot transferable DYNAMIC hedging — the rescued ZST act)

- **status: PRE-REGISTERED 2026-07-16 (Block R, `NEXT_STEPS_MASTER.md`; Kilian's full autonomous
  launch authority). Binding at launch; smoke gate before the batch.**
- **git SHA:** the commit landing this ledger + `scripts/train_dyn_generalist.py`.

## Why (CRITIQUE_16-07-26.md §6; the disjoint-baseline finding)

The static ZST act cannot claim superiority over naive methods at K=1 (the max-flow heuristic
transfers at 1.13x eq with zero training; R0a appendices). The DYNAMIC register is the one place
the aim's "policies that standard algorithms cannot achieve" is true BY CONSTRUCTION: against a
pattern-of-life adversary (softmax-BR to the defender's realised w-window), every STATIC object
— the disjoint heuristic, the LP equilibrium mixture, distilled or retrieved policies — is
mathematically capped at iid_eq; only history-conditioned play goes below it. gen19 proved the
mechanism single-instance (0.050 vs cap 0.147, causal no-window control, worst-case row); gen16
proved multi-city transfer of static hedging. gen27 composes them: ONE history-aware policy,
trained on three cities, evaluated zero-shot on a held-out city's dynamic games.

## Design (locked; gen19 + gen16 recipes verbatim, decisions recorded)

- **Game per instance:** fleet-route stacked; adversary = softmax-BR (tau=0.15) to the trailing
  w=3 realised-route window — the gen19 operating point, chosen from its banked sensitivity grid
  (tau=0.05 is degenerate-dodgeable; w=3 the non-degenerate point). Analytic expected-mission-
  failure reward; episodes = S=40 sorties chained with gamma=0.95; window cleared per episode.
- **Pools:** gen16 verbatim — kaliningrad + east_london + istanbul x 6 ODs (train),
  GDANSK x 6 held out entirely; pool-seed 0; N=3, K=1, band 0.15-0.95, k8 menus.
- **Conditioning:** per-instance menus + [cost, worst-vuln, window-frequency] route features on
  every observation (the gen15 per-transition plumbing + the gen19 third column), head-term lr
  3e-2. NO route_bias (identity does not transfer).
- **Yardsticks per instance (computed at pool build, exact):** static_det, iid_eq (the static
  cap), history_opt (RVI over the R^w window MDP) — the gen19 oracle, per instance.
- **Arms:** history-aware x seeds {0,1,2}; NO-WINDOW causal control x 1 seed (window feature
  zeroed; must land ~iid_eq as in gen19).
- **Selection:** select-on-TRAIN (mean train-instance ratio-to-iid_eq; standing default);
  select-on-test dual-reported as the optimistic bound; final iterate disclosed.
- **Budget:** 12,000 sorties (the gen16 budget), eval-every 500 (train n=400/instance, held-out
  n=1000/instance), per-eval ckpts. Timing probe + 240-sortie smoke gate BEFORE the batch
  (gate = plumbing sound + rw[2] trending negative, the gen19 anti-repeat signature; NOT a
  performance bar — smokes validate plumbing, not dynamics).

## Decision metric (PRE-REGISTERED)

Per held-out instance: stationary per-sortie expected mission failure vs ITS pattern-of-life
adversary (the gen19 estimator), ratio to ITS iid_eq.

> **PRIMARY (the unique claim): at the select-on-train checkpoint, held-out-GDANSK mean
> ratio-to-iid_eq < 1.0 AND < 1.0 on >= 4/6 ODs, on >= 2/3 seeds.** Beating iid_eq zero-shot =
> beating every static method (heuristic, LP, distillation, retrieval) by construction.
> **STRONG:** pooled held-out mean per-sortie loss <= 2x the per-instance history_opt mean.
> **CAUSAL CONTROL:** the no-window arm lands at ratio ~1.0 (the cap; gen19's landed 1.007).
> **REPORTED ROW (not gated):** worst-case = the marginal route mixture's single-shot
> exploitability under each OD's oracle BR vs its V_eq (gen19's premium was +6%; zero-shot will
> be looser; reported honestly).
> **Branches:** PASS = the crown jewel (zero-shot transferable dynamic hedging). PARTIAL
> (train-city ratios < 1, held-out >= 1) = dynamic hedging learned, transfer boundary measured.
> FAIL = gen19 stays the single-instance positive; the boundary is the result. All writable.

## Commands (pinned; batch via `scratch/gen27_batch.sh` after the smoke gate)

```bash
PYTHONPATH=. .venv/bin/python scripts/train_dyn_generalist.py \
  --sorties 12000 --eval-every 500 --seed $S --threads 3 \
  --json-out models/runs/gen27_dyn_generalist/seed$S.json \
  --ckpt-dir models/runs/gen27_dyn_generalist/seed${S}_ckpts
# control: --no-window --seed 0, json/ckpt paths suffixed _nowin
```

## RESULTS (appended per step; nothing above changes after launch)

### AMENDMENT (2026-07-16, DURING the batch, BEFORE any result was read; disclosed)

Two integrity additions, both decided before looking at any training outcome:
1. **Per-seed yardsticks + the LP-degeneracy wobble.** The three seeds' pool-prep logs show
   iid_eq differing by ~1-2% across processes on identical games (e.g. OD 193-278: 0.187 vs
   0.184): the equilibrium LP has degenerate optima and the solver's vertex choice is not
   process-stable. static_det and history_opt are identical (they do not depend on the vertex).
   Handling: each seed's ratios are scored against ITS OWN stored refs (as built); the wobble is
   disclosed and immaterial at the expected effect size (gen19: 66% below the cap).
2. **The "every static method" claim gets MEASURED rows, not by-construction wording.** iid_eq
   is the static EQUILIBRIUM mixture's value against this adversary — not the best static value.
   The pre-registered primary (beat iid_eq) therefore only certifies beating the LP mixture. At
   results time the following oracle-exact rows are added per held-out instance (eval-only,
   `scratch/gen27_static_rows.py`): the UNIFORM-DISJOINT heuristic's static value, the INV-VULN
   heuristic's static value, and a multi-start local-search STATIC OPTIMUM (disclosed as local).
   The act's claim is worded against whichever of these rows the results actually clear; the
   primary bar itself is unchanged (iid_eq).

### SECOND AMENDMENT (2026-07-16, second critic pass, DURING the batch, BEFORE any result was
### read; disclosed): the naive-DYNAMIC baseline rows

The first amendment's static rows do not close the sharpest hole: a naive DYNAMIC heuristic
exists, and it is the dynamic analogue of the disjoint-stack finding. Measured this session
(oracle-exact, `scratch/critique_followup_probes.py`, artefact
`models/runs/critique_followup_probes.json`), on the SAME 6 held-out Gdansk instances against
the SAME adversary (w=3, tau=0.15), with history_opt recomputed by RVI on the same L
(same-convention anchor):

| OD | iid_eq | history_opt | rotation (round-robin over disjoint) | **anti-repeat (uniform over disjoint routes not in the last-w window)** | anti/iid_eq |
|---|---|---|---|---|---|
| 249-95 | 0.223 | 0.057 | 0.207 | **0.118** | 0.53 |
| 106-173 | 0.213 | 0.072 | 0.203 | **0.119** | 0.56 |
| 351-210 | 0.232 | 0.078 | 0.206 | **0.117** | 0.50 |
| 146-296 | 0.189 | 0.064 | 0.196 | **0.115** | 0.61 |
| 275-72 | 0.218 | 0.051 | 0.203 | **0.115** | 0.53 |
| 193-278 | 0.187 | 0.075 | 0.186 | **0.112** | 0.60 |

**Binding wording rule (analogous to the R0a rule for the static acts):** beating iid_eq does
NOT certify beating every standard method. The two-line anti-repeat heuristic (needs only the
route list, zero training, transfers trivially) beats iid_eq by ~2x on every held-out OD
(ratio 0.50-0.61). All six ODs have m=3 disjoint routes, so deterministic rotation fails (the
w=3 window covers the whole disjoint set) but stochastic anti-repeat does not. The PRE-REGISTERED
PRIMARY BAR IS UNCHANGED (bars are never moved mid-run); the anti-repeat row is a REPORTED ROW
beside it, and the act's comparative wording must clear whichever rows the results actually
clear: the honest ladder per OD is iid_eq > anti-repeat (~0.55x) > history_opt (~0.25-0.35x).
A result near history_opt beats the heuristic decisively; a result near ~0.55x iid_eq matches
it and the claim re-scopes to label-free amortisation (the gen24/A6 pattern).

**Companion note for the gen19 ledger (measured in the same probe):** on 35-159 (m=4), plain
rotation over the 4 disjoint routes lands at 0.0413 = 1.07x the same-convention dynamic optimum
(RVI on the same L: 0.0388; the ledgered 0.049 is the same quantity under the screen's original
game build). gen19's "reaches the dynamic optimum" result is real, but on that instance the
optimum itself is nearly attained by a two-line rotation; the unique learning content is (a)
discovering anti-repeat behaviour without being told its form, and (b) the m=3 regime (all of
gen27's held-out pool) where rotation fails and calibrated stochastic anti-repeat is required.

### RESULT: the three history-aware seeds (2026-07-16 evening, 12,000 sorties each, SHA `2f4ffd5`-era + amendments): **PRIMARY PASSED on 3/3 seeds; STRONG PASSED**

(select-on-train = the standing deployable selection; per-seed refs per the amendment;
artefacts `models/runs/gen27_dyn_generalist/seed{0,1,2}.json`)

| seed | sel-on-train @ sortie | held-out mean ratio-to-cap | beats cap | vs history_opt | sel-on-test | final iterate |
|---|---|---|---|---|---|---|
| 0 | 11,000 | **0.605** | 6/6 | 1.66x | 0.602 | 0.647 |
| 1 | 9,520 | **0.644** | 5/6 | 1.76x | 0.615 | 0.816 |
| 2 | 11,000 | **0.666** | 5/6 | 1.81x | 0.630 | 0.631 |

> **Pooled held-out ratio-to-iid_eq 0.639 +/- 0.025 (3 seeds). PRIMARY (mean < 1.0 AND < 1.0 on
> >= 4/6 ODs, >= 2/3 seeds): PASS on every clause, 3/3 seeds. STRONG (<= 2x history_opt): PASS
> (mean 1.74x).** Select-on-test agrees with select-on-train (0.602-0.630 vs 0.605-0.666: no
> test-selection flattery); final-iterate drift mild (seed 1 to 0.816), disclosed as standing.
> The one hard OD (index 1, 106-173) sits at 0.90-1.07 across seeds: reported per-OD, no
> averaging-away.

**The static-baseline rows (oracle-exact, `static_rows.json`, pre-registered by the amendment):**
on every held-out OD the LOCAL-SEARCH STATIC OPTIMUM improves on the equilibrium-mixture cap by
only 2-5% (0.179-0.227 vs iid_eq 0.187-0.236), and both max-flow heuristics' static values sit
within +/-5% of the cap. **So beating the cap at 0.639 beats EVERY static object by a wide
measured margin — the LP mixture, both disjoint heuristics, and the locally-optimal static
mixture — as measurement, not construction.**

**The naive-DYNAMIC rows (the second pass's bar, reconciled 2026-07-16 late):** two variants,
both two-line rules:
- *anti-repeat over the FULL menu* (uniform over routes not in the last-3 window): **mean 1.368x
  the cap (WORSE than static play on 5/6 ODs)** — on shared-edge menus, avoiding your recent
  ROUTES still lands you on their shared SEGMENTS; naive anti-repeat fails structurally.
- *anti-repeat over the DISJOINT routes* (the COMPOSITION of both known insights: independence +
  anti-repeat; the second pass's exact power-iteration row): **0.50-0.61x the cap** — better
  than the trained policy's 0.639 pooled.

**The honest claim this act banks (binding wording):** *zero-shot on a never-seen city, one
history-aware policy beats every static method — including the locally-optimal static mixture —
by ~36%, reaches 1.74x the exact dynamic optimum, and matches (from slightly above) the band of
the composed independence+anti-repeat rule, having DISCOVERED both of that rule's insights from
adversarial experience alone: label-free, structure-untold.* The composed rule itself needs both
insights handed to it; the policy found them. Sentences claiming the policy BEATS every simple
dynamic rule are NOT licensed; sentences claiming no static or naive-dynamic single-insight rule
touches it are.

**Worst-case row + no-window causal control: appended when their runs complete (in flight).**

### Worst-case row (2026-07-16 night, eval-only, `worstcase.json`; the pre-registered reported row)

Seed-0 select-on-train checkpoint, marginal route mixture per held-out OD under that OD's ORACLE
best response, vs its single-shot stacked V_eq: premiums **1.43 / 1.91 / 1.56 / 1.60 / 1.49 /
1.44x (mean ~1.57x)**. gen19's in-distribution premium was 1.06x; zero-shot the dynamic policy
pays a REAL worst-case premium — it is specialised to exploit the adaptive adversary. Honest
scope sentence (binding): *the dynamic policy's advantage is regime-conditional: against a
pattern-of-life adversary it reaches 0.64x the static cap; against a worst-case committing
adversary its marginal is ~1.5x the equilibrium, so the static hedge remains the right play
when the adversary best-responds to the strategy rather than the pattern.* Both policies exist;
choosing between them is an intelligence question, not a modelling one.
