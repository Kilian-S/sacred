# Generation: gen11_menuhead (menu-head discriminability: can the post-fix multi-convoy leader reach the pre-fix headline on honest representations?)

- **status: PRE-REGISTERED 2026-07-10 (Kilian's explicit overnight launch authority + decision
  rules given in-conversation; binding at launch).** Results appended below.
- **git SHA:** pinned by the commit landing this ledger together with the gen11 machinery.

## Why (from CRITIQUE_PREFREEZE.md §2, §5.1, §8.2)

The pre-fix multi-convoy headline (exact 0.295 +/- 0.024, SHA `ad70a9c`) was produced under the
node-ordering bug, whose fixed permutation gave every route a distinct identity-hash signature: a
tabular learner over 12 abstract arms. Post-fix, the honest mean-pooled route embeddings of
overlapping routes are near-identical (route Jaccard mean 0.38, max 0.88 on 62-97 k8) and the
pipeline lands at a reproducible 0.447 plateau (gen10-MC 0.447 +/- 0.029; gen10-MC2 0.447 +/-
0.008 with the role-target rule reverted and the horizon doubled). Two mechanisms remain
unseparated inside that plateau:
- **(head)** the parameter-free menu head cannot discriminate overlapping routes on honest
  embeddings;
- **(push)** fleet-route mode pushes the followers' no-decision transitions (tagged with
  near-zero entropy targets after `follower_warmup`) into the same shared actor that must hold
  the leader at 0.5*lnR, on states differing only in the correlation signal: the observed
  H_lead=0 + alpha-runaway saturation parks are what this conflict predicts.

gen11 tests both, factorially, with the smallest principled additions. The feature arm is ALSO
the ZST map-conditioning mechanism (per-route cost/vulnerability delivered undiluted at the head),
so a pass here unblocks ZST step 1 by construction.

## Arms (all on the gen10-MC config, otherwise byte-identical; 3 seeds {0,1,2} each)

| arm | flags | tests |
|---|---|---|
| A (baseline) | - (= gen10-MC, already run) | the 0.447 plateau reference |
| **B (features)** | `--route-feats` | LEARNED weights (init 0) on per-route normalised COST + worst-case VULNERABILITY at policy AND critic heads (lever-2 pattern; Bellman-consistent) |
| **C (leader-only push)** | `--leader-only-push` | fleet-route pushes ONLY the leader's decision, terminal with the sortie reward (kills the follower-push conflict; one update/sortie unchanged) |
| **D (both)** | `--route-feats --leader-only-push` | the combination |
| **E (identity)** | `--route-bias` | LEARNED per-route scalar bias (init 0) = pure identity capacity, reconstructing exactly what the bug accidentally provided; separates "identity capacity" from "transferable features" |

Config: 62->97 k8 menu-select, band 0.15-0.95, N=3, K=1, fleet-route, smooth FP tau 0.05,
switch-every 200, smooth-window 250, leader-ent-frac 0.5, leader-alpha-floor 0.20, 1200 sorties,
eval-every 100, EXACT estimator, per-eval checkpoints saved, `--threads 3`, 3-parallel per arm
(arms staged serially).

## Decision metric (PRE-REGISTERED; decision rules fixed by Kilian in-conversation 2026-07-10)

Primary per arm = **exact best-checkpoint TAP, mean +/- pop std over 3 seeds** (selection rule
unchanged: lowest trailing-averaged-policy exploitability under the oracle BR interdictor,
TAP_K=5), against the oracle ladder (ALNS 0.699, equilibrium 0.216) and the two references
(pre-fix exact 0.295 +/- 0.024; post-fix plateau 0.447).

- **PASS = any arm's mean <= 0.295.** Consequence (Kilian, pre-authorised): that arm's result
  BECOMES the new multi-convoy headline (post-fix, honest representations; the pre-fix number and
  its caveat retire to the methods narrative). If several arms pass, the headline arm is the one
  with the LOWEST mean; ties broken toward B over D over C over E (fewer mechanisms / more
  transferable content; E passing alone is explicitly NOT a headline, see below).
- **E's special status (recorded up front):** arm E passing while B fails would show the plateau
  is pure identity capacity, i.e. the game is learnable tabularly but not via transferable
  features; E is a METHODS exhibit either way and is never the headline (it has no map semantics).
- **Partial (arms beat 0.447 but none reach 0.295):** the pre-fix headline stands with its caveat;
  the factor decomposition (B vs C vs D vs E deltas) is the reportable product; NO further design
  iteration tonight (Kilian's no-chasing bound); recommendations for further attempts written for
  the morning report.
- **Interpretation reads (secondary):** C vs A isolates the push conflict; B vs A the head
  features; D vs B+C interaction; E vs B identity-vs-features. Alpha/H_lead trajectories reported
  per arm (the saturation-park signature).

**Recorded prediction (before results):** C alone moves the plateau but not to 0.295 (the conflict
is real but not the whole story); B or D is the most likely pass; E lands near the pre-fix number
(identity capacity was what the bug provided) - if E << B, the honest conclusion is that route
identity, not route semantics, is what the current architecture can exploit, and that goes in the
thesis verbatim.

## Commands (pinned; via `scratch/gen11_orchestrator.sh`, detached, outputs under `models/runs/gen11_menuhead/`)

```bash
# per arm (flags per the table), per seed S in {0,1,2}, 3-parallel within an arm, arms serial:
PYTHONPATH=. .venv/bin/python scripts/train_multiconvoy.py \
  --od 62-97 --N 3 --K 1 --k-extra 8 --menu-select --band 0.15,0.95 \
  --fleet-route --attacker-mode smooth --fp-tau 0.05 --switch-every 200 --smooth-window 250 \
  --leader-ent-frac 0.5 --leader-alpha-floor 0.20 --sorties 1200 --eval-every 100 \
  <ARM FLAGS> --seed $S --threads 3 \
  --json-out models/runs/gen11_menuhead/<arm>_seed$S.json \
  --ckpt-dir models/runs/gen11_menuhead/<arm>_seed${S}_ckpts \
  > models/runs/gen11_menuhead/<arm>_seed$S.log 2>&1
```

Machinery this commit: `_route_head_terms` (networks.py; also applied in ProtagonistQNet.head),
`--route-feats` / `--route-bias` / `--leader-only-push` (train_multiconvoy.py; registration order
of the new parameters matched across q/target nets for `_soft_update`), 3 regression tests
(`tests/test_route_head_terms.py`); suite **152 passed**; all additive/flag-gated (absent flags =
byte-identical, incl. the banked paths).

## RESULT (2026-07-10 02:20, 12 runs, ~54 min at 3-parallel staged): NO ARM PASSES; the decomposition is the product

| arm | best-ckpt TAP per seed | mean +/- std | final TAP mean | read |
|---|---|---|---|---|
| A (baseline, = gen10-MC) | 0.478 / 0.409 / 0.454 | 0.447 +/- 0.029 | 0.788 | the plateau |
| B (cost+vuln features) | 0.410 / 0.441 / 0.479 | **0.443 +/- 0.028** | 0.708 | = plateau; `route_feat_w` ended at [0.001, 0.005]: NEVER TRAINED to relevance |
| C (leader-only push) | 0.980 / 0.980 / 0.980 | 0.980 +/- 0.000 | 0.980 | CATASTROPHIC: H_lead 0.00 from eval 1, alpha -> 295 |
| D (both) | 0.801 / 0.980 / 0.620 | 0.800 +/- 0.147 | 0.680 | dominated by C's pathology; two seeds partially escape late |
| E (identity bias) | 0.468 / 0.486 / 0.475 | 0.476 +/- 0.007 | 0.706 | = plateau; `route_bias` ended ~ +/-0.1: also never trained to relevance |

**PASS bar (<= 0.295): not met by any arm. Per the pre-registration: the pre-fix multi-convoy
headline (exact 0.295 +/- 0.024 at `ad70a9c`) STANDS with its disclosed caveat; no further design
iteration tonight (Kilian's no-chasing bound); the decomposition below is the reportable product.**

**What the decomposition establishes (each pre-diagnosed hypothesis answered):**
1. **The follower-push-conflict hypothesis (CRITIQUE_PREFREEZE §5.1) is FALSIFIED, dramatically.**
   Removing the followers' pushes (arm C) does not relieve the leader; it destroys it: with only
   the leader's decision state in the buffer, every replay sample is the SAME state (fleet-route
   resets to an identical observation each sortie), the actor's softmax saturates onto one route
   by the first eval and never escapes (H_lead 0.00, alpha 1.1 -> 295 across 1200 sorties, all
   three seeds identically). The follower pushes were LOAD-BEARING as the only source of state
   diversity regularising the shared actor. A single-state menu policy trained by SAC is a bandit
   with a saturating softmax: a finding worth a methods paragraph.
2. **The head-term arms (B, E) are INCONCLUSIVE on the concept and conclusive on the mechanics:**
   both added parameter sets stayed 1-2 orders of magnitude below the logit scale (feat_w
   [0.001, 0.005]; bias ~ +/-0.1) because the new param groups inherited the base optimiser
   learning rates (actor 3e-4) while needing O(1) magnitudes within 1200 updates. The arms
   therefore reproduced the baseline (0.443 / 0.476 ~ 0.447) rather than testing discriminability.
   The identity-vs-features question (E vs B) remains OPEN pending properly-scaled head terms.
3. The 0.447 plateau is now reproduced a FOURTH time (gen10-MC, gen10-MC2, gen11-B, gen11-E,
   under three different target rules and two horizons): it is a hard property of the current
   architecture-plus-optimisation on honest embeddings.

**Recommended further design attempts (for Kilian's morning decision; NOT run tonight):**
1. **gen11b (cheapest, highest confidence): re-run arms B and E with the head-term param groups at
   a dedicated lr (~3e-2, i.e. 100x actor lr) and/or follow_w-style init 1.0** so the terms reach
   O(1) within the horizon; 2 arms x 3 seeds ~ 30 min. This is the direct, still-unrun test of
   both gen11 hypotheses.
2. If gen11b's E passes but B does not: the architecture exploits identity, not semantics; report
   as the honest boundary (and the multi-convoy story keeps the pre-fix number + this account).
3. If B passes: the transferable-feature head becomes the headline mechanism AND the ZST step-1
   enabler; run the 3-seed lock immediately.
4. Independent of 1-3: a LEARNED per-route embedding replacing mean-pooling (a small nn.Embedding
   at the head, lr-matched), the "identity capacity done properly" variant; and state-diversity
   for fleet-route mode (e.g. train on follower states but with the LEADER's entropy target =
   removing the conflicting-target half of §5.1 while keeping the diversity half; one flag).

## Launch/config record

SHA `2addaee` (machinery + pre-registration); arms launched 01:26-02:20 via
`scratch/gen11_orchestrator.sh`; all JSONs/logs/per-eval ckpts under `models/runs/gen11_menuhead/`.
**gen12 sweep config consequence:** with no arm improving on the baseline, the sweeps run on the
PLAIN post-fix baseline config (no gen11 flags), keeping them comparable to the standing post-fix
numbers (recorded in the gen12 ledger launch record).
