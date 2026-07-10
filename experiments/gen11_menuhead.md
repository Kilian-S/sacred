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

## RESULT (to be appended; nothing above this line changes after launch)
