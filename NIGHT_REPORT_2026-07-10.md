# NIGHT REPORT (2026-07-10, ~00:30-04:15): the autonomous overnight programme

> Executed under Kilian's overnight launch authority and decision rules (gen11 pass = new
> headline; no design chasing beyond gen11; gen11 before F3). Every run pre-registered before
> launch; every result in a ledger; 8 commits, suite 155 green throughout. This file is the
> morning read; the ledgers are the citable record.

## What ran, in order

1. **gen11 (menu-head discriminability, 4 arms x 3 seeds)** - `experiments/gen11_menuhead.md`
2. **F3 SBO demonstrator (Obj-4)** - `experiments/f3_sbo_demonstrator.md`
3. **Fleet-cost column, vanilla best-checkpoint, vanilla to 3 seeds** - appended to
   `experiments/gen10_postfix.md` + gen12 batch
4. **gen12 disruption sweeps (Obj-5), incl. a held-out OD** - `experiments/gen12_sweeps.md`
5. **ZST step 0 (held-out-OD transfer, closes B2-S)** - `experiments/zst_step0.md`
6. Docs number-hygiene (README/SYSTEM: pointers only, ledgers = sole number source) + the
   featurise-cache multi-instance fix + the mission-objective closed-form vectorisation
   (28.8M convolutions -> one matmul; K=3 cells became trainable).

## The four results that matter

1. **All five objectives now have at least a demonstrated form.** Obj-4 was the last gap: the
   SBO demonstrator's neural metamodel predicts a placement/fleet design's equilibrium
   exploitability from cheap structural features with held-out Spearman **0.894** and argmin
   regret **0.0000** (450 designs, split by placement).
2. **Obj-5's "varied disruption" clause is banked with trained curves: SACRED beats the
   ALNS-certified deterministic optimum in 10/10 cells** (K in {1,2,3} x N in {2,3,5} x two
   instances), tracking the equilibrium's rise with K, with the margin over ALNS GROWING with
   fleet size (the oracle scan's prediction, now trained).
3. **THE FINDING OF THE NIGHT: the post-fix 0.447 plateau is instance-specific, not
   architectural.** On the held-out OD 35-159 (screened by the pre-registered criteria BEFORE
   training; MORE asymmetric than 62-97: leader H/lnR 0.44 vs 0.63) the plain post-fix pipeline
   reaches **best-checkpoint TAP 0.261 = 1.27x its equilibrium 0.206** (ALNS 0.699) at the
   headline cell, and 1.09-1.69x across all five of its cells. Where the INSTANCE supplies
   asymmetry (a sharp fictitious-play gradient), honest representations suffice; 62-97's flatter
   equilibrium is precisely where the old identity-hash bug had been supplying the missing
   discrimination. This dissolves the two-headline asymmetry problem (CRITIQUE_PREFREEZE §2), if
   the morning decision below confirms.
4. **gen11: no arm passed; the decomposition is the product.** (a) The follower-push hypothesis
   is FALSIFIED: leader-only pushes collapse the actor instantly (single-state replay = a
   saturating softmax bandit; H_lead 0.00 from eval 1, alpha -> 295, all seeds) - follower pushes
   are load-bearing state diversity. (b) The feature and identity head terms were silent no-ops:
   their param groups inherited the base learning rate and ended 1-2 orders of magnitude below
   the logit scale (feat_w [0.001, 0.005]; bias ~ +/-0.1) - the CONCEPT is untested, the
   mechanics are diagnosed. (c) The plateau reproduced a fifth time on 62-97 (0.483 +/- 0.041).

Also banked tonight: the **fleet-cost column** (SACRED's security premium 123.1 ~= the
equilibrium mixture's own 120.8, vs ALNS 96.1: optimal-play pricing, not RL inefficiency);
**post-fix vanilla at 3 seeds** (0.855 +/- 0.003; best-checkpoint ~0.81: selection symmetry does
not rescue the control); **ZST step 0** = the pre-registered scoping negative (transferred policy
beats shortest/uniform on the held-out game but LOSES to a random-init net, 0.699 vs 0.584: with
no observable threat map there is no transfer mechanism; this cleanly motivates map-conditioned
ZST step 1 and finally puts measured rows on the never-run B2-S instance).

## Morning decisions for Kilian (ranked recommendation)

1. **Lock the post-fix multi-convoy headline on 35-159 (recommended).** One launch: 3-seed the
   ho_N3K1 cell (~15 min at 3-parallel; pre-register "gen13-lock": bar = mean within ~0.05 of the
   single-seed 0.261, all seeds < ALNS 0.699). If it holds, the multi-convoy headline becomes a
   POST-FIX, honest-representation number on an instance screened before training, the pre-fix
   0.295 retires to the methods narrative (with the bug story as a first-class methods finding),
   and every citable number in the thesis sits on corrected code. This is the highest-value
   30 minutes available.
2. **gen11b (optional, mechanism completion):** re-run arms B and E with the head-term param
   groups at a dedicated lr (~3e-2) or follow_w-style init 1.0 (2 arms x 3 seeds, ~30 min). This
   answers identity-vs-features properly and B doubles as the ZST-step-1 mechanism test. Worth it
   if the thesis wants the "which capacity does the head need" paragraph; not needed for the
   headline if decision 1 lands.
3. **Scaling-figure restatement (no CPU):** tonight's vectorised objective matrix moved the naive
   oracle wall from K=3 to ~K=4-5 (RAM-bound: C(79,4) x 364 ~ 0.5 GB fine, K=5 ~ 70 GB not);
   `scratch/oracle_scaling_probe.py`'s numbers and the gen09 ledger's crossover section should be
   re-run/re-stated before any thesis figure uses them. The honest scaling claim remains
   amortisation + ZST, per CRITIQUE_PREFREEZE §7.
4. **ZST step 1 (the remaining aim-level promise):** now cleanly motivated by tonight's negative;
   needs the gen11b feature mechanism + an edge-vulnerability observation column + multi-OD
   training (~2-4 days). Decide against the 30 July Final Activities Report rail; my
   recommendation: attempt only if decisions 1-2 land by the weekend, else write it as designed
   future work with tonight's measured boundary as evidence.
5. **F2 (learned-antagonist co-evolution demo): still unrun**; one post-fix attempt remains
   justified (the pre-fix antagonist evidence is confounded twice over). An afternoon; optional.

## Ledger-state summary (pointers, per the numbers policy)

- Single-convoy headline: gen10-SC (`experiments/gen10_postfix.md`), supersession confirmed.
- Multi-convoy headline: pre-fix exact best-checkpoint (`gen10_postfix.md` §exact re-eval) with
  caveat; supersession candidate = gen12 ho_N3K1 (decision 1).
- Sweeps/curves: `experiments/gen12_sweeps.md`. Obj-4: `experiments/f3_sbo_demonstrator.md`.
- gen11 decomposition + morning design recommendations: `experiments/gen11_menuhead.md`.
- ZST step 0 / B2-S: `experiments/zst_step0.md`.
- Chronicle: `SACRED_PROGRESS.md` entries 17-18. Suite: **155 passed**. Tree committed throughout;
  nothing running as of 04:15.
