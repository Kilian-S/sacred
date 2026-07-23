# Generation: gen36_multiod_rescue (Phase-1 frontier hardening, point 3: executing gen29's two pre-committed untaken options - locating the wall, dynamics vs capacity)

**status: PRE-REGISTERED 2026-07-23.** Mandate: Kilian's 2026-07-23 Phase-1 direction (point 3:
reopen gen29 carefully). This act EXECUTES, without reopening, the two options the gen29 ledger
pre-committed and left pending on Kilian's decision: the DISTILLATION control and the single
permissible dense-credit self-play RE-AIM ("one re-aim maximum, then close",
`GEN29_MULTIOD_HANDOFF.md`). Framed strictly as LOCATING THE WALL (self-play dynamics vs policy
capacity), never as "SACRED works". Ledger committed before any new code; results appended
below the RESULTS line; nothing above it changes. Both launches require Kilian's explicit go.

**git SHA:** the commit landing this ledger; steps pin their own SHAs.

## Why

gen29's oracle half is the programme's only gap that survives a complete hostile baseline
family (median 31% vs the oracle-fitted cap over 55 cells); its trained half failed both tiers
with a measured mechanism: terminal-only reward over the 3-stream chain at ~375
sorties/instance (density starvation) + FP instability, blinded ~ sighted. The two pre-committed
options separate the remaining hypotheses:

- **Step A (distillation)** tests CAPACITY: can the policy class express and transfer the
  coordinated solution at all, when handed exact labels and spared self-play dynamics?
- **Step B (the one re-aim)** tests DYNAMICS: does exact per-stream dense credit let self-play
  find what terminal-only credit could not?

Label feasibility is verified (`scratch/gen36_label_probe.py`, 2026-07-23,
`models/runs/gen36_label_probe.json`): anchors reproduce EXACTLY (eq/cap dev 0.0 on 6/6 cells
incl. the headline 0.2046/0.3167), the joint optimal mixture extracts in ~0.05 s/cell (full
26-cell label pass ~1 s), supports are SPARSE (2-11 route-triples of ~1000+) and factorise
cleanly into the sequential per-stream conditionals the policy head already parameterises.
Sparse correlated supports are exactly what independent play cannot express - and an easy
supervised target.

## Step A: distillation control (pinned)

- **Labels:** per train instance, dstar = `_row_minimiser(env.obj_matrix)[1]`, factorised to
  P(r0), P(r1|r0), P(r2|r0,r1) in the trainer's own joint index order.
- **Trainer:** NEW script `scripts/distill_multiod.py` (additive; nothing existing modified):
  same policy class/architecture and obs as `train_multiod_generalist.py` (sighted), loss =
  cross-entropy of the factorised conditionals under dstar, train pool = gen29's 16 train
  cells, VALIDATION = gen29's 4 val cells with early stopping and select-on-val (the gen24
  discipline, pre-registered here so the gen24 overfit trap cannot recur), zero-shot eval on
  the 6 held-out cells via `exploitability_of_joint_dist` (exact).
- **Bars (inheriting gen29's tiers verbatim):**

> **A-TIER-1: pooled held-out ratio-to-eq < 1.44** (the best-independent-product row: the
> coordination channel carries something). **A-TIER-2: ratio-to-cap < 1.0 on >= 4/6 held-out
> cells.** Reported ungated: per-cell rows, val-curve, the untrained-policy anchor (1.90).

## Step B: the pre-committed re-aim (pinned; launched ONLY if Step A passes A-Tier-1)

- **Dense per-stream credit, exact telescoping decomposition** (no approximation, objective
  unchanged): with s_g = survival of stream g's routed choice under the drawn interdiction j,
  define Phi_f = -interception_loss x (1 - prod_{g<=f} s_g) and give stream f the immediate
  reward r_f = Phi_f - Phi_{f-1}. Sum over the 3 streams = the original terminal mission
  reward exactly; each stream is charged its marginal contribution to fleet failure given the
  already-committed prefix. Implemented flag-gated (`--dense-credit`) in
  `train_multiod_generalist.py`; flag-off path byte-identical, suite green with raw output at
  the build record.
- **Everything else = the gen29 failed run's config verbatim** (16-instance pool, fp-tau 0.05,
  smooth-window 250, alpha-floor 0.20, 3 seeds + blinded control seed 0), at the FULL 14000
  budget the re-aim wording reserved. The known density limitation (14000/16 = 875
  sorties/instance vs ~7000 single-instance need) is disclosed here, before launch: Step B
  tests credit assignment AT the pre-committed scale, not a density fantasy.
- **Bars (gen29's tiers verbatim):**

> **B-TIER-1: pooled held-out ratio-to-eq < 1.44 on >= 2/3 seeds. B-TIER-2: ratio-to-cap < 1.0
> on >= 4/6 held-out cells, >= 2/3 seeds. COORDINATION CLAUSE (causal): the sighted arm must
> beat the blinded arm; blinded ~ sighted voids any coordination claim regardless of tiers.**

## The wall-location matrix (pre-written; the act's finding whatever happens)

| A (capacity) | B (dynamics) | conclusion (bankable sentence) |
|---|---|---|
| PASS | PASS | the gen29 moat is capturable: capacity was there, terminal-only credit was the binding failure; self-play with exact dense credit collects the correlation gap |
| PASS | FAIL | the wall is SELF-PLAY DYNAMICS, not capacity: supervision reaches what adversarial training cannot at this density - the gen24 lesson at the coordination tier |
| FAIL | not launched | the wall is CAPACITY: the sequential policy class cannot express/transfer the sparse correlated optimum; gen29 closes as the final boundary cell, now with the mechanism separated |

Any cell of this matrix is a writable result; none of them reopens gen29's banked verdicts.

## Design decisions ledgered

1. A before B, and B gated on A-Tier-1: if capacity is absent, the one permissible re-aim is
   not spent on a foregone conclusion (the anti-chase dogma applied to the attempt budget).
2. The dense credit is an EXACT decomposition, not shaping-with-bias: pre-committed formula
   above; no reward-tuning lever exists mid-run.
3. Blinded control retained in B: coordination claims need the causal clause, exactly as
   gen29's design demanded.
4. Distillation is labelled honestly as supervised: a PASS licenses "the moat is capturable /
   where the wall is", never "adversarial training works".
5. Select-on-val everywhere in A (4 val cells exist precisely for this); select-on-train
   discipline in B as gen29.
6. Numbers live only in this ledger + its JSONs; anchors pinned from the committed probe.
7. Thread caps per SYSTEM.md on every multi-process launch.

## Commands (pinned; launch = Kilian's explicit go)

```bash
# Step A (cheap, CPU-minutes):
OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. \
  /Users/kilian/Kilian/ICL/Thesis/code/sacred/.venv/bin/python scripts/distill_multiod.py \
  --screen models/runs/gen29_screen.json --seed $S \
  --json-out models/runs/gen36_multiod_rescue/distill_seed$S.json
# Step B (only if A-Tier-1 passes), per seed 0 1 2 + blind:
OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. \
  /Users/kilian/Kilian/ICL/Thesis/code/sacred/.venv/bin/python scripts/train_multiod_generalist.py \
  --dense-credit --sorties 14000 --eval-every 1000 --seed $S --threads 2 \
  --screen models/runs/gen29_screen.json \
  --json-out models/runs/gen36_multiod_rescue/seed$S.json \
  --ckpt-dir models/runs/gen36_multiod_rescue/seed${S}_ckpts
# blind control: add --blind, seed 0
```

## Compute envelope

Step A: ~1 s labels + supervised training in CPU-minutes per seed (3 seeds; well under an
hour total including evals). Step B: the gen29 batch scale at 14000/seed, 3 seeds + blind,
2-parallel - plan two nights, hard ceiling three; no budget extension without a dated amendment
BEFORE results are read.

## RESULTS (appended per step; nothing above changes after launch)

### RESULT, STEP A (2026-07-23 21:51 BST; batch `scratch/gen36a_batch.sh`, SHA `3dd71c5`;
### artefacts `models/runs/gen36_multiod_rescue/distill_seed{0,1,2}.{json,log}`; ~7 min compute)

| seed | pooled held-out ratio-to-eq (A-TIER-1 bar < 1.44) | beats-cap (A-TIER-2 bar >= 4/6) | early stop |
|---|---|---|---|
| 0 | 1.796 | 2/6 | epoch 35 (val 2.24) |
| 1 | 2.453 | 1/6 | epoch 100 (val 1.96) |
| 2 | 2.492 | 1/6 | epoch 105 (val 1.87) |

**A-TIER-1 FAILS 0/3 seeds; A-TIER-2 FAILS 0/3.** The mechanism is unambiguous in the loss
curves: the cross-entropy to the exact coordinated labels plateaus FAR above the sparse-label
entropy floor (seed 0: 101 -> 87 over 85 epochs, still creeping, while val transfer never
improves) - the class cannot FIT the correlated targets even in-sample. This is the pure
CAPACITY failure mode, and it coheres exactly with gen29's blinded~sighted control: the
prefix-conditioning channel (overlap head column + taken_node_frac) is too weak to carry the
sharp conditionals the sparse correlated optimum requires, under supervision just as under
self-play.

> **VERDICT (the pre-written matrix, row 3): A FAIL -> STEP B IS NOT LAUNCHED (the
> pre-registered gate; the one permissible re-aim is not spent on a foregone conclusion).
> The wall is CAPACITY, not self-play dynamics.** gen29 closes as the final boundary cell
> with its mechanism now SEPARATED: the coordination moat (median 31% vs the cap, the
> programme's only surviving oracle gap) is unreachable by this policy class regardless of
> training signal - exact labels, validation early stopping and zero adversarial dynamics do
> not help. Bankable sentence: *the binding constraint on learned three-stream coordination at
> thesis scale is the policy class's conditioning capacity, not the training method.*
> Future-work register (recorded, NOT launched; the attempt budget is closed): a
> prefix-identity conditioning channel at the head (the gen11 lever-2 pattern applied to
> committed routes) is the constructive candidate an examiner would ask about.
