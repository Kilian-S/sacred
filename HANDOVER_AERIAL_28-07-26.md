# HANDOVER AERIAL, sessions of 2026-07-26/28 (gen39 concealment: steps 1-5 COMPLETE)

> **Supersedes `HANDOVER_AERIAL_25-07-26.md` entirely.** That file described a state in which the
> step-1 screen was void and nothing had run. Everything it lists as pending has since been done.
> Read this file, then the ledger `experiments/gen39_concealment.md` top-to-bottom: it is long
> because superseded blocks are deliberately left visible with the reason they were superseded,
> and there are now nine of them.

---

## THE ONE-PARAGRAPH STATE

**gen39 is COMPLETE through step 5, including zero-shot rows on all four theatres.** The act has
two banked positives (a concealment mechanic measured with an internal control; an LLM that
composes forces beating a hand-tuned doctrine, with the terrain control passing), one measured
negative that was then FIXED (the LLM as a training curriculum: failed at step 3, repaired at
step 5 once the curriculum was made strong), and one boundary that survives everything (no
trained policy beats the best simple observing rule, anywhere, on any map). **Nothing is
running. The working tree is clean. No decision is pending.** The natural next items are
writing, or the two recorded future-work threads at the bottom.

---

## WHAT HAPPENED, IN ORDER (each step's numbers live in the ledger)

| step | what it asked | verdict |
|---|---|---|
| **1** | is there room for a learned policy, and does concealment do anything? | **PASS** - gates clear on 86% of real cells across four maps; sight is worth 1.26-1.37x against a revealing force and **exactly 1.00x** against a concealed one (the mechanic, with an internal control) |
| **2** | can an LLM compose a better enemy force than a hand-tuned doctrine? | **PASS (llama, every clause; qwen partial)** - 0.0747 vs 0.0603 against the best simple defender, and the **binding terrain control passes decisively for both models** (relabel forest/open in the brief -> compositions change, forces collapse 10-13x): the first licensed terrain-reasoning claim of the whole LLM arc |
| **3** | does training SACRED against LLM-composed enemies beat the controls? | **FAIL 0/3 seeds** - the LLM curriculum beat random composition by 29% but lost to the tuned doctrine |
| **Phase 1** | why? | the mechanism: **curriculum value tracks the enemy's IRREDUCIBLE THREAT** (its damage against a defender that already knows where it is). LLM forces were concealment gambits: 0.0007 irreducible vs the doctrine's 0.0215 |
| **Phase 1c-1e** | can the LLM be made to compose robust forces? | briefing NO, feedback NO, **grounding YES** (12-40% -> 91% once given a readable slot catalogue), but the residual gap is combinatorial **search** |
| **Phase 1f** | where can an LLM genuinely earn its place? | **sample efficiency**: in a 1.3M-force space at equal simulation budget, the LLM leads everything at 8-16 evaluations and is overtaken by hill-climbing by 96 |
| **5** | does a STRONG curriculum fix step 3? | **PASS 3/3 seeds** - llm16 0.1288 vs the tuned control 0.1677 (**23% better**, paired -0.0389 +/- 0.0031) |
| **5 zero-shot** | does it transfer to unseen theatres? | **PASS 3/3 seeds on all three unseen maps**; llm16 also leads local16 on **9/9 map-seed pairs**, an ordering Narva alone could not separate |

---

## WHAT MAY AND MAY NOT BE CLAIMED (binding wording)

**MAY:**
- *Concealment shuts the defender's information channel completely* (1.00x vs 1.26-1.37x, same
  map, same rules, only the ground changes).
- *An LLM composes enemy forces that beat a hand-tuned doctrine against a defender that must
  search* (step 2, per-model: llama passes every clause, qwen partial), *and its composition is
  terrain-grounded* (the relabel control).
- *Training against enemies authored by a directed 16-evaluation search produces a defender 23%
  better on unseen strong enemies than the tuned-doctrine control, on every seed, and this
  transfers to three unseen theatres.*
- *Language-guided proposal dominates blind search when simulations are scarce* (Phase 1f, at
  8-16 evaluations) *and is overtaken when they are plentiful* (by 96).

**MAY NOT:**
- **No "trained policy beats the simple rules" sentence.** No arm beats the best observing rule
  on any cell, on any of the four maps. This is the act's hardest boundary.
- **No "the LLM curriculum is best".** llm16 and local16 are statistically indistinguishable on
  Narva (paired -0.0066 +/- 0.0265); only the SIGN is consistent out of distribution.
- **No "LLMs" in general.** Everything is per-model, and the models REVERSE between tasks
  (llama leads at composition, qwen at the grounded slot task).
- **Nothing may be mixed across game versions** (symmetric/asymmetric forest, leaked/masked
  concentration, grid/quota sampling, spot-at-site/spot-where-it-fires). Standing rule 8.

---

## THE SEVEN FAULTS AND THREE CORRECTIONS THIS ARC PAID FOR

The 25-07 handover lists seven faults found before step 1 could run. Since then, four more
corrections were made and are disclosed in the ledger rather than quietly fixed:

1. **The eval defect** (151k network calls per checkpoint) - the trainer's policy evaluation
   looped per state; it collapses to one call in closed form (route features are a pure logit
   shift). Pinned by `tests/test_gen39_trainer_eval.py`.
2. **Duplicate graphs in replay** - each stored transition memoised its own copy of the field
   graph, ~1 GB/run, which pinned the machine at the memory-compression threshold and was the
   true cause of every "system time" crawl (mis-diagnosed twice first). Shared per-field graph;
   tensor-exactness tested.
3. **A single-field B3 test** reported a PASS that was a field-selection artefact; the
   all-six-field rerun fails. Disclosed in place.
4. **A pre-declared prediction that failed:** kgd was pre-declared a negative cell for the LLM
   arm (the free gate had random search authoring better kgd forces); it turned out to be the
   arm's strongest map. Recorded as made-and-wrong.

**Process lesson worth inheriting:** `pkill -f <pattern>` matches the shell issuing it, so
"batch stopped" was reported three times while runs were still alive. Kill by explicit PID with a
self-excluding pattern and verify over 30 seconds.

---

## STATE OF THE CODE

**New this arc:** `scripts/train_gen39_conceal.py` (the concealment trainer: state = track window
x set-of-teams-seen, reveal channel as a head column, `--blind` causal control, step-5 arms),
`src/envs/aerial_conceal.py` (+ per-team doctrines, `choose_force`, spot-where-it-fires),
`src/envs/aerial_theatre_vec.py` (terrain v2, quota sampler), `src/redforce.py`
(`force_schema(terrain)`).

**Probes, in the order they were built:** `gen39_screen2.py` (the step-1 screen),
`gen39_compose.py` (step 2, three composers + the relabel control), `gen39_step5_prep.py` (the
four 16-eval curricula), `gen39_step5_score.py`, `gen39_zeroshot.py` (the map rows, with a
self-check that reproduces step 5 to 0.00000), `gen39_phase1{_confound,b_score,c,d,e,f}.py`
(the diagnosis chain), `gen39_freegate.py`.

**Tests:** `tests/test_gen39_terrain.py` (20 cases) + `tests/test_gen39_trainer_eval.py` (2).

**Artefacts:** `models/runs/gen39_step5/` (12 runs + curricula), `gen39_zeroshot.json`,
`gen39_phase1*.json`, `assets/gen39_sites_*.png`.

---

## RUNNING THINGS ON THIS MACHINE (measured, not guessed)

- The trainer costs **1.83 s/flight solo**, ~3.3 s at 4-way. 5000 flights x 4 concurrent = ~4.5 h.
- **~3.1 GB per run.** Four concurrent is the safe shape on 24 GB; twelve thrashes into 16 GB of
  swap and everything crawls.
- **Do not `nice` training runs** (measured 3x penalty from efficiency-core placement). Cap all
  maths thread pools to 1 and set `OMP_WAIT_POLICY=PASSIVE`.
- Oracle probes: use a 9-10 worker pool; the whole four-map zero-shot took 9 minutes.

---

## RECORDED FUTURE WORK (nothing scheduled)

1. **The hybrid the evidence points at:** LLM proposes the neighbourhood, local search refines
   inside it. Both halves are measured (the LLM leads at 8-16 evaluations, hill-climbing wins by
   96); the combination is untested.
2. **The validation-set mismatch:** step 5's validation cache is inherited from step 3 and is
   built from TUNED-family enemies, so the strong arms may be losing their best checkpoints to a
   mismatched family. A rebuild is cheap and would tighten every step-5 number.
3. The LLM-doctrine arm is **excluded on evidence** (free gate: LLM-written doctrine is 0.53-0.75x
   the tuned recipe on identical positions, all four maps). Do not reopen without new evidence.
