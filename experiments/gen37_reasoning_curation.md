# Generation: gen37_reasoning_curation (Phase-2 pivot, Kilian 2026-07-24: SACRED enhanced by LLM reasoning - action-space curation on the coordination game)

**status: PRE-REGISTERED 2026-07-24.** Mandate: Kilian's 2026-07-24 direction (recorded
in-conversation): replace the staged gen33 metric-2 curriculum act with a REASONING-ASSISTED
algorithm act: *"SACRED enhanced via LLM reasoning"*, application target = route-space
curation ("if a reasoning LLM effectively cuts down the routes to a couple worth considering
and does it well, the value would be immense"). Full launch control granted (2026-07-23,
reaffirmed "once you have built you can automatically launch"). Ledger committed BEFORE the
harness runs; results appended below the RESULTS line; nothing above it changes.

**git SHA:** the commit landing this ledger; steps pin their own SHAs.

## Why here

The gen29 coordination game is the programme's only surviving oracle moat (median 31% vs the
fitted cap), and this week separated its trained failure into a CAPACITY wall: the policy
class cannot represent/fit the correlated optimum over the ~1000-1331 joint route-triples
(gen36: distillation with exact labels fails even in-sample), yet the true coordinated
optimum is SPARSE (2-11 triples). If an LLM's structural reasoning can curate the joint space
down to ~50 candidate triples containing the value, the SAME SACRED agent retrained inside
the curated space no longer needs the representation it provably lacks. That is the exact
sense of "SACRED enhanced by LLM reasoning": one reasoned component swapped, everything else
identical, banked failures as the before, exact solvers as the after.

Evidence-arc fit: B2 = the models know concepts but cannot produce numbers; gen33 = their
force composition is not terrain-grounded; gen37 asks the remaining question - can their
REASONING contribute the structure while algorithms do all arithmetic?

## The act (pinned)

**Game/instances:** the gen29 screen verbatim (`models/runs/gen29_screen.json`): 16 train +
4 val + 6 held-out cells, K=1, F=3 streams, mission P(>=1 lost); trainer
`scripts/train_multiod_generalist.py` config verbatim (14000 sorties, fp-tau 0.05,
smooth-window 250, alpha-floor 0.20, 3 seeds).

**The one change - prefix-conditional action masks from a per-instance shortlist S of M=50
route-triples:** stream 0 may pick r0 present in S; stream 1 given r0 may pick r1 with
(r0,r1,*) in S; stream 2 completes a triple in S. Implemented flag-gated
(`--shortlist FILE`); flag-off path byte-identical; suite green with raw output at the build
record.

**Arms (identical trainer/budget/seeds; only S differs):**
- `llm`: S from llama-3.3-70b (pinned; the stronger gen33 model), temperature 0.2, ONE call
  per instance, guided retry <= 3 on invalid JSON (the gen33 contract pattern). Prompt =
  structured route summaries per stream (travel cost, worst edge vulnerability) + the
  cross-stream shared-segment matrix + the mission rules; task = select the 50 triples worth
  including in a coordination playbook, JSON out. NO payoff/equilibrium values are shown; the
  model sees the same instance data the deployed policy would. Full transcripts committed.
- `random`: uniform 50 triples without replacement, seed 9000+instance-index.
- `heuristic`: the strongest afternoon rule - rank triples by (max pairwise shared-edge count
  ascending, then summed worst-vulnerability ascending), take 50 (corridor-disjointness first,
  safety second).

**Mechanism rows (oracle-exact, computed BEFORE training):** per instance and arm:
(a) CONTAINMENT = the fraction of the true optimal mixture's mass (dstar, recomputed via
`_row_minimiser(env.obj_matrix)`) inside S; (b) LP-OVER-SHORTLIST = the exact game value
restricted to S (tiny LP) - the ceiling any policy inside S can reach, beside the full
equilibrium and the fitted cap. These decompose the outcome: did the REASONING find the
value; did the TRAINING then capture it.

## Decision metric (PRE-REGISTERED; inheriting gen29's tiers so before/after is exact)

Held-out zero-shot (the 6 cells, each with ITS OWN arm-generated shortlist; select-on-train
as gen29), ratio-to-eq per cell.

> **TIER-1 (the enhancement claim): the llm arm's pooled held-out ratio-to-eq < 1.44 (the
> best-independent-product row gen29's trained half never reached) on >= 2/3 seeds.**
> **TIER-2 (the moat claim): llm arm beats the fitted cap (ratio-to-cap < 1.0) on >= 4/6
> held-out cells on >= 2/3 seeds.**
> **COMPARATIVE CLAUSE (the reasoning must be the ingredient): llm arm pooled < random arm
> pooled AND < heuristic arm pooled (each at matched budget/seeds); an llm win with random ~
> equal is "any curation suffices", reported as such.**
> **MECHANISM CLAUSES (reported, gate Tier interpretation): llm containment and
> LP-over-shortlist beside both control arms; if llm LP-over-shortlist ~ eq but training
> lands far above it, the residual is a TRAINING gap, not a reasoning gap (and conversely).**
> **Branches (all writable):** Tier-1+2 + comparative = the headline: *the identical RL agent
> captures the coordination moat it provably could not capture alone, when LLM structural
> reasoning curates its action space; random/heuristic curation does not suffice.* Tier-1
> without comparative = curation-not-reasoning (still repairs the gen29 deployable, honestly
> attributed). All-arms-fail = the capacity wall is not bypassable by pruning at this scale
> (the sharpest possible statement that the failure is not about action-space size); the
> mechanism rows say which stage died. Every branch cites the gen29/gen36 banked numbers as
> the before.

## Application register (recorded, contingent)

The aerial "thousands of routes" showcase (Kilian's application framing) is pre-registered as
gen37-B, BUILT ONLY IF Tier-1 + comparative land here: same mechanism on a large-menu
vec-theatre cell, screened first by free oracle probes. Not part of this act's bars.

## Design decisions ledgered

1. M=50 (~20-25x prune): large enough that containment is not gifted, small enough to bypass
   the measured capacity wall; fixed before any generation.
2. One LLM call per instance at temp 0.2: the cheapest defensible protocol; no
   prompt-iteration after the first committed transcript (the anti-cherry-pick rule).
3. Held-out instances get LLM shortlists too: that IS the deployment pipeline (curation at
   deployment); no payoff information flows, disclosed plainly.
4. Tabular concession pre-written: within a 50-atom shortlist a tabular learner could also
   mix; the act's claim is therefore the ZERO-SHOT GENERALIST register (gen29's), never
   single-instance superiority.
5. The blinded-channel control is NOT rerun (the causal question here is the shortlist;
   random/heuristic arms carry it); gen29's blinded result stands as context.
6. Numbers live only in this ledger + its JSONs; transcripts in
   `models/runs/gen37_reasoning_curation/transcripts/` (committed).
7. Thread caps per SYSTEM.md; 9 training runs staged 6-then-3 to bound oversubscription.

## Commands (pinned)

```bash
# 1. shortlists + mechanism rows (LLM via the tunnel; oracle LPs local):
OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. \
  /Users/kilian/Kilian/ICL/Thesis/code/sacred/.venv/bin/python scratch/gen37_shortlist.py
# 2. training, per arm in llm random heuristic, per seed 0 1 2:
... scripts/train_multiod_generalist.py --shortlist models/runs/gen37_reasoning_curation/shortlists_${arm}.json \
  --sorties 14000 --eval-every 1000 --seed $S --threads 2 \
  --json-out models/runs/gen37_reasoning_curation/${arm}_s${S}.json \
  --ckpt-dir models/runs/gen37_reasoning_curation/${arm}_s${S}_ckpts
```

## Compute envelope

Shortlists: 26 LLM calls (~15-60 s each) + 3x26 tiny LPs = well under an hour. Training: 9
runs x 14000 sorties, staged 6-then-3 at 2 threads each; estimate one long night into
morning. Hard ceiling: 36 h wall; no extension without a dated amendment before results.

## RESULTS (appended per step; nothing above changes after launch)
