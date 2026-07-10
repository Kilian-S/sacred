# SACRED

**SACRED** (Soft Actor-Critic Robust Evolutionary Deep RL) is an MSc thesis project (Imperial College
London) framing contested convoy routing as an adversarial security game: a SAC dispatcher routes
convoys base -> FOB while a hidden, committing interdictor places ambushes, and the only defence is
anticipation + unpredictable (mixed-strategy) routing, validated against a computable minimax
equilibrium.

**Numbers policy (2026-07-10): citable numbers live ONLY in the `experiments/` ledgers.** Prose
documents (this file, HANDOVER, SYSTEM, ROADMAP, the storyline docs) carry pointers, not values;
the standing headline numbers changed twice in 48 hours during the node-ordering-fix arc and any
value quoted outside a ledger should be treated as potentially stale.

**Current standing results (pointers):**
- Single-convoy headline: `experiments/gen10_postfix.md` (gen10-SC, post-fix, supersedes B2-P3 in
  `experiments/gen08_interdiction.md`).
- Multi-convoy headline: `experiments/gen09_multiconvoy.md` best-checkpoint as EXACTLY re-evaluated
  in `experiments/gen10_postfix.md` (pre-fix, caveat disclosed there); post-fix reproduction and
  the plateau decomposition: `experiments/gen10_postfix.md` + `experiments/gen11_menuhead.md`.
- Obj-4 SBO demonstrator: `experiments/f3_sbo_demonstrator.md`.
- Disruption sweeps (K / N / held-out OD): `experiments/gen12_sweeps.md`.
- The three critiques that shaped the final arc: `CRITIQUE.md` (2026-07-02),
  `CRITIQUE_INTERDICTION.md` (2026-07-09, the node-ordering bug), `CRITIQUE_PREFREEZE.md`
  (2026-07-10, the two-headline asymmetry + ranked programme).

Start here: `HANDOVER.md` (new-agent read order at the top). Full progression:
`SACRED_PROGRESS.md` (chronicle). Historical campaign record: `CONTEXT.md`,
`experiments/gen0[1-7]*.md`.
