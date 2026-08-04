# SACRED (aerial worktree)

**SACRED** (Soft Actor-Critic Robust Evolutionary Deep RL) is an MSc thesis project (Imperial
College London, 2026) that asks **where learning pays in contested logistics routing**, and
answers it with a measured boundary map, always scored against computable optima.

This is the AERIAL worktree (branch `gen28-aerial`). It holds the free-flight and vec-theatre
acts (gen28, gen31, gen32, gen33, gen39): the aerial dynamic trained positives on synthetic and
real Kaliningrad terrain, the concealment mechanic, and the LLM composition/curriculum arc. The
master documentation home is the roads worktree at `../sacred` (branch `gen08-interdiction`).

**Start here.** `CLAUDE.md` (identity, house rules, project map) then `HANDOVER.md` (this
branch's state) then `../sacred/HANDOVER.md` (the master state and claims register).

**Numbers policy.** Citable numbers live ONLY in the `experiments/` ledgers; prose documents
carry pointers. Historical documents live in `docs/archive/` (see its `INDEX.md`).

**Running things.** Python via the repo venv, `PYTHONPATH=. .venv/bin/python`; test suite
`PYTHONPATH=. .venv/bin/python -m pytest tests/`. Training is CPU-only on the M4 and every
launch needs Kilian's explicit go.
