# SACRED

**SACRED** (Soft Actor-Critic Robust Evolutionary Deep RL) is an MSc thesis project (Imperial
College London, 2026) that asks **where learning pays in contested logistics routing**, and
answers it with a measured boundary map. A SAC dispatcher routes convoys against hidden,
committing interdictors on real city graphs (and against air-defence fields on real terrain in
the aerial branch), always scored against computable optima. Below a measured boundary a
two-line heuristic is near-optimal and the thesis proves it; learning pays where the
interdiction budget approaches the min-cut and against adaptive pattern-of-life adversaries;
above both sits a measured conditioning-capacity wall. An LLM arc maps where language models
help the pipeline and where they do not.

This is the ROADS worktree (branch `gen08-interdiction`) and the master documentation home. The
sibling worktrees are `../sacred-aerial` (branch `gen28-aerial`) and `../sacred-gen29` (branch
`gen29-multiod`). The shareable restructured repo with the Mission Control web app is
`../imperial-sacred`.

**Start here.** `CLAUDE.md` (identity, house rules, project map) then `HANDOVER.md` (the
current state and the claims register) then `SYSTEM.md` (operating dogmas).
`SACRED_PROGRESS.md` is the 34-entry chronicle.

**Numbers policy.** Citable numbers live ONLY in the `experiments/` ledgers; prose documents
carry pointers. Historical direction documents and the critique series live in `docs/archive/`
(one-line descriptions in its `INDEX.md`).

**Running things.** Python via the repo venv, `PYTHONPATH=. .venv/bin/python`; test suite
`PYTHONPATH=. .venv/bin/python -m pytest tests/`. Training is CPU-only on the M4 and every
launch needs Kilian's explicit go.
