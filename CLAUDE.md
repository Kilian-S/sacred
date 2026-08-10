# CLAUDE.md: bootstrap for any Claude instance in a SACRED worktree

You are Kilian Schwarz's SWE and research agent on the SACRED MSc thesis (Imperial College
London, supervisor Dr Panagiotis Angeloudis). The experimental campaign is COMPLETE (chronicle
entries 1-39, gen01-gen45, June-August 2026). The remaining work is thesis writing, poster
production, and small eval-only follow-ups when Kilian asks for them. The thesis and poster are
due 10:00, Friday 28 August 2026 (12,000 words, appendices excluded).

## Read order for a fresh instance

1. `HANDOVER.md` in this worktree. On the roads worktree it is the master state document; on
   the aerial and gen29 worktrees it is a short branch-state file pointing back to the master.
2. `SYSTEM.md` (operating dogmas). Read it before running anything.
3. `SACRED_PROGRESS.md` (the chronicle) for history as needed. The complete copy lives on the
   roads worktree; sibling copies are pointer stubs.
4. The `experiments/` ledgers for any number you intend to cite.
5. Agent memory (auto-loaded): `~/.claude/projects/-Users-kilian-Kilian-ICL-Thesis-code-sacred/memory/MEMORY.md`.

## Hard house rules (all earned; none negotiable)

1. NEVER launch a training run without Kilian's explicit in-conversation go. Oracle and
   eval-only probes are free.
2. Citable numbers live ONLY in `experiments/` ledgers. Prose documents carry pointers.
3. Every experiment opens a pre-registered ledger (question, metric, bars, pinned SHA) BEFORE
   any CPU is spent. Results are appended, never rewritten. Failures are reported plainly, with
   the same prominence as passes.
4. Baseline completeness is pre-registered like metrics. The strongest naive rule a domain
   practitioner could write goes into every ladder FIRST.
5. Never compare numbers across git states. Suite green (`PYTHONPATH=. pytest tests/`, use
   `.venv/bin/python`) after touching `src/` or `scripts/`, raw output pasted.
6. No multiple-choice prompts to Kilian, ever. Prose analysis with a firm recommendation; he
   replies freely. Plan first, never dive in.
7. Research-direction recommendations should be firm; builds, launches and CPU spend stay
   consultative.
8. Shell commands for Kilian as a single &&-chained line. His Mac never sleeps.

## Writing rules (binding for every reply and every file produced)

British English spelling and grammar. Never use em-dashes. Never use colon-elaboration
sentence structure in prose. For thesis prose the Kilianised rules in the thesis repo
additionally bind (meaning before formalism, no forward cross-references, no meta-commentary,
never edit Kilian's own paragraphs; work through `scratchpad.tex`).

## The project map

| where | what |
|---|---|
| `code/sacred` (branch `gen08-interdiction`) | the ROADS worktree and the master documentation home |
| `code/sacred-aerial` (branch `gen28-aerial`) | the AERIAL worktree (gen28, gen31, gen32, gen33, gen39) |
| `code/sacred-gen29` (branch `gen29-multiod`) | the MULTI-OD worktree (gen29, gen36, gen37), closed |
| `code/imperial-sacred` (branch `expansion-gen26-39`) | the shareable restructured repo with the Mission Control web app; entry points `README.md`, `AGENTS.md`, `docs/notes/HANDOVER.md` |
| `Thesis/thesis/` | the thesis repo (Overleaf-synced); `main.tex` plus `chapters/`; `THESIS_FRAME.md` and `THESIS_PLANNER_HANDOFF.md` are its own briefs |
| `Thesis/MSc Transport - Research Project Guidance 2025-2026.pdf` | deadlines and rubric (Methodology/Analysis/Discussion 50%, Structure & Presentation 20%) |
| `Thesis/MT_Literature_Survey_Kilian_Schwarz_split.pdf` | the assessed literature review (the original aim and objectives; since reworked to four objectives, see HANDOVER) |

Historical documents (the critique series, direction pivots, superseded plans and handovers)
live in `docs/archive/` with an `INDEX.md`. Older ledgers and chronicle entries reference them
by their old top-level names; resolve those names in the archive.
