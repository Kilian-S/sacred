# Reference: ref_artificialanalysis_llm_index (external LLM capability index for the Act 5 exam roster)

- **status: EXTERNAL REFERENCE. NOT AN EXPERIMENT. Nothing was run, trained, or computed for
  this file. Every number below is a third-party figure transcribed from a public web page on
  the access date, and its only role in the thesis is as a cited external comparator for the
  Act 5 placement-exam roster.**
- **captured: 2026-08-12 by the thesis agent, at Kilian's instruction, for citation in
  \cref{sec:res:act5}.**
- **source: Artificial Analysis, Artificial Analysis Intelligence Index v4.1.1.**
- **volatility warning: these are live web figures that are re-scored when the index is
  revised. Any citation MUST pin both the index version (v4.1.1) and the access date
  (2026-08-12). Do not refresh the numbers without also refreshing the version string.**

## Why this file exists

The Act 5 placement exam scores seven model configurations on our own instrument (share of
ceiling on forty emplacement items). That instrument is ours alone, so a reader has no way to
tell whether the capability ordering it produces is a property of the models or a property of
our exam. This index is the external cross-check: an independent, vendor-neutral composite
that happens to cover five of the seven exam rows on one scale.

It is a **comparator, not evidence**. No thesis claim rests on it. It supports one sentence of
external validity and nothing more.

## What the index is

Artificial Analysis Intelligence Index **v4.1.1**, a composite of nine evaluations:

    GDPval-AA v2, tau^3-Banking, Terminal-Bench v2.1, SciCode, Humanity's Last Exam,
    GPQA Diamond, CritPt, AA-Omniscience, AA-LCR

It is a single scalar over nine heterogeneous tests, several of them agentic or coding rather
than reasoning in the narrow sense. It is therefore NOT a reasoning benchmark, and must not be
described as one. The honest description is "a composite capability index".

## The table (verbatim, 2026-08-12, index v4.1.1)

| Exam row (ours)    | AA model page          | Index | Reasoning mode | Measurement status                      |
|--------------------|------------------------|-------|----------------|-----------------------------------------|
| Llama-3.3-70B      | Llama 3.3 Instruct 70B | 9     | none exists    | not flagged estimated                   |
| Qwen3.5-2B         | *not listed*           | -     | -              | absent from the index                   |
| Qwen3.5-4B         | Qwen3.5 4B (Reasoning) | 20    | reasoning ON   | **flagged "(estimated)"**               |
| Qwen3.5-9B         | Qwen3.5 9B (Reasoning) | 22    | reasoning ON   | not flagged estimated                   |
| Qwen3.5-27B        | Qwen3.5 27B (Reasoning)| 35    | reasoning ON   | **flagged "(estimated)", independent evaluation forthcoming** |
| Qwen3.6-27B (think ON) | Qwen3.6 27B (Reasoning) | 38 | reasoning ON   | not flagged estimated                   |
| Qwen3.6-27B (think OFF) | Qwen3.6 27B (Non-reasoning) | **31** | reasoning OFF | not flagged estimated                   |

Reference points given on the pages: median 9 for comparable open-weight models (against
Qwen3.5-4B, Qwen3.5-27B and Qwen3.6-27B), median 7 for Llama 3.3 70B's size class.

Llama 3.3 70B release date, per the source page: **6 December 2024**.

### Per-row provenance

| Figure | URL |
|---|---|
| Llama-3.3-70B = 9 | https://artificialanalysis.ai/models/llama-3-3-instruct-70b |
| Qwen3.5-4B = 20 | https://artificialanalysis.ai/models/qwen3-5-4b |
| Qwen3.5-9B = 22 | https://artificialanalysis.ai/models/qwen3-5-9b |
| Qwen3.5-27B = 35 | https://artificialanalysis.ai/models/qwen3-5-27b |
| Qwen3.6-27B = 38 | https://artificialanalysis.ai/models/qwen3-6-27b |
| Qwen3.6-27B non-reasoning = 31 | https://artificialanalysis.ai/models/comparisons/qwen3-6-27b-non-reasoning-vs-qwen3-6-27b |
| 38 vs 22 side by side | https://artificialanalysis.ai/models/comparisons/qwen3-6-27b-vs-qwen3-5-9b |

Every figure above was read from the model's own page, not from a search summary. One
transcription conflict was seen and resolved: a search snippet reported Qwen3.5-9B as 21,
while the model page and the pairwise comparison page both give 22. **22 is banked**; the 21
is recorded here only so the discrepancy is not rediscovered later.

Individual per-benchmark scores (GPQA Diamond, HLE and so on) were NOT captured. Only the
composite is banked. Do not quote a component score from this file.

## Cross-check against our own exam

Our placement exam, share of ceiling (scratchpad Act 5 table):

| Model | Ours (share of ceiling) | AA index |
|---|---|---|
| Qwen3.5-2B  | 0.483 | absent |
| Qwen3.5-4B  | 0.649 | 20 |
| Qwen3.5-9B  | 0.719 | 22 |
| Qwen3.5-27B | 0.783 | 35 |

**Finding (agreement of ordering).** The two instruments rank the three shared rungs
identically, and agree on the shape: a small step from 4B to 9B (0.649 -> 0.719; 20 -> 22)
and a large one to 27B (-> 0.783; -> 35). This is consistent with, and independent of, our own
result that cumulative size contrasts separate with intervals excluding zero while no single
adjacent step does.

**Finding (vintage asymmetry priced).** A 27B model of 2026 scores 38 against 9 for a 70B
model of December 2024. This supports the frame's existing vintage-asymmetry wording and our
decision to hold the Llama row outside the family statistics as a reference line.

**Finding (the Llama row is a different class of object).** The source page states explicitly
that Llama 3.3 70B is a non-reasoning model: "No reasoning. This page shows the non-reasoning
version of this model." Its 9 is therefore a non-reasoning score by necessity, not by
configuration. This is a fact about the model, not a choice of ours, and it is the cleanest
available justification for the reference-line treatment.

## Wording rules for any sentence that cites this file

1. Call it a **composite capability index**, never a reasoning benchmark.
2. Cite the **version and access date** in the same sentence or in the note: v4.1.1,
   accessed 12 August 2026.
3. The Qwen3.5-4B and Qwen3.5-27B figures are **flagged estimated on the source page** and
   must be reported as estimates wherever they appear. The 27B page adds "independent
   evaluation forthcoming".
4. **The index DOES carry both modes for Qwen3.6-27B: 31 off, 38 on** (corrected 2026-08-13;
   an earlier version of this file wrongly recorded the off entry as absent). It therefore
   shows a substantial external deliberation effect, roughly a 23 per cent relative gain and
   the same size as the 9B-to-27B step within Qwen3.5.

   This does NOT relax binding rule (x). The index measures deliberation on graduate reasoning
   and coding; our instruments measure it on emplacement choice and force composition, where
   it is invisible twice (the placement exam, within seed re-roll spread; gen39 step-2
   composition, off-on gap smaller than the same-config draw-to-draw gap). The licensed
   sentence is the CONTRAST, not agreement with either side:

   > Deliberation is worth seven index points on a general capability composite, and nothing
   > our instruments can resolve on this task.

   No sentence may use the index to claim our null is wrong, and none may use our null to
   claim the index effect is absent. Different tasks, different verdicts, both reported.
5. Per-model only, never pooled across the family, per binding rule of the frame.
6. Qwen3.5-2B has no external counterpart. Any ladder sentence citing this index must say so
   rather than quietly starting the ladder at 4B.
7. This index is a comparator for **capability ordering** only. It says nothing about our
   task, our items, or curriculum authorship, and no causal sentence may lean on it.

## Bibliography entry (biblatex, matches references.bib style)

```bibtex
@online{artificial_analysis_intelligence_index_2026,
	title = {Artificial Analysis Intelligence Index v4.1.1},
	url = {https://artificialanalysis.ai/models},
	author = {{Artificial Analysis}},
	urldate = {2026-08-12},
	date = {2026},
	note = {Composite of nine evaluations: GDPval-AA v2, tau\textasciicircum 3-Banking,
	        Terminal-Bench v2.1, SciCode, Humanity's Last Exam, GPQA Diamond, CritPt,
	        AA-Omniscience, AA-LCR. Scores for Qwen3.5-4B and Qwen3.5-27B are marked as
	        estimated by the source.},
}
```

## Known gaps, stated so they are not rediscovered

- Qwen3.5-2B: absent from the index.
- Qwen3.6-27B with deliberation off IS listed (31). Earlier versions of this file said
  otherwise; that error was corrected on 2026-08-13.
- Component scores: not captured, only the composite.
- Two of the five figures are source-flagged estimates.
- "Not flagged estimated" for the remaining three means no such flag was observed on the page
  on the access date. That is weaker evidence than an explicit measured marker, and should not
  be reported as "independently measured".
