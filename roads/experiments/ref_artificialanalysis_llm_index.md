# ref_artificialanalysis_llm_index: external capability index for the exam roster

External reference, not an experiment. Third-party figures transcribed from the source on
the access date. Source, Artificial Analysis Intelligence Index v4.1.1, accessed
2026-08-12. The index is a composite of nine evaluations (GDPval-AA v2, tau^3-Banking,
Terminal-Bench v2.1, SciCode, Humanity's Last Exam, GPQA Diamond, CritPt, AA-Omniscience,
AA-LCR). Figures are re-scored when the index is revised, so any citation pins the
version and access date.

## Index values

| exam configuration | AA model page | index | reasoning mode | status on page |
|---|---|---|---|---|
| Llama-3.3-70B | Llama 3.3 Instruct 70B | 9 | none exists | measured |
| Qwen3.5-2B | not listed | - | - | absent from the index |
| Qwen3.5-4B | Qwen3.5 4B (Reasoning) | 20 | reasoning on | flagged estimated |
| Qwen3.5-9B | Qwen3.5 9B (Reasoning) | 22 | reasoning on | measured |
| Qwen3.5-27B | Qwen3.5 27B (Reasoning) | 35 | reasoning on | flagged estimated |
| Qwen3.6-27B (thinking on) | Qwen3.6 27B (Reasoning) | 38 | reasoning on | measured |
| Qwen3.6-27B (thinking off) | Qwen3.6 27B (Non-reasoning) | 31 | reasoning off | measured |

Reference medians on the source pages, 9 for open-weight models comparable to the Qwen
entries, 7 for Llama 3.3 70B's size class. Llama 3.3 70B release date per the source
page, 6 December 2024; the page states it is a non-reasoning model.

## Provenance

| figure | URL |
|---|---|
| Llama-3.3-70B = 9 | https://artificialanalysis.ai/models/llama-3-3-instruct-70b |
| Qwen3.5-4B = 20 | https://artificialanalysis.ai/models/qwen3-5-4b |
| Qwen3.5-9B = 22 | https://artificialanalysis.ai/models/qwen3-5-9b |
| Qwen3.5-27B = 35 | https://artificialanalysis.ai/models/qwen3-5-27b |
| Qwen3.6-27B = 38 | https://artificialanalysis.ai/models/qwen3-6-27b |
| Qwen3.6-27B non-reasoning = 31 | https://artificialanalysis.ai/models/comparisons/qwen3-6-27b-non-reasoning-vs-qwen3-6-27b |

Component scores were not captured; only the composite is recorded here. A search snippet
reported Qwen3.5-9B as 21; the model page and the pairwise comparison page both give 22,
which is the recorded value.

## Ordering cross-check against the placement exam (gen43_exam)

| model | exam share of ceiling | AA index |
|---|---|---|
| Qwen3.5-2B | 0.483 | absent |
| Qwen3.5-4B | 0.649 | 20 |
| Qwen3.5-9B | 0.719 | 22 |
| Qwen3.5-27B | 0.783 | 35 |

The two instruments rank the three shared rungs identically.

## Bibliography entry

```bibtex
@online{artificial_analysis_intelligence_index_2026,
	title = {Artificial Analysis Intelligence Index v4.1.1},
	url = {https://artificialanalysis.ai/models},
	author = {{Artificial Analysis}},
	urldate = {2026-08-12},
	date = {2026},
	note = {Composite of nine evaluations. Scores for Qwen3.5-4B and Qwen3.5-27B are
	        marked as estimated by the source.},
}
```
