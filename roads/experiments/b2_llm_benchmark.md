# B2: agentic-LLM exploitability benchmark
Registered 2026-07-12 (protocol and registers), 2026-08-12 (71-33 cell), 2026-08-13 (qwen thinking-mode rerun).
Results 2026-07-17 (35-159, gdansk 249-95), 2026-08-12 (71-33), 2026-08-13 (thinking-mode rerun).
Code: harness at SHA `83781ff`; 71-33 registration SHA `f129694`.
Artefacts: `models/runs/b2_llm/batch_35159/` + `batch_35159_scored.json`, `models/runs/b2_llm/batch_7133/` +
`batch_7133_scored.json` + `b2_7133_anchors.json`, `models/runs/b2_llm/batch_{35159,gdansk,7133}_think/` +
`batch_think_scored.json`, `gdansk_dyn_anchors.json`. Scripts `analysis/b2_llm_benchmark.py`,
`analysis/b2_batch_35159.sh`, `analysis/b2_batch_gdansk.sh`, `analysis/b2_batch_7133.sh`,
`analysis/b2_batch_think.sh`, `analysis/b2_batch_think_b32k.sh`, `analysis/b2_score_7133.py`,
`analysis/b2_score_think.py`, `analysis/b2_7133_anchor_probe.py`, `analysis/dyn_exact.py`,
`analysis/disjoint_baseline_probe.py`.

## Question
Given a routing security game specified in full and no tools, do pinned open-weight language models state a
calibrated mixed strategy and adapt in context against an adaptive adversary?

## Game
- Instances: kaliningrad 35-159 (N=3, K=1, band 0.15-0.95, 12 routes); gdansk OD 249-95, one of gen27's six
  held-out test ODs (R=10, det/eq 2.45); kaliningrad 71-33 (N=3, K=1, k-extra 8, menu-select, band 0.15-0.95,
  R=11, exact attacker).
- Register (a) deterministic: choose one route, adversary best-responds, scored as the chosen route's worst case.
- Register (b) stated-strategy: commit to a probability distribution, scored as the exact stated mixture under
  oracle best response (`env.exploitability_of_occupancy_dist`).
- Register (c) agentic-sequential: T=30 sorties against the gen19 pattern-of-life adversary (w=3, tau=0.15) with
  per-sortie outcome feedback, scored as realised mean mission failure.
- The prompt states per-edge lengths, interception probabilities, the mission objective and the register's
  adversary model. No tools. Prompts, model ids and full transcripts are logged verbatim in every per-run JSON.
- Models: `llama-3.3-70b` (AWQ-INT4, 32K ctx) and `qwen3-27b` (64K ctx; served alias, identity Qwen3.6-27B),
  served locally at pinned open weights, thinking off for the three cells below.
- Footprint per model per cell: (a) x10 seeds, (b) x10 seeds, (c) x5 episodes; temperature 0.7, max-tokens
  12000, one retry per conversation. Seeds drive the pre-registered label permutation.
- Thinking-mode rerun: qwen3-27b only (llama has no deliberation mode), `enable_thinking: true` via
  `chat_template_kwargs`, all three instances, same registers, footprints, seeds, temperature and endpoint;
  reply cap 16,000 tokens on registers (a) and (c), 32,000 on register (b). Thinking rows are labelled and
  never pooled with thinking-off rows.
- Eval only; no training anywhere in this act.

## Criteria
Pre-registered expectations, both directions reportable, judged per model, per instance and per mode, never pooled.
- (a) lands at or above loss_det.
- (b) on 35-159 lands between the uniform stack and the equilibrium; on 71-33 lands at or above uniform-full-menu
  0.2252 and far above v* 0.1276, with a possible model x instance split.
- (c) discovers anti-repeat hedging from feedback, dropping below the model's own static play, with repeat rate
  below the uniform agent's ~0.32, staying above the exact dynamic optimum and short of the trained policy.
- Scored side question: does the stated mixture put mass on the disjoint core (distance to the
  inverse-vulnerability stack versus to uniform), and does the rationale name route independence or shared edges?
- Thinking rerun: registered readable positive is variance collapse toward the stack on 71-33 register (b);
  registered readable negative is spread unchanged or worse. Noise floors from the thinking-off measurements,
  (b) sd 0.076 at n=10 and (c) sd 0.043 at n=5.

## Baselines
- loss_det: worst case of the best fixed single route. Equilibrium v*: one-shot game value against the exact attacker.
- Inverse-vulnerability disjoint stack: two-line max-flow heuristic over edge-disjoint routes; on 71-33 at K=1
  the worst-edge and budget-max definitions coincide and attain v* exactly.
- Uniform-disjoint, uniform-full-menu and inv-vuln-full stacks: naive stacks over the disjoint core or the full menu.
- SACRED static / dynamic: the trained policy on the same instance.
- Dynamic optimum: exact minimum-mean-cycle value (Karp, `analysis/dyn_exact.py`). Rotation: best fixed cyclic
  order. Composed anti-repeat: rule over the disjoint core. iid_eq: static equilibrium mixture played i.i.d.
  static_det: best fixed route in the dynamic register. Matched-budget window-Q: budget-matched tabular control.
- gen27 zero-shot policy: the trained dynamic generalist deployed on gdansk 249-95 unseen.

## Results

### The three cells, thinking off (35-159 and gdansk 2026-07-17; 71-33 2026-08-12)
| cell | register | anchors | llama-3.3-70b | qwen3-27b |
|---|---|---|---|---|
| 35-159 | (a) | loss_det 0.699 | 0.978 | 0.841 |
| 35-159 | (b) | eq 0.206 · disjoint stack 0.250 (inv-vuln 0.241) · SACRED 0.256 · uniform-full 0.442 | 0.604 +/- 0.100 (gate 1.0/3) | 0.523 +/- 0.161 (gate 2.1/3) |
| 35-159 | (c) | opt 0.0413 = rotation · SACRED 0.050 · iid_eq 0.1468 · static_det 0.613 | 0.177 +/- 0.018 (best 0.149) | 0.297 +/- 0.176 (best 0.059) |
| gdansk 249-95 | (a) | loss_det 0.740 | 0.867 | 0.867 |
| gdansk 249-95 | (b) | eq 0.302 · disjoint heuristic 0.333 · uniform-menu-stack 0.694 · det 0.740 | 0.798 +/- 0.072 (gate 2.0/3) | 0.354 +/- 0.066 (gate 2.1/3) |
| gdansk 249-95 | (c) | opt 0.0723 (Karp) · gen27 zero-shot ~0.098 (0.44x cap) · rotation 0.2069 · iid_eq cap 0.223 · static_det 0.692 | 0.325 +/- 0.059 (best 0.214) | 0.394 +/- 0.047 (best 0.346) |
| 71-33 | (a) | loss_det 0.4199 | 0.641 (route 4 on 9/10) | 0.572 (route 5 on 10/10) |
| 71-33 | (b) | v* 0.1276 = inv-vuln stack · uniform-disjoint 0.1666 · uniform-full 0.2252 · inv-vuln-full 0.2502 · SACRED 0.160 +/- 0.003 | 0.619 +/- 0.000 (gate 1.2/3) | 0.254 +/- 0.076 (gate 2.1/3) |
| 71-33 | (c) | opt 0.0313 · rotation 0.0387 · composed anti-repeat 0.0423 · full-menu anti-repeat 0.0728 · iid_eq 0.0967 · static_det 0.3835 · SACRED 0.0462 +/- 0.0008 · window-Q 0.0472 | 0.069 +/- 0.024 (best 0.033) | 0.054 +/- 0.043 (best 0.000) |

The 71-33 anchors were reproduced before any call (`analysis/b2_7133_anchor_probe.py`), the register-(b) value
v* 0.1276 being attained exactly by the inverse-vulnerability disjoint stack at K=1.

On 35-159 the measured anti-repeat rate in (c) is ~0.00 and qwen's episodes span 0.059 to 0.605. On gdansk
249-95 llama's stated mixture scores above the best deterministic route (0.798 versus 0.740), and both models
miss the disjoint heuristic (0.333) and the equilibrium (0.302).

71-33 per-seed (b): llama 0.619 x10 (two distinct supports, identical value, core mass 0.28); qwen 0.298, 0.157,
0.173, 0.298, 0.375, 0.298, 0.128, 0.298, 0.216, 0.297 (ten distinct distributions, core mass 0.55, below
uniform-full on 4/10 seeds, below uniform-disjoint on 2/10). Per-episode (c): llama 0.0614, 0.0330, 0.0798,
0.0652, 0.1063 (repeat-in-window rate 0.00); qwen 0.0598, 0.0664, 0.1241, 0.0000, 0.0174 (repeat rate 0.21;
the 0.0000 episode is a realisation of the sampled adversary, not a stationary value).

Side question on the 71-33 post-probe transcripts: qwen names the exact maximal independent set {0,1,2,3,4,5}
on 6/10 probes (pairwise-disjoint on 7/10); llama names the valid pair {1,3} on 6/10 and invalid larger sets
otherwise (exact 0/10).

Criterion outcomes on 71-33: (a) at or above loss_det on both models; (b) at or above uniform-full-menu 0.2252
on both means; (c) below each model's own static play and above optimum 0.0313, with SACRED 0.0462 ahead of
both means.

### Thinking-mode rerun, qwen3-27b only (2026-08-13)
Scored table; the 16k-cap register-(b) sidecars are excluded from it.

| cell | register | thinking on | thinking off | anchors |
|---|---|---|---|---|
| 71-33 | (a) | 0.572 (route 5 x10) | 0.572 | loss_det 0.4199 |
| 71-33 | (b) | 0.291 +/- 0.060 [0.128, 0.386] | 0.254 +/- 0.076 | v* = stack 0.1276 · uni-full 0.2252 · SACRED 0.160 |
| 71-33 | (c) | 0.057 +/- 0.025 (best 0.034) | 0.054 +/- 0.043 | opt 0.0313 · rot 0.0387 · SACRED 0.0462 · iid 0.0967 |
| 35-159 | (a) | 0.841 (route 6 x10) | 0.841 | loss_det 0.699 |
| 35-159 | (b) | 0.261 +/- 0.099 [0.209, 0.555] | 0.523 +/- 0.161 | eq 0.206 · stack 0.250 · SACRED 0.256 · uni-full 0.442 |
| 35-159 | (c) | 0.065 +/- 0.027 (best 0.037) | 0.297 +/- 0.176 | opt 0.0413 = rotation · SACRED 0.050 · iid 0.1468 |
| gdansk | (a) | 0.867 (route 1 x10) | 0.867 | loss_det 0.740 |
| gdansk | (b) | 0.326 +/- 0.046 [0.303, 0.464] | 0.354 +/- 0.066 | eq 0.302 · stack 0.333 · uni-full 0.694 |
| gdansk | (c) | 0.133 +/- 0.065 (best 0.068) | 0.394 +/- 0.047 | opt 0.0723 · SACRED ~0.098 · rot 0.2069 · iid 0.223 |

Mixture structure: seven of ten 71-33 seeds commit near-identical mixtures (0.2975-0.3007, modal 0.298) and one
seed sits at 0.128; eight of ten gdansk seeds commit exactly 0.3111, and the modal mixture sits below the
two-line stack (0.311 versus 0.333) on 9/10 seeds and within 3% of the equilibrium (0.302). Thinking-off mode
produced ten distinct mixtures per cell.

Comprehension gates 3.0/3 on every conversation of every thinking cell (thinking off 1.2-2.2). On the post-probe
thinking-mode qwen names the exact six-corridor independent set on 9/10 probes (thinking off 6/10). Register (c)
repeat rate 0.00 on the Kaliningrad instances.

Criterion outcomes: the registered readable positive (variance collapse toward the stack on 71-33 register (b))
did not occur, and the readable-negative branch fired (0.291 +/- 0.060 against 0.254 +/- 0.076). Register (a) is
unchanged on every cell. Register (c) lands between the trained policy and static play on every cell and reaches
neither the trained policy nor the exact optimum anywhere.
