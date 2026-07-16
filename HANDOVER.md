# HANDOVER.md: master state & onboarding for the incoming agent (2026-07-07, refreshed 2026-07-16 NIGHT)

> **★★★★★★ 2026-07-16 NIGHT (fresh agent: START HERE; this banner supersedes the stack below).**
> Today's session (the Block R rescue programme, Kilian's full autonomous launch authority)
> changed what the thesis claims and where its positive results live. Read in this order:
> 1. **`CRITIQUE_16-07-26.md`** — the disjoint-baseline finding (a 2-line max-flow heuristic
>    matches/beats every trained static K=1 number; the old ladders' "uniform" anchors were
>    padded-menu strawmen). VERIFIED, folded into all seven affected ledgers with binding
>    wording rules (R0a appendices). Never cite an Obj-5/ZST comparative claim without them.
> 2. **`NEXT_STEPS_MASTER.md` Block R** — the active programme + its PROGRESS ticks.
> 3. **`experiments/gen26_kboundary.md`** — the K-to-min-cut act, COMPLETE: at K = m-1 SACRED
>    beats both max-flow heuristic variants on the exact yardstick (K=3 n=3: 0.664 +/- 0.018 vs
>    0.737/0.738, eq 0.604); past the exact wall (71-33 m=6, certified greedy yardstick,
>    fidelity <= 1.8%): K=5 = 0.667 +/- 0.016 < uniform-disjoint 0.705 (STRONG < 0.638 not met);
>    K=6 single seed 0.718 beats both variants. SECOND-PASS TEMPERING (in the ledger): at K=5 a
>    FULL-MENU uniform ties SACRED (0.666) and a TABULAR-FP-with-greedy-BR learner beats it
>    (0.621/0.690), so the surviving static claim is the BOUNDARY MAP
>    (`assets/k_boundary_map.png`), not deep-RL superiority; K=6 needs n=3 before any sentence.
> 4. **`experiments/gen27_dynamic_generalist.md`** — the rescued ZST act, PRIMARY + STRONG
>    PASSED 3/3 seeds: one history-aware policy, trained on 3 cities, ZERO-SHOT on Gdansk beats
>    the static cap at **0.639 +/- 0.025** (every static object beaten by MEASUREMENT incl. the
>    local static optimum; full-menu anti-repeat fails at 1.37x; the composed
>    disjoint+anti-repeat rule 0.50-0.61 bounds below; worst-case premium 1.57x = the
>    regime-conditional scope sentence). NO-WINDOW causal control TRAINING OVERNIGHT
>    (`models/runs/gen27_dyn_generalist/seed0_nowin.*`): fold its result + tick the ledger when
>    it lands (expected ~iid_eq, as gen19's control).
> 5. **`AERIAL_BRANCH_HANDOFF.md`** — if you are the AERIAL instance: the complete build brief
>    for the free-flight act (Kilian 2026-07-16: TRAINED aerial result = MUST-HAVE). Work on a
>    new branch per the brief; do not disturb this branch's running jobs.
> 6. **B2 (the LLM benchmark) is LIVE and stays in the ORIGINAL conversation** (design
>    finalised: llama-3.3-70b + qwen3-27b, unhinted, 3 instances; first live transcript
>    `scratch/b2_livetest_llama_transcript.txt`; gateway now DIRECTLY reachable at
>    http://100.88.32.88:8080/v1, key iits-local-key; qwen start needs Kilian's ssh). Do not
>    re-run B2 from a fresh session while that conversation is active.
> **New operating dogmas earned today (also in SYSTEM.md):** (a) BASELINE COMPLETENESS is
> pre-registered like metrics — every ladder carries the strongest naive baseline a
> practitioner could write (max-flow/disjoint variants; composed rules in dynamic games);
> (b) multi-process launches cap ALL thread pools (OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 +
> torch threads), not just torch's — uncapped pools showed up as 36% system time; (c) commit
> critique artefacts IN the session that produces them (the lost-15-07-file lesson); (d) screens
> select instances by the HEURISTIC-gap, not det/eq. House rules otherwise unchanged (ledgers
> before CPU; prose docs carry pointers, numbers live in ledgers; never compare across SHAs;
> no multiple-choice prompts to Kilian). Suite 167 green. FAR hard-due 30 July; freeze 3 Aug;
> thesis 28 Aug. The banner stack below is the historical state.

> **★★★★★ UPDATE 2026-07-16: B2 IS UNBLOCKED — a local LLM workbench exists (Prof. Angeloudis's
> box), no API keys or spend needed.** The one open computational item (B2, the agentic-LLM
> exploitability benchmark) was blocked on commercial API keys; it no longer is. Prof. Angeloudis
> provided an SSH-reachable GPU server hosting an OpenAI-compatible LLM stack. Full connection +
> inventory details below; the essentials:
> - **Host `cv-iits-w05`**, reached over Tailscale. **SSH: `ssh killian@100.88.32.88`** (NOTE: the
>   username is `killian`, DOUBLE-L, not `kilian`; password `tsl2026`). Instructions file:
>   `../../Connecting to local LLM workbench.txt`.
> - **The gateway on port 8080 is NOT directly reachable from Kilian's Mac** (the box is a *shared*
>   Tailscale node from the prof's tailnet; its own IP is `100.73.116.67`, the Mac sees it as
>   `100.88.32.88`, and the tailnet ACL allows SSH but not 8080). **Workaround (verified working):**
>   an SSH tunnel `ssh -N -L 18080:localhost:8080 killian@100.88.32.88`, then hit
>   `http://localhost:18080/v1`. (Alternative: ask the prof to open 8080 in the ACL / `tailscale serve`.)
> - **OpenAI-compatible gateway**, API key `iits-local-key` (`Authorization: Bearer iits-local-key`).
>   Every request+response is auto-logged to `/home/llm/vllm-server/audit/YYYY-MM-DD.jsonl` (the B2
>   transcript record, for free; note the log is shared/readable by the `llm` group = pan, leo).
> - **Live model: `llama-3.3-70b`** (AWQ-INT4, 32K ctx) — tested end-to-end, generates. Three more
>   defined-but-disabled: `qwen3-27b` (27B, 64K ctx; fits alongside the 70B per the VRAM budget,
>   one `./start.sh start qwen3-27b`), `qwen3-coder-30b`, `qwen3-coder-next` (80B MoE). 2x RTX A6000
>   (96 GB VRAM), 251 GB RAM, Ubuntu 24.04. Config is `models.json` (single source of truth);
>   operate via `/home/llm/vllm-server/start.sh {start|stop|status}`. Full README at
>   `/home/llm/README.md`. Kilian's account is in groups `sudo, docker, llm` (can start/stop/edit).
> - **Why it matters:** this is arguably BETTER than the pre-registered commercial-API design: pinned
>   open-weight models (reproducible), no tools available (exactly the informative no-tools register
>   the B2 pre-registration argues for), free, and on the supervisor's own hardware. **B2 is ready to
>   run** the moment Kilian says go: point `scratch/b2_llm_benchmark.py` at the tunnelled endpoint
>   (`--base http://localhost:18080/v1 --key iits-local-key --model llama-3.3-70b`), then enable and
>   run Qwen3-27B as the second model. NOTE: the harness currently supports `--provider anthropic|
>   openai|dry`; add a generic OpenAI-compatible base-url path (trivial: the OpenAI branch already
>   posts to `/v1/chat/completions`, just parameterise the URL) before the live run.
> - **Operational rule respected:** no benchmark was fired at the shared GPU box without Kilian's go;
>   only read-only survey + three tiny test generations (~60 tokens total) were run.
>
> **★★★★★ HANDOVER 2026-07-15 (fresh agent: START HERE).** State is STABLE and fully committed
> (working tree clean at `5cd1e02`; test suite green; branch `gen08-interdiction`). NOTHING is
> mid-flight and nothing should be launched. **Reading order:** this banner stack top-down, then
> `NEXT_STEPS_MASTER.md` (the active checklist), then its §0 onboarding order
> (`CRITIQUE_12-07-26.md` -> `CRITIQUE_EXAMINER.md` -> `NEXT_STEPS_11-07-26.md` -> the `experiments/`
> ledgers). **Where things stand:** Blocks A + B of `NEXT_STEPS_MASTER.md` (the entire 2026-07-12/13
> claims-defence + differentiator programme) are COMPLETE; the SOLE open computational item is
> **B2's live-LLM benchmark, blocked on API keys + Kilian's spend decision** (harness built and
> dry-run-validated, `experiments/b2_llm_benchmark.md`). **Block C (chronicle/doc-hygiene, the FAR,
> the interactive exhibit, and the THESIS_STORYLINE four-act rewrite = NEXT_STEPS item 7) is NOT
> started, deliberately paused on Kilian's standing instruction that writing stays off the table.**
> The one EXTERNAL deadline that binds regardless is the **Final Activities Report, hard-due 30 July**
> (content freeze 3 Aug). **So a new agent's correct default is:** do not launch training; await
> Kilian's explicit go before opening Block C (FAR / storyline) or unblocking B2; do not re-open the
> settled gates (gen17/C4, gen18/C2, gen23/C1) or the pre-deferred A4 K=5 cell. Everything a claim
> rests on lives in an `experiments/` ledger; cite numbers only from there. The dated banners below
> are the full state this rests on, newest first.
>
> **★★★★ ACTIVE PLAN AS OF 2026-07-12 = `NEXT_STEPS_MASTER.md` (read it FIRST, then this banner
> stack).** Agreed Kilian + the outgoing Fable instance on Fable's last day: an ordered checklist
> merging the two 2026-07-12 critiques (`CRITIQUE_12-07-26.md`, `CRITIQUE_EXAMINER.md`), with
> Kilian's ordering decision: computational claims-defence and differentiator experiments FIRST;
> the FAR (hard external deadline 30 July), the interactive exhibit and the THESIS_STORYLINE
> rewrite at the BACK. House rules unchanged (no training without Kilian's explicit go;
> pre-registered ledger per item; numbers only in ledgers). The banners below are the project
> state that plan operates on.
>
> **UPDATE 2026-07-13 (the autonomous claims-defence run, Kilian's launch authority): BLOCKS A
> AND B ARE COMPLETE** (sole exception: B2's live LLM runs await API keys; the harness is built
> and dry-run-validated). The progress ticks + one-line results live at the top of
> `NEXT_STEPS_MASTER.md`; the four claim-changing findings (distillation/retrieval match the
> generalist where labels exist; per-edge map-reading is not the transfer mechanism; the
> gap-closure ladder decays 0.90 -> 0.04; d3-Gdansk's 0.109 was seed-specific) carry BINDING
> wording rules recorded in their ledgers (`gen24_distill`, `zst_map_robustness`,
> `a6_a7_a8_completions`, `d3_gdansk`) and in memory (`zst-act-rescope-2026-07-12`). The
> completed positives: prevalence figure, risk-aversion three-regime law, multi-OD gap 14.4%,
> integration gap (joint = safe default), vanilla n=3 + DR causal control (best-response
> pressure is the causal transfer ingredient, `gen25_dr_control`). Block C (chronicle, FAR,
> exhibit, storyline rewrite) is NOT started, paused on Kilian's instruction; the FAR's 30 July
> deadline binds regardless.
>
> **★★★ EXPANSION PROGRAMME COMPLETE (2026-07-11; NEXT_STEPS list all done bar the deferred A4).**
> The 2026-07-11 experiment list is banked. **gen20/F2 PASSED = Obj-1's antagonist AGENT closed
> POSITIVE:** a LEARNED interdictor co-evolves to **0.81x the oracle's strength** and its defender
> lands at 0.330 (within 0.074 of the oracle-trained 0.256, beats ALNS) - and the campaign REVERSAL
> (the learned adversary that could not learn congestion CAN learn to interdict) validates the
> Act-III pivot (`experiments/gen20_f2_learned.md`). **gen21 vanilla control: transfers 2.34x =
> WORSE than random-init 1.99x** -> adversarial training is CAUSAL for ZST (Obj-5 transfer control,
> measured not inferred). **gen22 Istanbul rotation PASSED (1.880 < random 2.30):** two rotation
> points now (Gdansk 1.68, Istanbul 1.88) = transfer holds to whichever city is held out.
> **Zero-shot K/N rows:** the hedge SURVIVES budget shift (K=2 1.29x) and fleet shift (N=5 1.79x).
> **Whole-Kyiv (6083n):** 1.88x, beats random - the scale axis. **D3-on-Gdansk (poster exhibit):**
> the composite on a never-trained city (policy-vs-oracle design corr 0.109 vs 0.768 in-dist = the
> ZST-vs-LP backbone). **gen23/C1: ERB-from-ALNS HURTS** (deterministic metaheuristic demos bias a
> mixed-strategy learner toward exploitable determinism - Obj-3 closed with a mechanism). Doc
> hygiene done (chronicle entries 19-21, dual-selection folded in, select-on-train default). **A4
> K=5 cell DEFERRED** (drop-first, hedged scaling claim, trainer-refactor risk; verified core
> stands). Suite 161. **All five objectives now have trained evidence in demonstrated-or-better
> form; ZST realised at OD/city/scale with rotation + K/N robustness; SBO is a full stack; the two
> failed gates (gen17/18) and one negative (C1) are measured boundaries.** NEXT (Kilian's call):
> the storyline compression + self-critique (NEXT_STEPS item 7); writing was kept off the table.
> The banner below is the prior state.
>
> **★★ EXPANSION PROGRAMME: ZST + C-CHAIN COMPLETE (2026-07-11 morning).** **gen16 PASSED = the
> first cross-CITY zero-shot transfer** (train Kaliningrad+East London+Istanbul, held-out GDANSK
> 1.677 +/- 0.072x its equilibria, beats loss_det 17/18; A2-RESCUE confirmed: 1.90 vs random 2.43
> on the graph where single-source tied random; `experiments/gen16_multicity.md`). Transfer ladder:
> in-graph OD 1.59 (gen15) -> held-out city 1.68 (gen16) -> single-source cross-graph ~random (A2).
> **gen17/C4 FAILED the hold-the-tail bar** (annealed smoothing delays, does not prevent, the
> drift; 4 failed attempts across 2 instances/eras -> the transient finding is INHERENT;
> best-checkpoint discipline is FINAL; hard gate closed). **gen18/C2 FAILED** (follow_w trained to
> 2.93 - the lr fix worked - yet followers still collapse to fixed routes, stack 0.08: the
> structural-stacking caveat is a REAL boundary, cleanly measured post-fix; future work is
> exploration-side). Maps: gdansk/east_london/istanbul in `data/maps/` (length-repaired;
> `scripts/extract_city.py` fixed). Remaining optional list items: A4 large-K training cells,
> B1-lite-2, B1(full, adds demand-S), B3, C1, F2, B5. The banner below is the prior state.
>
> **★★ B1-lite-1 (gen19) PASSED = the D restored + solved (2026-07-11).** First SACRED game with
> WITHIN-EPISODE dynamism (pattern-of-life interdictor softmax-BRs to the defender's realised
> routes over a 3-sortie window). Oracle screen: static_det 0.613 >> iid_eq 0.147 > history_opt
> 0.049 (dynamism pays, exact optimum). SACRED history-aware **0.050 +/- 0.001 ~ history_opt
> 0.049** (PRIMARY/STRONG 3/3); NO-WINDOW causal control 0.148 = iid_eq (gain IS the window
> conditioning); worst-case row: policy marginal 0.219 ~ eq 0.206 (no fragility vs a non-adaptive
> attacker). `experiments/gen19_b1lite1.md`. On S vs D: the headline has stochastic OUTCOMES
> (Bernoulli interception + risk objective) + strategic uncertainty, and now within-episode D
> (gen19); it does NOT have demand-side S (Poisson arrivals) - that is full B1, scoped.
>
> **★ EXPANSION PROGRAMME: KEYSTONE ARC COMPLETE (2026-07-10 evening; `DIRECTION_EXPANSION.md`).**
> Kilian: "disregard thesis writing"; full launch authority. **DONE:** C3 (gen14: both headlines
> n=10 CIs - MC 0.256 [0.246,0.266], SC paired dD 0.175 [0.137,0.213] excl 0; the stats weak point
> is closed); **A1 (gen15) PASSED** = first TRAINED zero-shot transfer (held-out mean ratio 1.59
> +/- 0.10, beats loss_det 17/18 cells; `experiments/gen15_generalist.md`); D1 SBO loop PASS+STRONG
> (Obj-4 proper); A3 amortisation (honest: wall-clock does not favour the policy; ZST+D3 carry the
> scaling story); **D3 composite PASS** (surrogate over the TRAINED policy Spearman 0.959; policy-vs-
> oracle design-target corr 0.768 = designing against the real policy differs from the equilibrium);
> B4, B0, A4-core, D2 all DONE. **KEY FINDING (`experiments/a2_graph_transfer.md`): a single-source-
> graph generalist transfers across OD PAIRS but NOT across GRAPHS (ties random on a different
> graph)** -> cross-CITY ZST needs a MULTI-GRAPH generalist (train on N cities, hold one out), not
> single-source transfer; the 2nd-city (Kyiv+) plan is reframed accordingly (Kilian sourcing
> cities). Suite 161. NEXT: multi-city generalist once city graphs exist; C4 (bounded last-iterate
> attempt on 35-159); C2 (learned-follower redo post-fix). All ledgers under `experiments/`.

> **★★★★★★★ BOTH HEADLINES POST-FIX (2026-07-10 morning, gen13-lock PASSED). READ
> `NIGHT_REPORT_2026-07-10.md` + `experiments/gen13_lock.md` FIRST; numbers live ONLY in ledgers.**
> **Multi-convoy headline = gen13-lock (35-159, held-out-screened instance, honest
> representations): the LOCKED ladder is in `experiments/gen13_lock.md`; the pre-fix 62-97 number
> retires to the methods narrative.** Single-convoy headline = gen10-SC
> (`experiments/gen10_postfix.md`, confirmed). The two-headline pre-fix/post-fix asymmetry is
> RETIRED. Obj-4 MET in reduced form (`experiments/f3_sbo_demonstrator.md`); Obj-5 disruption
> curves banked, SACRED < ALNS 10/10 cells (`experiments/gen12_sweeps.md`); gen11/gen11b menu-head
> decomposition in `experiments/gen11_menuhead.md`; ZST step 0 = pre-registered scoping negative,
> B2-S closed (`experiments/zst_step0.md`); oracle-scaling re-measured
> (`scratch/oracle_scaling_output_v2.txt`, gen09 ledger update note). Chronicle:
> `SACRED_PROGRESS.md` entries 16-21 (the chronicle now spans the audit, the expansion programme,
> the ZST city-scale arc and the boundary gates). NEXT: thesis writing on the two post-fix headlines; optional
> remaining experiments (ZST step 1, F2 demo) per the night report's decision list.

> **★★★★★ AUDIT + NODE-ORDERING FIX + gen10 RE-RUNS (2026-07-09 late, fix SHA `e9acb56`). READ
> THIS FIRST; it amends the banners below.** An examiner-grade audit (`CRITIQUE_INTERDICTION.md`)
> found a project-wide representation bug (featurize_state sorts node ids; every consumer indexed
> by dict insertion order -> every net ever trained read a fixed permutation of the wrong nodes'
> embeddings). Fixed (`node_index_map` + 3 regression tests, suite 149 green) together with a
> role-alpha Bellman-target fix and EXACT fleet-route evaluation. Pre-registered re-runs
> (`experiments/gen10_postfix.md`, Kilian's explicit go):
> - **Single-convoy gen10-SC: PASSED every clause, pooled sacred TAP 0.276 vs vanilla 0.480**
>   (banked B2-P3: 0.362/0.477): ~44% of the residual equilibrium gap was the bug. RECOMMENDED to
>   supersede 0.362 as the single-convoy headline (Kilian to confirm).
> - **Multi-convoy gen10-MC: REGRESSED to best-ckpt TAP 0.447 +/- 0.029** (exact estimator;
>   prediction violated, reported as measured; Obj-5 ordering still holds: 0.447 << ALNS 0.699 <<
>   post-fix vanilla 0.859). The citable multi-convoy number stays the banked pre-fix best-ckpt,
>   now correctly stated as the EXACT re-evaluated **0.295 +/- 0.024** (SHA `ad70a9c`; the MC 0.283
>   carried a min-selection-on-noise bias), caveat disclosed. **gen10-MC2 diagnostic (2026-07-10,
>   Kilian's go, SHA `1ff5526`): NO recovery: 0.447 +/- 0.008, identical with the role-alpha fix
>   reverted and a doubled horizon -> the regression is the menu head losing discriminability once
>   embeddings are correct (the pre-fix permutation was an accidental route-identity hash). Next
>   step = a DESIGN change (gen11 proposal: undiluted per-route cost+vulnerability features at the
>   menu head, the lever-2 pattern; also the ZST enabler), pre-registered separately, launch =
>   Kilian's go. Kilian 2026-07-10: the single-convoy supersession (0.362 -> gen10-SC 0.276) is
>   CONFIRMED.** The critique also scores all five objectives (Obj-4 SBO unmet: the F3
>   demonstrator is an afternoon; ZST doubly blocked pre-fix, now needs an edge-vulnerability
>   feature), and ranks the pre-freeze programme (§8). `SACRED_PROGRESS.md` entry 17 is the
>   narrative; all banked numbers below stand at their pinned SHAs with the caveat disclosed.

> **NEW-AGENT READ ORDER (2026-07-09; the banners below are a reverse-chronological stack = current
> state first, then how we got here). The project has TWO banked, pre-registered headlines, both
> scored against a computable minimax equilibrium; the experimental work is essentially DONE and the
> next phase is THESIS WRITING.**
> 1. **This top banner** (the current state: multi-convoy headline LOCKED at best-checkpoint TAP
>    0.283 +/- 0.021; single-convoy B2-P3 0.362 the other headline).
> 2. **`experiments/gen09_multiconvoy.md`** - the AUTHORITATIVE multi-convoy record: the locked
>    headline + ladder + fairness rows, the gen09-STAB-1/2/3 stabilisation arc, and the oracle-scaling
>    "why not just solve the LP" probe.
> 3. **`REDESIGN_INTERDICTION.md`** (the north star: why interdiction; §10 = multi-convoy) ->
>    **`THESIS_STORYLINE.md`** (the 4-act argument, Act IV realised) -> **`SACRED_PROGRESS.md`**
>    entries 12-16 (the narrative chronicle) -> **`ROADMAP.md`** Phase M (the plan, findings, future work).
> 4. **`experiments/gen08_interdiction.md`** - the single-convoy interdiction ledger through the banked
>    B2-P3 pass, plus the Phase M sections (superseded-for-the-number by gen09).
> 5. Then the campaign history as needed: **`SACRED_PROGRESS.md`** 1-11, the **`experiments/gen0[1-7]*.md`**
>    ledgers, **`CONTEXT.md`** / **`DIRECTION.md`** / **`CRITIQUE.md`** / **`PROBLEM_REDESIGN.md`**
>    (all banner-marked historical), and **`SYSTEM.md`** (operating dogmas - read before running anything).
> **Operating rules (HARD): never launch training without Kilian's explicit in-conversation go; no
> multiple-choice prompts (prose + a firm recommendation); plan-first; oracle/screen probes are free.
> Thesis + poster due 10:00, 28 Aug 2026 (12k words); experimental freeze Aug 3.** Thesis planner brief:
> `../../thesis/THESIS_PLANNER_HANDOFF.md`.

> **★★★★ MULTI-CONVOY PHASE M COMPLETE - HEADLINE LOCKED (2026-07-09). READ THIS FIRST.**
> Phase M (multi-convoy interdiction, Fork A) is DONE. The LOCKED multi-convoy headline is the
> **fleet-route best-checkpoint on 62-97 k_extra=8** (shared-edge, 12-route menu, N=3, K=1, soft,
> mission; definitive 3-seed saved run gen09-HEADLINE, SHA `ad70a9c`, ledger
> `experiments/gen09_multiconvoy.md`): **best-checkpoint TAP 0.283 +/- 0.021** (3 seeds), ladder
> shortest 0.973 > vanilla ~0.945 > ALNS-forced-stack 0.912 > ALNS 0.699 > **SACRED 0.283** >
> equilibrium 0.216 - Obj-5 met (beats the SOTA metaheuristic AND the non-adversarial control).
> **The leader over-trains toward uniform after the best-checkpoint (inherent last-iterate FP cycling;
> resolved the standard single-convoy way = BEST-CHECKPOINT selection, drift saved + disclosed; three
> "hold-the-tail" stabilisation attempts gen09-STAB-1/2/3 are on record and failed, establishing the
> equilibrium is a reproducible transient).** The old 0.257 was an unsaved transient best-checkpoint,
> superseded by the locked 0.283 +/- 0.021. Single-convoy B2-P3 (0.362) stays the OTHER banked
> headline. **CAVEAT (honest, in the ledger): the fleet stacking is STRUCTURAL (followers copy the
> leader by construction), not learned.**
>
> **The learned-follower arc (6 attempts, the mechanistic SECONDARY result).** We tried to make the
> followers LEARN to copy the leader (genuine emergent coordination). Blocker = a chicken-and-egg:
> under independent exploration the convoys stack only at the ~2% random-coincidence rate, so the
> CRITIC never experiences the reward for following and the followers collapse onto fixed routes. Fix
> chain: (1) explicit route-correlation signal; (2) menu-select route-index action (shared-edge, NO
> walk trie); (3) two role-alphas (leader high entropy / follower ~0); (4) forced-copy warmup with a
> FROZEN mixing leader (demonstration bootstrapping / Obj-3 ERB); (5) LEVER 2 = a LEARNED, undiluted
> per-route "taken" term at the policy head AND the critic Q head; (6) prioritised replay of stacks +
> a steadier/softer smooth-FP attacker (switch_every 200, fp-tau 0.15). **THE BREAKTHROUGH: `follow_w`
> (the learned critic-side correlation weight) CLIMBS monotonically (attempts 4-6, 1.0 -> 1.25) = proof
> the critic CAN be made to value emergent coordination (the four-attempt blocker, fixed by the
> critic-side lever 2); the learned-coordination TIME-AVERAGE 0.482 beats ALNS (+0.217) and vanilla
> (+0.463).** BUT coordination SATURATED weak (tail stack ~0.18, follow_w plateaued 1.25) so 0.482 is
> WORSE than the structural fallback's 0.257 (full stacking > partial). Per the pre-committed exit
> criterion the FALLBACK is the headline; the learned bootstrap is a genuine-but-weaker Obj-3 result.
> Coordination-dynamics work is CLOSED (diminishing returns; fp-tau was the last reserved lever).
>
> **STRUCTURAL FINDING (`scratch/multiconvoy_instance_screen.py`, oracle only, NO training): DISJOINT
> route sets are ALWAYS near-uniform-leader (H/lnR >= 0.97 over 72 OD pairs) -> flat FP landscape ->
> the 33->71 leader failure is STRUCTURAL, not instance-specific. A non-uniform leader (asymmetry = an
> FP gradient) REQUIRES shared edges. 62-97 k8 was screened for asymmetry (leader H/lnR 0.63) + margin
> (ALNS/eq 3.2x) + high stack mass (0.97).** New dogmas: on a joint/correlated objective the
> coordination signal must be explicit AND reach the scoring head UNDILUTED, AND the CRITIC must value
> coordination (follow_w climbing is the diagnostic) - the actor cannot follow what the critic won't
> rank; disjoint routes give structurally uniform leader equilibria (asymmetry needs shared edges);
> zero-sum FP cycles by construction, judge on the stationary-tail TIME-AVERAGE, not per-eval play.
>
> **CODE/REPO STATE (branch `gen08-interdiction`, suite 146 green, COMMITTED through `908de0f`,
> tree clean, nothing running; gen09 ledger `experiments/gen09_multiconvoy.md` is authoritative):**
> `scripts/train_multiconvoy.py` (all machinery: menu-select, two-alpha, route-correlation, lever-2
> follow_w on actor+critic, forced-copy / frozen-leader bootstrap, prioritised replay `--stack-dup`,
> `--fp-tau`); `src/agents/sac.py` + `networks.py` (menu head + follow_w + role-alpha + per-sample
> target_entropy; featurize col 14 = route-correlation); `src/envs/multiconvoy_interdiction.py`
> (menu_select + route-index routing + taken_node_frac + absolute_vuln_norm); `scratch/
> multiconvoy_instance_screen.py`. All additive/flag-gated; the campaign path is byte-identical (14th
> feature col sliced off by `_clip_x`; follow_w exists only in menu+adversarial mode; +4 tests updated
> for the col-14 width bump). COMMITTED through `7bcb499` (2026-07-09). **NEXT: (1) DONE = MULTI-CONVOY
> HEADLINE LOCKED (best-checkpoint TAP 0.283 +/- 0.021, 3-seed saved run gen09-HEADLINE SHA `ad70a9c`,
> `experiments/gen09_multiconvoy.md`; the leader-stabilisation chase gen09-STAB-1/2/3 is closed - the
> leader over-trains toward uniform, resolved via best-checkpoint selection + disclosed drift, standard
> minimax discipline; leader-alpha floor + per-eval checkpoint saving + ALNS-forced-stack fairness row
> all landed). NO more leader experimentation (Kilian). (2) THESIS WRITING on the two banked headlines
> (single-convoy B2-P3 0.362; multi-convoy fleet-route best-checkpoint 0.283 +/- 0.021 << ALNS 0.699)
> - thesis planner brief `../../thesis/THESIS_PLANNER_HANDOFF.md`; (3) OPTIONAL future (each launch is
> Kilian's explicit go, only if runway before the freeze): the scaling tier (N / K / connectivity
> curves); learned coordination stays the banked Obj-3 secondary.**
> Operating rules UNCHANGED (never launch training without Kilian's in-conversation go; no
> multiple-choice prompts, prose + firm recommendation; plan-first; oracle/screen probes are free).
> The M3 SMOKE banner below is SUPERSEDED by this; `REDESIGN_INTERDICTION.md` §10 and the gen08
> ledger Phase M section carry the full detail; `SACRED_PROGRESS.md` entry 15 is the narrative.

> **★★★ M3 SMOKE UPDATE (2026-07-08): the multi-convoy trainer WORKS but needs CORRELATION. READ
> THIS FIRST.** Phase M: M0 (oracle proof), M1 (env+oracle, G-M1 gate), M2 (ALNS baseline reaching
> loss_det) are DONE and committed (HEAD 596708f); M3 (`scripts/train_multiconvoy.py`) is BUILT and
> smoked (1000 sorties, 110->135 N=3 K=1, latest-FP, seed 0). Suite 146 green.
> - **Timing (measured):** ~0.368 s/sortie STEADY (flat 367-368ms, no drift over 1000 sorties);
>   warm-up first ~11 sorties faster (replay buffer < batch); eval 2.7 s / 250 sorties. Full run
>   (3000 sorties/arm, vanilla+sacred, 3 seeds): **~50 min at 3-parallel `--threads 3`** (9 <= 10
>   cores), ~1.9 h serial. (Default 4 threads oversubscribes at 3-parallel = ~1.7 h; don't.)
> - **Result (ladder, 1000 sorties):** shortest_path 1.000 > ALNS 0.904 (optimal deterministic) >
>   vanilla 0.700 (TAP) ~ sacred 0.645 (TAP) >> equilibrium 0.328. **SACRED BEATS the optimal
>   classical planner (ALNS) = the Obj-5 metaheuristic win, and is STABLE (no collapse; occupancy-
>   entropy ~2.0 throughout, unlike the symmetric single-convoy).**
> - **THE OPEN PROBLEM (the crux to solve next): sacred ~ vanilla and both far from 0.328 because
>   the policy routes the convoys ~INDEPENDENTLY, not the CORRELATED stack-and-randomise optimum.**
>   The equilibrium puts ALL mass on "all 3 convoys on ONE random route" ([3,0,0]/[0,3,0]/[0,0,3]);
>   sacred's occupancy dist instead spreads over [2,1,0]0.20/[1,1,1]0.20/[1,2,0]0.15/... (independent
>   mixing), which cannot reach 0.328, and vanilla mixes incidentally to ~0.68 so the sacred-vs-
>   vanilla gap is inside the noise. The env EXPOSES earlier convoys' routes (via truck positions)
>   but the policy under-weights the signal.
> - **NEXT STEP (Kilian was deciding at session pause, confirm it first):** RECOMMENDED = make
>   correlation learnable: add an explicit "convoys-committed-so-far per route" feature to the
>   per-convoy observation so convoys 1,2 learn to FOLLOW convoy 0; re-smoke (expect sacred ->
>   toward 0.328 + a clean vanilla gap); THEN the full 3-seed launch + pre-register a gen09 ledger.
>   ALTERNATIVE: launch the primary (beats ALNS) as-is and treat correlation as the refinement.
> - Config: multi-convoy reward = -interception_loss(10)*mission_failure (sacred) / normalised travel
>   (vanilla); N-step sortie episode (terminal reward on the LAST convoy, bootstrap through the
>   fleet); FP attacker = oracle BR to the empirical OCCUPANCY play. Interactive Kaliningrad view:
>   `scratch/build_multiconvoy_view.py` (classical 90.4% vs SACRED 32.8% at the equilibrium).
> The MULTI-CONVOY PIVOT banner below is the strategic context; read it next.
>
> **★★ LATEST DIRECTION (2026-07-07 evening): THE MULTI-CONVOY PIVOT. Read THIS banner first, then
> the B2-P3 START HERE banner below.** After B2-P3 banked the single-convoy shared-edge headline we
> tried to broaden it (F1: the wave A/C sweeps) and hit two walls that redirected the programme:
> 1. **F1 launched then KILLED.** The single-convoy SYMMETRIC K-sweep (wave A, 33->71 disjoint) is
>    the ANTI-GOAL: uniform == equilibrium at every K, so vanilla mixes incidentally near-optimally
>    and adversarial training is a LIABILITY, sacred DESTABILISES under long training (A-K1 sacred
>    TAP 0.38 / 1.00 / 0.40 vs vanilla ~0.31; seed 1 FULLY collapsed with alpha runaway, the
>    flat-landscape SAC instability from the early campaign). Dropped from the deliverable.
> 2. **Obj-5's metaheuristic clause cannot be met by single-convoy.** One convoy on one route makes
>    a "SOTA adaptive metaheuristic" (ALNS) degenerate to shortest-path, so there is no non-trivial
>    classical opponent to beat.
> Under Kilian's **"make SACRED work"** mandate (HARD invariants: SAC, adversarial training, deep RL,
> robust routing; everything else fluid) the direction is now **MULTI-CONVOY interdiction**, which
> the ORACLE proves (three stress-test probes `scratch/multiconvoy_{probe,scan,spectrum,cost}.py`,
> NO training) both makes SACRED win AND fixes Obj-5, in the realistic regime of **SOFT
> (probabilistic) interception + a LOSS-AVERSE (mission-failure, P(>=1 convoy lost)) objective**:
> - GENERALISES: 20 random high-connectivity OD pairs, N=2 mission gap median **0.48** (80% > 0.30,
>   80% deterministic-coordination non-degenerate); N=3 median 0.58 -> the gap GROWS with fleet size.
> - REAL METAHEURISTIC: the deterministic coordinator trades travel-cost vs interception risk (a
>   genuine ALNS problem -> Obj-5 non-degenerate); SACRED dominates its cost-security frontier.
> - THE TRAP: a RISK-NEUTRAL (expected-fraction) objective dilutes the gap to ~0 (deterministic
>   spreading substitutes for mixing) -> the loss-averse objective is REQUIRED and also the realistic
>   one. Boundary: K < #routes (else the interdictor saturates coverage).
> - **ALL FIVE objectives now met** (Obj-5 metaheuristic FIXED; Obj-4 gains fleet composition; closer
>   to SDVRP). Confirmed against Kilian's "confirm all five or ask" gate.
> **Single-convoy B2-P3 (shared-edge, smooth-FP) stays the BANKED, proven headline; multi-convoy is
> the EXTENSION that meets the full objective set and wins bigger.** STATE: oracle proof DONE
> (positive, oracle-level only); the BUILD is next (multi-convoy env + mission-failure reward + ALNS
> baseline + training). Design: `REDESIGN_INTERDICTION.md` §10. Plan: `ROADMAP.md` (new phase).
> Record: `experiments/gen08_interdiction.md` (multi-convoy pivot section) + `SACRED_PROGRESS.md`
> entry 14. **Operating rules UNCHANGED: never launch training without Kilian's explicit
> in-conversation go (F1 itself was killed on his call); oracle probes are free; plan-first; no
> multiple-choice prompts; new dogma: on a symmetric/flat game adversarial training DESTABILISES,
> pick instances where vanilla provably cannot imitate the equilibrium.**
>
> **★ START HERE (new agent, 2026-07-07 end-of-session; the previous instance signed off after
> the B2-P3 PASS). READ ORDER for exact parity:**
> 1. **`REDESIGN_INTERDICTION.md`**: the north star: why the pivot was necessary (§0.5 full
>    evidence chain) + the equilibrium proof (§1).
> 2. **`experiments/gen08_interdiction.md`**: THE live ledger: every gate, pre-registration,
>    result and pinned SHA of the interdiction programme (G1/G2 PASSED; I3 wave 1 FAILED with
>    mechanism; B2-P/B2-P2 FAILED with mechanism; **B2-P3 PASSED: the citable headline**).
> 3. **`ROADMAP.md` Phase I**: findings to date + future work (short/mid/long term).
> 4. **`THESIS_STORYLINE.md`** (4-act arc, Act IV updated) and `SACRED_PROGRESS.md` entry 13
>    (the gen08 narrative in one entry).
> 5. Then history as needed: `DIRECTION.md`, `experiments/gen07_contested_matrix.md`,
>    `SACRED_PROGRESS.md` 1-12, `SYSTEM.md` (dogmas, updated), §1-5 of this file (the campaign).
>
> **RESULT STATE (branch `gen08-interdiction`, suite 131 green, tree clean, nothing running):**
> **THE PRE-REGISTERED HEADLINE IS BANKED (B2-P3, 2026-07-07, SHA `874d3f3`, ledger `ccb168e`):
> on the shared-edge instance (33->71, 11 routes, hidden K=1 interdictor) the exploitability
> ladder is shortest_path 1.000 > vanilla 0.477 > uniform 0.455 > SACRED 0.362 >> equilibrium
> 0.167 (TAP metric, 3/3 seeds + pooled, every pre-registered clause).** Adversarial training
> beats the deterministic default ~2.8x, the non-adversarial SAC control ~1.32x, and naive
> noise; vanilla sits ABOVE uniform (cost-calibrated mixing = predictability with extra steps,
> exactly as the oracle bound predicted). Strong form NOT met (distance-to-equilibrium
> 0.163-0.239): reported plainly in the ledger. The road there was three pre-registered
> dynamics iterations (pure-BR cycles / stale-mixture parks / smooth-FP passes): a measured
> fictitious-play dynamics study that is itself thesis material. **Dynamics work is CLOSED by
> Kilian's pre-committed exit criterion: do NOT reopen it.**
> Key gotchas paid for: SAC `reward_scale` default 0.001 far too small (use ~1.0 with
> interception_loss ~10); smokes validate plumbing, NOT slow-timescale dynamics (use the
> 1000-sortie drift signature); TAP (trailing-averaged policy) is the deployable estimator for
> FP learners; the walk trie is REQUIRED when candidate routes share first hops.
> **Operating rules (hard, learned the hard way this session): NEVER launch any training run
> without Kilian's explicit go in that conversation (a launch made under a briefly-broad mandate
> was killed mid-run); no AskUserQuestion multiple-choice prompts: prose + firm recommendation;
> plan-first; oracle-only probes (seconds, no training) are free.** Kilian's decisions on
> record: Kaliningrad graph; single convoy first; fallback-vs-upgrade exit criterion (upgrade
> achieved); freeze Aug 3 HARD; thesis + poster due 10:00, 28 Aug 2026.
>
> **⚠️⚠️ CURRENT DIRECTION (2026-07-06, latest): THE INTERDICTION-GAME REDESIGN.**
> Read **`REDESIGN_INTERDICTION.md` FIRST**: it is the north star. Short version: the campaign
> (gen03-06) and the exploitability follow-up (gen07) established that adversarial RL cannot win
> with a *congestion* adversary, because congestion is observable/reroutable/reversible, giving a
> reactive-dominated, FLAT attack landscape (proven: the corrected best-response gate lands at
> 0.35× random; every block is equally damaging). The fix is to change the ADVERSARY, not tune the
> old one: model Application 1's real threat, **interdiction/ambush** (hidden, irreversible,
> pre-committed), which is a **Stackelberg security game** where a deterministic router is
> maximally exploitable and the minimax **mixed strategy** (which SAC's entropy produces) provably
> cuts interception. **Proven at the equilibrium level on the real Kaliningrad graph: deterministic
> routing 100% intercepted → mixed 17-33%** (`scratch/interdiction_game_probe.py`). Decisions
> (Kilian 2026-07-06): Kaliningrad graph, single convoy first. **Read order now:**
> `REDESIGN_INTERDICTION.md` → `ROADMAP.md` (build plan) → `THESIS_STORYLINE.md` (4-act arc) →
> `experiments/gen08_interdiction.md` (forward pre-reg) → then the history below +
> `DIRECTION.md`/`experiments/gen07_contested_matrix.md` (why the exploitability path was
> necessary and where it hit the wall).
>
> **⚠️ (Superseded) REDIRECTION banner (2026-07-06, evening):** the exploitability reframe in a
> contested-resupply framing. Right instinct (minimax → worst-case robustness), but its
> destination-arena / learned-BR *realization* hit the flat-landscape wall; it has crystallised
> into the interdiction game above. `DIRECTION.md` records the reasoning bridge; `ROADMAP.md` is
> the active plan (interdiction phases). The campaign record below stands unchanged.

You are **Kilian Schwarz's SWE/planner agent on the SACRED MSc thesis** (Imperial College London,
supervisor Dr. Panagiotis Angeloudis). A prior Fable instance ran the project from the 2026-07-01
handover through the complete experimental campaign. **The campaign is finished.** Your job is to
support what comes next: thesis writing, the supervisor conversation, small follow-up
experiments only if Kilian asks: while preserving the standards that made the results
trustworthy. Everything below is verifiable in the repo; never cite numbers from anywhere but the
`experiments/` ledgers.

---

## 1. The finding (what the campaign established)

**Adversarial co-training, as naturally formulated for the stochastic-dynamic VRP, does not
confer robustness: and measurably worsens it.** The evidence chain, all pre-registered:

1. **gen03** (`experiments/gen03_robustness_dynassign.md`): ATLA co-evolution vs an identical
   vanilla-SAC control: no robustness delta (dD ≈ −250…−290 ± ~300–500, n.s.). Mechanism found:
   the **learned adversary attacks worse than uniform-random blocking** (D ≈ 0.6–1.9k vs random
   ≈ 1.7–2.1k), while a 40-line scripted heuristic hits 3–6× harder (≈ 4.9–5.9k).
2. **gen04** (`gen04_antag_gate.md`): after giving the adversary full motion observability (edge
   occupancy features), a retrained best-response attacker is *still* ≈ random (ratio 0.84 vs the
   pre-registered 1.25). Diagnosis: **entropy pinning** (max-entropy SAC over ~120 flat options
   with drowned advantages is mandated to play near-uniform) + reward SNR + γ-myopia.
3. **gen05** (`gen05_hybrid_matrix.md`): the matrix in the rich hybrid arena was
   **competence-void**: neither arm learned the task (clean W ≈ 5.6× greedy), and degradation
   near the saturation ceiling is compressed (**weak policies fake robustness**: an identified
   evaluation pitfall). One nugget: against a *competent* victim (greedy), the seeing learned
   attacker became the strongest attacker in the portfolio (+1667 > scripted's +1154/+714) -
   learned adversaries work where the reach mask aims for them and the victim is predictable.
4. **gen06** (`gen06_dynassign_matrix.md`): **the definitive result.** Competence gate PASSED
   (all six arms within +5.5–7.0% of greedy clean; gen03's band replicated). Primary
   **significantly reversed**: pooled `dD_targeted = −881 ± 284` (n=90, 0/3 pairings positive);
   the adversarially-trained arm is worse even under **its own training attacker**
   (dD_pathrand = −775 ± 244); dead even under random attack; zero clean cost. Robustness
   ranking: **greedy (4921) > vanilla (5196–5882) > adversarially-trained (6361–6575)**: the
   reactive classical dispatcher is the most robust policy measured (consistent with Ritzinger
   et al. 2015, cited in the literature review).

**Unifying mechanism:** the zero-sum latency reward (−queue/tick) buries each agent's
controllable contribution under a large uncontrollable shared baseline. The attacker's critic
can't resolve which blocks worked (→ pinned at ~uniform); the defender trained under attack gets
several-fold worse return SNR for the same sample budget (→ learns *less*, and the deficit
surfaces exactly where queue compounding amplifies policy quality: aimed attacks).

**Constructive contributions:** (a) four named preconditions for adversarial training to work in
this domain: a real coping channel in the action space, attacks with learnable structure, a
competence-first curriculum, variance-reduced (counterfactual-baselined) rewards; (b) the
evaluation methodology: pre-registration, competence gates, held-out attack portfolios,
per-policy best responses, paired instances, stochastic evaluation of max-entropy policies -
each earned by a specific documented failure (the static-3b retraction, gen05's ceiling trap).

## 2. Read in this order

1. **This file.**
2. **`SACRED_PROGRESS.md`**: the chronological run chronicle (10 entries, entry template at the
   top; Kilian wants every significant future run family appended there).
3. **`CRITIQUE.md`**: the 2026-07-02 critique that reframed the thesis; §5's skeptical-examiner
   questions still shape the writing.
4. **Ledgers** `experiments/gen02…gen06*.md`: pre-registered metrics + all citable numbers
   (+ portfolio JSONs beside them).
5. **`CONTEXT.md`** (banner explains what's historical), **`PROBLEM_REDESIGN.md`** (the pivot's
   design rationale), **`SYSTEM.md`** (operating dogmas: read fully before running anything),
   **`TASK.md`** (banner = plan state; body historical), **`docs/archive/`** (retired docs).
6. PDFs (extract with `.venv/bin/python` + pypdf):
   `../../MT_Literature_Survey_Kilian_Schwarz_split.pdf` (the assessed lit review; §2 =
   the five research objectives) and `../../MSc Transport - Research Project Guidance
   2025-2026.pdf` (deadlines/rubric: the thesis planner's first read).
7. Figures: `scratch/hybrid_geometry.png`, `scratch/chokepoints.png`, `assets/kaliningrad_*.png`,
   `scratch/dynassign_demo.gif`; probe scripts in `scratch/` are the reproducibility record.
8. Code (in this order): `src/env/smdp_wrapper.py` + `src/env/graph_env.py` (physics, SMDP
   events, antagonist reach/budget, hybrid state machine), `src/agents/networks.py`
   (featurization: 13 node / 4 edge dims incl. goal + motion columns; width-slicing back-compat),
   `src/agents/sac.py` (SAC math; `infer_node_in_dim`/`infer_edge_in_dim`; `_clip_x`/`_clip_ea`),
   `src/agents/sacred_atla.py` (trainer modes: atla · vanilla · antagonist_only ·
   scripted_adversary; `--update-every`), `src/baselines/greedy_dispatch.py` + `attackers.py`
   (random/targeted/pathrand/mask-first "gateway"), `scripts/train_sacred.py` (all flags),
   `scripts/evaluate_portfolio.py` (the robustness harness: paired W/D/dD, `--select-best
   --select-attacker`, held-out seed bases 10_000_019/20_000_019), `scripts/run_generation.py`
   (recipes + ledger discipline).

## 3. Evaluation vocabulary (fixed: use exactly this)

`W(arm, attack)` = mean total delivery latency (total_wait), lower better. `D(arm, a) = W(a) −
W(none)`, paired per instance. `dD = D(vanilla, a) − D(other, a)`; positive = the other arm is
more robust. Decision metrics are fixed in the ledger BEFORE looking; ≥3 seeds; paired instances;
CIs always; competence gate before interpreting any robustness comparison.

## 4. State of the machinery

- **Suite:** 83 tests green on the frozen campaign record; **131 green on `gen08-interdiction`**
  (`PYTHONPATH=. pytest tests/`: run after touching agents/env, paste raw output). All five
  problem rungs runnable: `--problem {osm,stage0,assign,dynassign,hybrid}`.
- **Selected checkpoints (gen06):** vanilla ep750/ep100/ep100, scripted ep450/ep200/ep600 under
  `models/runs/gen06_dynassign_matrix/*/snapshots/`; BR actors under `.../br_*_s0_seed0/`.
  gen05's analogues under `models/runs/gen05_hybrid_matrix/`.
- **Back-compat:** checkpoints of any feature-width era stay evaluable: agents slice features to
  their trained width; loaders infer widths from the checkpoint (`infer_*_in_dim`).
- **Hardware/ops:** M4 CPU-only (MPS 2.4–4× slower, settled), 4 torch threads solo / 3×3
  parallel; ~18–55 s/ep depending on rung; eval is cheap (~0.2–0.6 s/ep: don't over-estimate).
  Long jobs: `nohup … & disown` in their own session (harness-managed background tasks got reaped
  once and killed the children: documented in the gen05 ledger recovery note). Pause/resume:
  `pkill -STOP/-CONT -f train_sacred.py`. Kilian sometimes forbids scheduled wakeups: ask/obey.
- **Never** train without a ledger (pre-registered metric + pinned SHA), never compare across git
  states, never argmax-eval a max-entropy policy, gate multi-hour runs on cheap probes.

## 5. Open decisions & outlook (Kilian owns all of these)

1. **Freeze-and-write vs the option-(b) stretch.** Recommendation on record: freeze on gen06
   (defensible, complete); option (b): make the hybrid learnable (tighter corridor slack vs the
   slack-1.4× detour tension, γ↑, ERB warm-start from greedy demos = Obj-3 material): is now a
   test of the four preconditions and belongs in future work unless Kilian wants one more swing
   before ~Jul 16–18 (the campaign freeze target).
2. **Supervisor conversation (the old "D4").** Agenda: the gen06 finding + framing ("when and why
   adversarial VRP training fails, with conditions for success"), and descoping of ERB
   (inconclusive n=1), SBO (untouched), rolling-ALNS baseline, and ZST (options: a small transfer
   test *of the diagnosis* on a held-out geometry, or descope).
3. **Thesis writing.** A separate planner instance is briefed at
   `/Users/kilian/Kilian/ICL/Thesis/thesis/THESIS_PLANNER_HANDOFF.md` (launch = open Fable in
   `thesis/`, say "read THESIS_PLANNER_HANDOFF.md and begin"). That repo (Overleaf-synced) is
   where all report writing happens; this code repo is its read-only evidence base. Figure/
   experiment requests flow back here as a precise list.
4. **Back-pocket register** (recorded options, not scheduled): ATLA rider arm in a
   competent-protagonist matrix (motivated by gen05's +1667 finding); gen04b antagonist
   entropy-target re-gate (~2 h, tests the pinning hypothesis directly); option-(b) as above.
5. **Deferred chores:** `src/env/` vs `src/envs/` merge (mechanical, post-freeze, suite-guarded -
   TASK.md banner TODO 2); visualiser dims touch-up (`spar_visual*.py` builds 11-dim agents) if
   thesis figures need old-checkpoint rendering.

## 6. How to work with Kilian

Single `&&`-chained shell commands; his Mac never sleeps; he pauses runs for heat/noise; he
decides CPU spend and design changes: always present options with a recommendation and wait;
report honestly including self-corrections (this project retracted a headline claim once and is
stronger for it); ask questions when direction is genuinely his to choose, otherwise act.
Persistent memory for this repo path exists (`~/.claude/projects/-Users-kilian-Kilian-ICL-Thesis-
code-sacred/memory/`): read `MEMORY.md` there at session start; keep it and
`SACRED_PROGRESS.md` current as work proceeds.
