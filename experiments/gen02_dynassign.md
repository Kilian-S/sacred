# Generation: gen02_dynassign

- **git SHA:** `dd96228` (runs are only comparable within this code state)
- **date:** 2026-06-29 16:38
- **configs:** dynassign
- **seeds:** [0, 1]
- **common args:** `--episodes 800 --switch-every 50 --batch-size 32 --hidden-dim 64 --device cpu --eval-every 50 --group gen02_dynassign --threads 4`

## Question

**Does adversarially-trained RL beat greedy-insertion under attack in the dynamic (Poisson)
multi-truck assignment regime — and is it real, not seed luck?**

Config: Stage-1.5 `dynassign`, λ=0.06 (ρ≈1, delivery ~0.71 unattacked), **full-blockage antagonist**
(`congestion_levels=(1.0,)`, 1 roadblock/decision event, `congestion_duration=120`, budget 4000 →
~30 sustained roadblocks/ep), destination-mode assignment (routing deferred), capacity-1, 800 ep,
switch-every 50, 2 seeds. Headroom gate (pre-training): urgency loses everywhere; heuristic
antagonist attack cost ~+8% at λ=0.06; clairvoyant gap mostly free-flow optimism.

**Decision metric (fixed in advance, before looking):** for each seed, select the protagonist's
**best checkpoint** via `select_best_checkpoint` (evaluated vs the **fixed final antagonist** over N
held-out Poisson demand seeds), then report **mean ± std of that best-checkpoint `gap_atk`** across
the 2 seeds. **Success = mean `gap_atk` < 0** (RL beats greedy under attack) with std not swamping
it. Also report `gap_noatk` (expect a small nominal loss — the robustness trade). Explicitly NOT
the final checkpoint (misleading under co-evolution).

Caveats: **2 seeds is a pilot** (methodology wants ≥3) — treat a positive as promising-pending-a-
third-seed. The git SHA `dd96228` is **stale**: the working tree carries the uncommitted dynassign +
full-blockage-antagonist work (incl. the two level-mapping bug fixes), so it does not identify this
code state.

## Result (2026-06-30) — RL does NOT beat greedy (near-wash)

Best-checkpoint `gap_atk` vs the fixed final antagonist (5 held-out demand seeds):
- seed0: best = **ep50 (untrained)**, gap_atk **−33 ± 1223**, gap_noatk +328
- seed1: best = ep550, gap_atk **−179 ± 909**, gap_noatk +368
- **cross-seed mean gap_atk ≈ −106**, but **±~1000 std (SEM ±450) → NOT significant**, and selection-biased (min over 15 noisy snapshots; seed0's "best" being untrained ep50 is the tell — most of seed0's snapshots lose, +200…+650).
- `gap_noatk` ≈ **+348 (~+6%)**, low-variance → RL **reliably loses statically**.

Diagnostics: Q_Spread healthy (~7, no collapse), entropy 0.56→0.47, **antagonist Q 37→116 (runaway)**;
gap_atk swings ±1000 with it. Delivery ~0.55, mean delivered latency ~145, final queue ~22 (flat).

**Verdict:** destination-mode (auto-routed) assignment is structurally too thin — same near-wash as
Stage 0 / static-3b. The machinery is sound; the missing lever is **next-hop routing**. 3rd seed not
worthwhile (noise-dominated + structural). Next: **Stage 2 hybrid (assignment + routing)**. Full
record in `CONTEXT.md` §2.
