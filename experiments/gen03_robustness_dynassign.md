# Generation: gen03_robustness_dynassign (Phase 1 — protocol pilot)

- **git SHA:** `c361b4c` (Phase-1 infra commit; runs must execute on this state or a descendant that leaves `src/` + eval untouched)
- **date opened:** 2026-07-02
- **status:** LEDGER OPEN — training not yet launched (awaiting Kilian's go)

## Question (fixed before looking)

**Does adversarial (ATLA) training buy measurable robustness to held-out attacks, relative to a
non-adversarially trained but otherwise identical SAC dispatcher?** (The reframed headline —
CRITIQUE.md §6, D1 accepted 2026-07-02. This is the *pilot*: it shakes down the portfolio
protocol on the dynassign rung and reuses the sunk gen02 SACRED runs. The thesis verdict comes
from the Phase-3 hybrid matrix.)

## Arms

| arm | training | source |
|---|---|---|
| `sacred` | ATLA co-evolution (existing) | `gen02_dynassign` seeds 0,1 — per-phase snapshots on disk |
| `vanilla` | `--vanilla` (protagonist trains every episode, antagonist inert; env/reward/nets/hparams identical) | **to train**: 3 seeds × 800 ep, λ=0.06 |
| `greedy` | none (reference line only — no win/lose framing) | `greedy_insertion_policy` |

Note the acknowledged asymmetries, both conservative *against* the robustness claim:
vanilla gets ~2× the protagonist gradient updates per episode (no antagonist phases) and 3 seeds
vs sacred's 2 (gen02 was a 2-seed pilot).

## Attack portfolio (evaluation only — none seen verbatim in training)

| attacker | kind | role |
|---|---|---|
| `none` | — | clean baseline for D |
| `random` | scripted, seeded/instance | undirected-disruption floor |
| `targeted` | scripted (block first blockable edge ahead of the truck nearest its goal) | **VALIDATION attacker** (checkpoint selection) |
| `br_sacred` | learned: `--train-antagonist-only` vs the selected sacred checkpoint (~300 ep) | held-out **TEST** |
| `br_vanilla` | learned: same vs the selected vanilla checkpoint | held-out **TEST** |

## Decision metric (PRE-REGISTERED)

For each arm and attack: `W` = mean total_wait over paired instances; `D(arm, a) = W(a) − W(none)`
paired per instance.

> **Primary:** `dD = D(vanilla, br_vanilla) − D(sacred, br_sacred)` — each policy against **its
> own best-response attacker** (the worst-case-fair comparison; kills the old off-target-adversary
> bias). Pilot success = `dD > 0` with the paired 95% CI excluding 0, in the same direction for
> both sacred seeds.

Secondary (reported, not gating): `dD` under `targeted` and `random`; the cross-BR 2×2
(generalization of attacks across policies); clean premium `W(sacred, none) − W(vanilla, none)`
(want ≲ +15%); greedy's row for context.

Protocol constants: **test instances** = demand seeds 10_000_019…+29 (N=30, paired across all
cells; held out from every training stream — training uses seed·100003+episode);
**validation instances** = 20_000_019…+7 (N=8, selection only); protagonists act
**stochastically** (per-episode crc32 seeding); learned attackers act deterministically;
checkpoint selection per arm = min mean attacked-wait under `targeted` on validation instances
(`evaluate_portfolio.py --select-best`), never on the test attackers/instances.

## Commands

```bash
# 1. vanilla control (3 seeds, parallel, ~overnight; 800 ep matches gen02's budget)
PYTHONPATH=. python scripts/run_generation.py --group gen03_robustness_dynassign --configs vanilla --seeds 0,1,2 --episodes 800 --threads 3 --max-concurrent 3
#    (or per-seed: PYTHONPATH=. python scripts/train_sacred.py --problem dynassign --vanilla --episodes 800 --switch-every 50 --batch-size 32 --eval-every 0 --seed <k> --group gen03_robustness_dynassign --tag vanilla --threads 3)

# 2. checkpoint selection (per arm+seed, validation attacker)
PYTHONPATH=. python scripts/evaluate_portfolio.py --problem dynassign --select-best models/runs/<run> --instances 8

# 3. best-response attackers (one per selected checkpoint, ~300 ep each)
PYTHONPATH=. python scripts/train_sacred.py --problem dynassign --train-antagonist-only --protagonist-snapshot <selected.pt> --episodes 300 --switch-every 50 --eval-every 0 --seed 0 --group gen03_robustness_dynassign --tag br_<arm>_seed<k>

# 4. the portfolio (per seed-pairing; ~30 paired instances)
PYTHONPATH=. python scripts/evaluate_portfolio.py --problem dynassign \
  --policy sacred=<selected sacred ckpt> --policy vanilla=<selected vanilla ckpt> \
  --br sacred=<br_sacred actor> --br vanilla=<br_vanilla actor> \
  --instances 30 --out experiments/gen03_portfolio_seed<k>.json
```

## Result (2026-07-03) — PRIMARY METRIC: NULL. Diagnosis: the learned adversary does not learn to attack.

**Pre-registered primary — FAILED (wrong sign, not significant):**
`dD = D(vanilla, br_own) − D(sacred, br_own)`: pair0 **−291 ± 500**, pair1 **−255 ± 295** (95% CI,
n=30 paired instances each). Success required dD > 0 with CI excluding 0 in both pairs. Common
attacks are a wash too (targeted: −96/+388; random: +247/+54 — all n.s.). Pooled per-seed
D(br_own): sacred {1374, 884} vs vanilla {1083, 629, 1865} — indistinguishable.

**The mechanism (the pilot's real finding).** Attacker hierarchy measured on the same paired
instances, D = degradation vs clean:

| attacker | D on greedy | D on learned arms |
|---|---|---|
| scripted `targeted` (40-line heuristic: block ahead of the nearest-to-goal truck) | **+4921 ± 393 (~+79%)** | +5434…+5822 |
| `random` (uniform maskable edge) | +1718 ± 385 | +1770…+2108 |
| **learned best-response (300 dedicated ep each)** | +927…+1276 | **+607…+1865** |

The learned attackers are **weaker than random blocking** and **3–6× weaker than the scripted
heuristic**. During all five BR trainings the antagonist's true episode reward *fell* slightly
(~9.0k→8.4k) while its Q tripled (35→115) — the gen02 "Q runaway" is critic over-estimation, not
attack skill. Root cause consistent with CRITIQUE.md §2(i): the antagonist cannot see truck motion
(mid-edge trucks are invisible; the exact state the targeted heuristic exploits), and its reward
is dominated by the uncontrollable queue baseline (its marginal effect ~1–2k of ~8–9k per episode).

**Reading:** ATLA co-evolution trained SACRED against an incompetent sparring partner, so there
was no robustness to learn (and no clean-performance cost either: W(none) sacred ≈ vanilla ≈
6.62–6.68k vs greedy 6.20k, the familiar ~+7%). The strong scripted attack proves a large, real
attack surface exists in this arena — the learned adversary just cannot find it. **The binding
constraint on the whole SACRED thesis is adversary competence, not protagonist learning.**
Protocol itself validated: paired CIs ±300–500 (vs ±1000 single-instance before), selection found
real mid-training checkpoints, attacks held out as designed.

**Consequences for Phase 3 (hybrid matrix):** (1) fix antagonist motion observability (edge
occupancy) and GATE on "BR beats random" before spending on co-evolution; (2) proposed third
training arm `scripted-adversarial` (train vs the targeted heuristic — strong, stationary) so the
headline claim is testable even if co-evolution stays weak — D3 amendment, Kilian to decide.
Artifacts: `gen03_portfolio_{pair0,pair1,v2}.json`, `scratch/gen03_aggregate.py`, BR runs under
`models/runs/gen03_robustness_dynassign/br_*` (local).

## Launch record (2026-07-03 00:29)

- **git SHA:** `de5ff7d`
- **configs:** vanilla  **seeds:** [0, 1, 2]
- **common args:** `--episodes 800 --switch-every 50 --batch-size 32 --hidden-dim 64 --device cpu --eval-every 50 --group gen03_robustness_dynassign --threads 3`
