# Generation: gen05_hybrid_matrix (Phase 3 — the headline robustness matrix)

- **git SHA:** `cd11f14` (Phase-3 build commit)
- **date opened:** 2026-07-04
- **status:** LEDGER OPEN — awaiting Kilian's launch

## Question (fixed before looking)

**Does adversarial training against a strong adversary buy robustness to held-out attacks?**
Arms: `hybrid_vanilla` (no adversary during training) vs `hybrid_scripted` (trained against the
scripted `targeted` attacker — promoted per the gen04 gate FAIL; the learned/co-evolved adversary
is parked in the back pocket along with the gen04b entropy re-gate). Arena: the FIXED hybrid rung
(assignment + next-hop routing, chokepoint geometry, route-reach, full-block, budget 1500,
max_ticks 800, `--update-every 8` — identical for both arms).

## Design

| | value | why |
|---|---|---|
| arms | `hybrid_vanilla`, `hybrid_scripted` × seeds {0,1,2} | 2×3 runs; identical env/reward/nets/hparams, only training-time adversary differs |
| episodes | 400, switch-every 25 (= snapshot cadence, 16 snapshots) | hybrid episodes carry ~5–10× more decisions than dynassign; 400 ep ≈ gen03's gradient budget at `--update-every 8` |
| budget / horizon | 1500 / 800 ticks | budget sweep (+84% on greedy, episodes end ~tick 416 attacked); 800 halves untrained-wandering cost with full headroom |
| smoke timings | vanilla ~28 s/ep, scripted ~18 s/ep (untrained; shortens as policies learn) | 6 runs ≈ 2–3 h each, 3 parallel → **~5–6 h wall** |

## Attack portfolio (eval)

| attacker | role |
|---|---|
| `none` | clean baseline for D |
| `random` | undirected floor |
| `targeted` | **in-distribution** for the scripted arm (its training attacker) — reported, NOT the primary; also the VALIDATION attacker for checkpoint selection (same selector for both arms; noted asymmetry: it is train-attack for one arm) |
| `gateway` (first-maskable-edge, route-reach) | **HELD OUT — the PRIMARY test attack.** Never used in training or selection; did +40…+184% on greedy in the budget sweep |
| `br_vanilla_s0`, `br_scripted_s0` | learned best-response rows (one per arm, seed 0 only — gen04 showed BRs ≈ random, kept as the learned-attack reference, not the primary) |

## Decision metric (PRE-REGISTERED)

W = mean total_wait over 24 paired rollout instances (static demand → instance = episode seed);
D(arm, a) = W(a) − W(none) paired per instance.

> **Primary:** `dD_gateway = D(vanilla, gateway) − D(scripted, gateway)` per seed pairing
> (v_k vs s_k), pooled across the 3 pairings. **Success = pooled dD_gateway > 0 with the paired
> 95% CI excluding 0, and ≥ 2/3 pairings individually positive.**

Secondary (reported, not gating): dD under `random` and the br rows; the `targeted` row
(in-distribution — expect the largest gap; explicitly not claimable as held-out robustness);
clean premium W(scripted, none) − W(vanilla, none) (want ≲ +15%); greedy reference rows;
Eval/* training curves. Checkpoint selection per arm: `evaluate_portfolio.py --select-best
--problem hybrid` (targeted attacker, validation rollout seeds 20_000_019+, 8 instances).

## Commands

```bash
# 1. the matrix (6 runs, 3 parallel, ~5-6 h)
PYTHONPATH=. python scripts/run_generation.py --group gen05_hybrid_matrix --configs hybrid_vanilla,hybrid_scripted --seeds 0,1,2 --episodes 400 --switch-every 25 --update-every 8 --eval-every 50 --threads 3 --max-concurrent 3

# 2. selection per run (validation attacker, 8 val instances)
PYTHONPATH=. python scripts/evaluate_portfolio.py --problem hybrid --select-best models/runs/gen05_hybrid_matrix/<run> --instances 8

# 3. two BR attackers (seed-0 selected checkpoints, ~300 ep each)
PYTHONPATH=. python scripts/train_sacred.py --problem hybrid --train-antagonist-only --protagonist-snapshot <sel> --episodes 300 --switch-every 50 --eval-every 0 --seed 0 --group gen05_hybrid_matrix --tag br_<arm>_s0

# 4. portfolio (7 arms x 6 attacks x 24 instances ~ 1000 episodes, ~15-20 min)
PYTHONPATH=. python scripts/evaluate_portfolio.py --problem hybrid \
  --policy vanilla_s0=<sel> --policy vanilla_s1=<sel> --policy vanilla_s2=<sel> \
  --policy scripted_s0=<sel> --policy scripted_s1=<sel> --policy scripted_s2=<sel> \
  --br vanilla_s0=<br actor> --br scripted_s0=<br actor> \
  --attackers none,random,targeted,gateway,br_vanilla_s0,br_scripted_s0 \
  --instances 24 --out experiments/gen05_portfolio.json
```

## Result

_(to be filled)_
