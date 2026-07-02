# Generation: gen01_erb_ablation

- **git SHA:** `dd96228` (runs are only comparable within this code state)
- **date:** 2026-06-28 22:50
- **configs:** assign_erb, assign_noerb
- **seeds:** [0, 1, 2]
- **common args:** `--episodes 1000 --switch-every 50 --batch-size 32 --hidden-dim 64 --device cpu --eval-every 50 --group gen01_erb_ablation --threads 3`

## Question

**Does ERB-seeding (greedy no-attack demos) help the assignment probe, and is the win real (not seed luck)?**
The single-run `assign_probe_claimfix` milestone showed RL beats greedy-insertion under attack (final gap −56, 6.5%) but *lost statically* (gap_noatk ≈ +68) and oscillated badly (6/20 wins). This generation runs **3 seeds each** of:
- `assign_erb` — protagonist replay buffer seeded with 1600 greedy no-attack demos (`data/erb_assign.pt`).
- `assign_noerb` — identical config, cold buffer (the control).

**Success = ERB drives `gap_noatk` → ~0 (matches greedy statically) AND keeps `gap_atk` negative (retains the under-attack win), with lower seed-to-seed variance than the control.** Failure mode to catch: `gap_atk` regresses toward 0 (greedy-imitation washed out the robustness). Decision metric (fixed in advance): mean ± std of `gap_atk` and `gap_noatk` over the last 4 eval points, across the 3 seeds.

## Result (PARTIAL — seed0 only; generation paused for heat/noise)

**`assign_erb_seed0` (n=1, inconclusive but discouraging):**
- `gap_noatk` stayed **~+50** (start +54) — ERB did **not** fix the static partition (its stated goal); barely better than no-ERB's +68. This signal is the least seed-dependent → the concerning one.
- `gap_atk` very noisy (8/20 wins, best −96 mid-run at ep 550–600) but **collapsed late**: ep 950 (post-protag) +232, ep 1000 +282. `Q_Spread` 7.1→1.9, entropy 0.5→0.7, antagonist Q 19→56 → **antagonist ran away in the late arms race and wrecked the protagonist.**
- **Two findings:** (1) ERB-as-built isn't earning its keep (static gap unfixed, no stabilisation); (2) **late co-evolution instability** (antagonist runaway) is the bigger, config-agnostic problem and likely the source of our pervasive noise. The final-checkpoint metric is also misleading under co-evolution (penalises the protagonist for whichever phase training ends on; this run was clearly capable mid-training at −96).

**Status:** seed1/seed2 paused at ep 50 (resumable); no-ERB seeds not started. Do NOT conclude on ERB from n=1. Next: either complete the seeds (quiet/sequential) for a real mean±std, OR pivot to the instability (best-checkpoint selection, rein in antagonist, earlier stop).
