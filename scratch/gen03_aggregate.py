"""Aggregate the gen03 portfolio JSONs into the pre-registered decision metrics.

Reads experiments/gen03_portfolio_pair0.json / pair1.json / v2.json (produced by
scripts/evaluate_portfolio.py) and computes, per the gen03 ledger:

  PRIMARY   dD = D(vanilla, br_vanilla) - D(sacred, br_sacred), paired per instance within each
            pairing JSON (each arm vs its OWN best-response attacker). dD > 0 => the vanilla
            control degrades more, i.e. adversarial training bought robustness.
  SECONDARY dD under the common attacks (targeted / random); the cross-BR 2x2; the clean premium
            W(sacred,none) - W(vanilla,none); greedy reference rows.

    PYTHONPATH=. python scratch/gen03_aggregate.py
"""

from __future__ import annotations

import json
import math
import statistics


def mean_ci(xs: list[float]) -> tuple[float, float]:
    m = statistics.mean(xs)
    if len(xs) < 2:
        return m, float("nan")
    sem = statistics.stdev(xs) / math.sqrt(len(xs))
    return m, 1.96 * sem


def paired_diff(a: list[float], b: list[float]) -> list[float]:
    return [y - x for x, y in zip(a, b)]


def d_of(res: dict, arm: str, attack: str) -> list[float]:
    """Per-instance degradation D(arm, attack) = W(attack) - W(none)."""
    return paired_diff(res[arm]["none"], res[arm][attack])


def load(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)["results"]


def main() -> None:
    pair0 = load("experiments/gen03_portfolio_pair0.json")
    pair1 = load("experiments/gen03_portfolio_pair1.json")
    v2 = load("experiments/gen03_portfolio_v2.json")

    print("=" * 76)
    print("gen03 — PRE-REGISTERED PRIMARY: dD = D(vanilla, br_own) - D(sacred, br_own)")
    print("=" * 76)
    primary_by_pair = {}
    for name, res in (("pair0", pair0), ("pair1", pair1)):
        d_sac = d_of(res, "sacred", "br_sacred")
        d_van = d_of(res, "vanilla", "br_vanilla")
        diffs = paired_diff(d_sac, d_van)  # d_van - d_sac
        m, ci = mean_ci(diffs)
        primary_by_pair[name] = (m, ci)
        verdict = "SACRED more robust" if m > 0 else "vanilla more robust"
        sig = "significant" if not math.isnan(ci) and abs(m) > ci else "NOT significant"
        print(f"  {name}: dD = {m:+8.0f} ± {ci:6.0f} (95% CI, n={len(diffs)})  [{verdict}; {sig}]")
        print(f"         D(sacred, br_sacred) = {mean_ci(d_sac)[0]:8.0f}   "
              f"D(vanilla, br_vanilla) = {mean_ci(d_van)[0]:8.0f}")

    print("\nSECONDARY — common attacks (same attacker for both arms, paired):")
    for attack in ("targeted", "random"):
        for name, res in (("pair0", pair0), ("pair1", pair1)):
            diffs = paired_diff(d_of(res, "sacred", attack), d_of(res, "vanilla", attack))
            m, ci = mean_ci(diffs)
            print(f"  {name} {attack:>9}: dD = {m:+8.0f} ± {ci:6.0f}")

    print("\nSECONDARY — cross-BR 2x2 (attack generalization), D(arm, attack):")
    for name, res in (("pair0", pair0), ("pair1", pair1)):
        for arm in ("sacred", "vanilla"):
            row = []
            for attack in ("br_sacred", "br_vanilla"):
                m, ci = mean_ci(d_of(res, arm, attack))
                row.append(f"{attack}: {m:8.0f} ± {ci:5.0f}")
            print(f"  {name} {arm:>8} | " + " | ".join(row))

    print("\nSECONDARY — clean premium W(arm, none) and greedy reference:")
    for name, res in (("pair0", pair0), ("pair1", pair1), ("v2", v2)):
        for arm in res:
            m, ci = mean_ci(res[arm]["none"])
            print(f"  {name} {arm:>8} W(none) = {m:8.0f} ± {ci:5.0f}")

    print("\nvanilla_seed2 (no sacred partner) — D(vanilla, br_own) for the cross-seed picture:")
    d_v2 = d_of(v2, "vanilla", "br_vanilla")
    m, ci = mean_ci(d_v2)
    print(f"  v2: D(vanilla, br_vanilla) = {m:8.0f} ± {ci:6.0f}")

    print("\nGreedy under the learned attacks (reference):")
    for name, res in (("pair0", pair0), ("pair1", pair1)):
        for attack in ("br_sacred", "br_vanilla", "targeted"):
            if attack in res.get("greedy", {}):
                m, ci = mean_ci(d_of(res, "greedy", attack))
                print(f"  {name} greedy D({attack}) = {m:8.0f} ± {ci:5.0f}")


if __name__ == "__main__":
    main()
