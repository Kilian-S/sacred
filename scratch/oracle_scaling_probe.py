"""Oracle-scaling probe (gen09, ORACLE-ONLY, NO TRAINING): the "why not just solve the LP" figure.

Finds the size at which the EXACT minimax equilibrium (the naive full-LP oracle `solve_multiconvoy`)
becomes infeasible, while a deep-RL dispatcher (SACRED) stays ~linear. Sweeps SHARED-EDGE instances
(the headline regime: OD 62-97, soft band 0.15-0.95, absolute vulnerability norm) over increasing:
  * N  = convoys           -> #occupancies = C(N+R-1, R-1)  (the defender pure-strategy count)
  * R  = routes (k_extra)  -> more occupancies AND more candidate edges E
  * K  = interdiction assets -> #interdiction-sets = C(E, K)  (the attacker pure-strategy count)
For each size it records wall-clock time + peak memory of `solve_multiconvoy` (which builds the
[#occ x #iset] objective matrix and solves two LPs), the combinatorics alongside it so the blow-up is
visible, and runs the ACTUAL solve until a time/RAM budget wall, then PROJECTS beyond it. A projected
SACRED cost line (measured headline per-sortie x N forward passes x sorties, linear) is overlaid; the
crossover = the size beyond which the oracle is infeasible but projected SACRED is not.

TWO HONESTY NOTES (also in the ledger):
  (1) This is the NAIVE FULL-LP oracle (enumerate every occupancy x every K-subset). A
      column-generation / DOUBLE-ORACLE solver would scale much better, so the crossover is against
      NAIVE ENUMERATION. The airtight scaling claim pairs this with ZERO-SHOT TRANSFER (ZST): there
      NO oracle competes at all, because an exact solver must RE-SOLVE from scratch for every new
      instance whereas a trained policy transfers with a forward pass.
  (2) The projected SACRED line is an ESTIMATE (linear extrapolation of one measured N=3 run); the
      actual scaled run validates it.

Run: PYTHONPATH=. .venv/bin/python scratch/oracle_scaling_probe.py
"""
from __future__ import annotations

import json
import math
import resource
import sys
import time

import networkx as nx
import numpy as np

from src.baselines.interdiction_oracle import (
    build_interdiction_game, build_route_set, edges_of_route,
    length_band_vulnerability, survival_intercept_fn)
from src.baselines.multiconvoy_oracle import solve_multiconvoy
from src.envs.multiconvoy_interdiction import _DEFAULT_EDGES, _DEFAULT_NODES, _DEFAULT_TASKS
from src.utils.graph_utils import load_osm_graph_and_demands

OD = ("62", "97")               # the headline OD (shared-edge, asymmetric)
BAND = (0.15, 0.95)             # soft interception band (headline)
# 2026-07-10 UPDATE: objective_matrix's mission/linear closed forms are now VECTORISED (one matmul
# instead of a per-entry Poisson-binomial convolution), which moves the naive-oracle wall from the
# matrix BUILD to the LP SOLVE + RAM. Caps raised accordingly so the K=3/K=4 points are MEASURED,
# not projected; the pre-vectorisation output is preserved in the gen09 ledger for the record.
TIME_WALL_S = 300.0             # stop RUNNING actual solves once one exceeds this (record the wall)
ENTRIES_RUN_CAP = 600_000_000   # obj-matrix entries above which we DON'T attempt the solve (RAM guard) -> project
MEM_BUDGET_GB = 8.0             # the "infeasible" RAM bar for the crossover
SORTIES = 1200                  # the headline sortie budget (for the SACRED projection)
N_BASE = 3                      # the headline fleet size
# measured headline per-sortie wall-time (N=3), 3-parallel-contended (conservative; a single run is
# ~0.37 s/sortie per the M3 smoke, ~2.7x faster). SACRED is ~linear in N (N convoys routed per sortie).
PER_SORTIE_BASE_S = 0.99


def rss_peak_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e6 if sys.platform == "darwin" else r / 1e3   # macOS bytes, Linux KB


def comb(n: int, k: int) -> int:
    return math.comb(n, k) if 0 <= k <= n else 0


# --- graph + route-set cache (route set depends only on k_extra) ---
_nodes, _edges = load_osm_graph_and_demands(_DEFAULT_NODES, _DEFAULT_EDGES, _DEFAULT_TASKS)
G = nx.Graph()
for u, v, d in _edges:
    G.add_edge(str(u), str(v), w=float(d.get("distance", 1.0)))
ALL_EDGES = list(G.edges())
_ROUTE_CACHE: dict[int, tuple[list, list]] = {}


def route_set(k_extra: int) -> tuple[list, list]:
    if k_extra not in _ROUTE_CACHE:
        routes = build_route_set(G, OD[0], OD[1], k_extra, "w")
        cand = sorted(set().union(*(edges_of_route(r) for r in routes)), key=repr)
        _ROUTE_CACHE[k_extra] = (routes, cand)
    return _ROUTE_CACHE[k_extra]


def sacred_projection() -> tuple[float, str]:
    """Per-sortie base (s) from the locked headline run if present, else the constant fallback."""
    try:
        d = json.load(open("models/runs/gen09_multiconvoy/headline_seed0.json"))["fleet_route"]
        h = d["history"]
        sorts = [x[0] for x in h]
        interval = sorts[1] - sorts[0] if len(sorts) > 1 else 100
        per = float(np.mean([x[9] for x in h])) / interval
        return per, f"measured from gen09-HEADLINE seed0 ({per:.3f} s/sortie, N=3, 3-parallel)"
    except Exception:
        return PER_SORTIE_BASE_S, f"fallback constant {PER_SORTIE_BASE_S} s/sortie (N=3)"


def sacred_time_s(N: int, per_sortie_base: float) -> float:
    """~linear projection: N forward passes/sortie x SORTIES; scale the N=3 base by N/N_BASE."""
    return per_sortie_base * (N / N_BASE) * SORTIES


# --- the sweep grid (dedup by (N, k_extra, K)) ---
GRID: list[tuple[int, int, int]] = []
GRID += [(N, 8, 1) for N in range(2, 11)]              # N blow-up (fixed R~12, K=1)
GRID += [(3, kx, 1) for kx in (8, 16, 24, 32, 48)]     # R blow-up (fixed N=3, K=1)
GRID += [(3, 8, K) for K in (1, 2, 3, 4)]              # K blow-up (fixed N=3, R~12)
GRID += [(N, 8, 2) for N in (4, 5, 6, 7)]              # combined N x K=2 (toward the wall)
GRID += [(N, 8, 3) for N in (4, 5, 6)]                 # combined N x K=3 (past the wall)
GRID = sorted(set(GRID))


def combinatorics(N: int, k_extra: int, K: int) -> dict:
    routes, cand = route_set(k_extra)
    R, E = len(routes), len(cand)
    n_occ = comb(N + R - 1, R - 1)
    n_iset = comb(E, K)
    entries = n_occ * n_iset
    return {"N": N, "k_extra": k_extra, "K": K, "R": R, "E": E,
            "n_occ": n_occ, "n_iset": n_iset, "entries": entries,
            "obj_MB": entries * 8 / 1e6}


def main() -> None:
    per_sortie, per_src = sacred_projection()
    rows = sorted((combinatorics(*g) for g in GRID), key=lambda r: r["entries"])

    print("=" * 118)
    print("ORACLE-SCALING PROBE (naive full-LP equilibrium) vs projected SACRED (linear).  OD 62-97, "
          "shared-edge, soft band 0.15-0.95.")
    print(f"SACRED projection base: {per_src}; SORTIES={SORTIES}; ~linear in N (N convoys/sortie).")
    print(f"Budgets: RUN actual solve up to {ENTRIES_RUN_CAP/1e6:.0f}M obj entries AND {TIME_WALL_S:.0f}s; "
          f"'infeasible' RAM bar {MEM_BUDGET_GB} GB.")
    print("=" * 118)
    hdr = (f"{'N':>2} {'R':>3} {'E':>3} {'K':>2} | {'#occ=C(N+R-1,R-1)':>18} {'#iset=C(E,K)':>12} "
           f"{'obj entries':>12} {'obj MB':>8} | {'oracle time':>12} {'peakMB':>7} | {'SACRED proj':>11} | status")
    print(hdr); print("-" * len(hdr))

    wall_hit = False
    measured = []   # (entries, oracle_s) for the fit
    results = []
    for r in rows:
        N, R, E, K = r["N"], r["R"], r["E"], r["K"]
        sac_s = sacred_time_s(N, per_sortie)
        infeasible_ram = r["obj_MB"] / 1024.0 > MEM_BUDGET_GB
        run = (not wall_hit) and (r["entries"] <= ENTRIES_RUN_CAP) and (not infeasible_ram)
        oracle_s, peak_mb, status = None, None, ""
        if run:
            rss0 = rss_peak_mb()
            t0 = time.perf_counter()
            solve_multiconvoy(build_game_cached(r["k_extra"], K), N, "mission")
            oracle_s = time.perf_counter() - t0
            peak_mb = rss_peak_mb() - rss0
            measured.append((r["entries"], oracle_s))
            if oracle_s > TIME_WALL_S:
                status = "*** WALL (measured > budget) ***"; wall_hit = True
            else:
                status = "solved"
        else:
            status = ("INFEASIBLE (RAM)" if infeasible_ram
                      else "INFEASIBLE (projected; > run cap / past wall)")
        r.update({"oracle_s": oracle_s, "peak_mb": peak_mb, "sacred_s": sac_s, "status": status})
        results.append(r)
        os = f"{oracle_s:10.2f}s" if oracle_s is not None else f"{'--':>11}"
        pm = f"{peak_mb:6.0f}" if peak_mb is not None else f"{'--':>6}"
        print(f"{N:>2} {R:>3} {E:>3} {K:>2} | {r['n_occ']:>18,} {r['n_iset']:>12,} "
              f"{r['entries']:>12,} {r['obj_MB']:>8.1f} | {os:>12} {pm:>7} | {sac_s:>9.0f}s | {status}")

    # --- projected oracle time beyond the wall (linear-in-entries fit; a LOWER bound, the LP is superlinear) ---
    if len(measured) >= 2:
        ent = np.array([m[0] for m in measured]); tim = np.array([m[1] for m in measured])
        a = float(np.sum(ent * tim) / np.sum(ent * ent))   # time ~ a * entries (through origin)
        print("-" * len(hdr))
        print(f"oracle-time fit (lower bound; LP is super-linear): ~{a*1e6:.2f} s per 1M obj entries")
        for r in results:
            if r["oracle_s"] is None:
                proj = a * r["entries"]
                print(f"  projected oracle time  N={r['N']} R={r['R']} K={r['K']}: "
                      f"~{proj:,.0f}s (~{proj/3600:.1f}h)  [obj {r['obj_MB']:,.0f} MB]  "
                      f"vs SACRED proj {r['sacred_s']:.0f}s")

    # attach a projected oracle time to every row (measured where available, else the fit)
    a = (float(np.sum(np.array([m[0] for m in measured]) * np.array([m[1] for m in measured])) /
              np.sum(np.array([m[0] for m in measured]) ** 2)) if len(measured) >= 2 else 0.0)
    for r in results:
        r["oracle_est_s"] = r["oracle_s"] if r["oracle_s"] is not None else a * r["entries"]

    # --- CROSSOVER: defined against the SACRED cost, NOT an arbitrary time budget. This is the real
    # "why not just solve the LP" answer: past here, solving the exact LP ONCE already costs more than
    # training SACRED (which is then FREE per new instance via ZST). ---
    print("=" * 118)
    ordered = sorted(results, key=lambda r: r["entries"])
    cross = next((r for r in ordered if r["oracle_est_s"] > r["sacred_s"]), None)
    last_feasible = max((r for r in results if r["oracle_s"] is not None), key=lambda r: r["entries"])
    print(f"MEASURED WALL: the naive oracle is actually solved up to ~{last_feasible['entries']:,} obj "
          f"entries (N={last_feasible['N']}, R={last_feasible['R']}, K={last_feasible['K']}: "
          f"{last_feasible['oracle_s']:.0f}s, {last_feasible['peak_mb']:.0f} MB); larger sizes are projected.")
    if cross:
        print(f"CROSSOVER (oracle cost > one-time SACRED cost): at ~{cross['entries']:,} obj entries "
              f"(N={cross['N']}, R={cross['R']}, K={cross['K']}): oracle ~{cross['oracle_est_s']:,.0f}s "
              f"vs SACRED ~{cross['sacred_s']:.0f}s. Past here the exact LP costs MORE to solve ONCE than "
              f"training SACRED; and SACRED then transfers per-instance for FREE (ZST) while the oracle "
              f"re-solves from scratch each time.")
    # the sharpest axis is K: the attacker interdiction-set count C(E,K) explodes.
    E8 = len(route_set(8)[1])
    print(f"SHARPEST BLOW-UP is K (attacker sets C(E={E8},K)): "
          f"K=1->{comb(E8,1):,}, K=2->{comb(E8,2):,}, K=3->{comb(E8,3):,}, K=4->{comb(E8,4):,} sets; "
          f"and N (defender occupancies C(N+R-1,R-1)); the trained-SACRED forward-pass cost is "
          f"independent of both R and K (it routes N convoys over the graph).")

    # --- FINDING: the crossover axis is the INTERDICTION BUDGET K (attacker sets C(E,K)), not the
    # fleet. Fleet size alone keeps the oracle feasible to N=10 (~25 min); the wall is K>=3. ---
    k1_max = max((r for r in results if r["K"] == 1), key=lambda r: r["oracle_est_s"])
    print(f"NOTE: increasing FLEET SIZE alone does NOT break the naive oracle here - even N=10, K=1 is "
          f"~{k1_max['oracle_est_s']:.0f}s ({k1_max['obj_MB']:.0f} MB, occupancies grow only polynomially). "
          f"The wall is the INTERDICTION BUDGET K (and R x K jointly).")

    # candidates: past-crossover, K>=3 (the wall axis) or RAM-heavy; span smallest / middle / heaviest.
    cands = [r for r in ordered if r["oracle_s"] is None and r["oracle_est_s"] > r["sacred_s"]
             and (r["K"] >= 3 or r["obj_MB"] / 1024 > MEM_BUDGET_GB)]
    picks = []
    if cands:
        picks = [cands[0], cands[len(cands) // 2], max(cands, key=lambda r: r["obj_MB"])]
        seen = set(); picks = [p for p in picks if id(p) not in seen and not seen.add(id(p))]
    print("CANDIDATE SCALED INSTANCES for the follow-up SCALED RUN (oracle genuinely infeasible along "
          "the K axis; SACRED feasible; smallest / mid / heaviest):")
    for r in (picks or cands[:3]):
        why = (f"~{r['oracle_est_s']/3600:.1f} h" + (f", {r['obj_MB']/1024:.1f} GB obj" if r["obj_MB"]/1024 > 1 else "")
               if r["oracle_est_s"] > 3600 else f"~{r['oracle_est_s']:,.0f}s")
        print(f"  - N={r['N']}, k_extra={r['k_extra']} (R={r['R']}, E={r['E']}), K={r['K']}: "
              f"#occ {r['n_occ']:,} x #iset {r['n_iset']:,} = {r['entries']:,} entries "
              f"({r['obj_MB']:,.0f} MB obj) -> oracle INFEASIBLE ({why}) vs SACRED proj "
              f"{r['sacred_s']:.0f}s (~{r['sacred_s']/60:.0f} min).")
    print("=" * 118)
    print("HONESTY: (1) NAIVE full-LP oracle -> a double-oracle / column-generation solver scales "
          "better; the crossover is vs naive enumeration, and the AIRTIGHT scaling claim pairs this "
          "with ZST (no oracle competes: it must re-solve each instance). (2) The SACRED line is a "
          "linear ESTIMATE; the actual scaled run validates it.")


_GAME_CACHE: dict[tuple[int, int], object] = {}


def build_game_cached(k_extra: int, K: int):
    if (k_extra, K) not in _GAME_CACHE:
        routes, cand = route_set(k_extra)
        vuln = length_band_vulnerability(G, cand, band=BAND, weight="w", norm_edges=ALL_EDGES)
        _GAME_CACHE[(k_extra, K)] = build_interdiction_game(
            G, OD[0], OD[1], K, k_extra=k_extra, weight="w", intercept_fn=survival_intercept_fn(vuln))
    return _GAME_CACHE[(k_extra, K)]


if __name__ == "__main__":
    main()
