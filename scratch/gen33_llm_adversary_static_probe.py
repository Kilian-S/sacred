"""gen33_llm_adversary PHASE 0 (oracle-only, 2026-07-21; folded into
experiments/gen33_llm_adversary.md 2026-07-22): does a HETEROGENEOUS multi-system laydown
(Kilian's proposal: 2 long-range SAMs + 3 short-range MANPADS) tie the naive frontier when
reshuffled per episode but NON-reacting, and does making the short-range teams RE-CUE within the
episode open a corridor? Tests the static-vs-reactive distinction on real Kaliningrad->Gvardeysk
terrain. FINDING (binding for the LLM-adversary act): static heterogeneous placement ties the
naive frontier (0.99-1.02x, outlier 1.29x); the corridor needs the MIXED/dynamic register.

Systems (Kilian's characteristics):
  SAM  : range 2.5 km, p_max 0.90, eligible on open/field terrain (long-range, LOS).   x2
  MANPADS: range 1.2 km, p_max 0.85, eligible on any emplaceable terrain.               x3
Mission objective P(>=1 of 3 stacked drones lost); survival = product over the 5 systems.
"""
import itertools
import json
import time

import numpy as np

from src.envs.aerial_theatre_vec import (build_theatre_game, load_vec_theatre,
                                         hazard_sites, route_survival)

N, W = 3, 3
SAM = dict(r=2.5, p=0.90, terr={"open", "field"}, k=2)
MAN = dict(r=1.2, p=0.85, terr={"open", "field", "forest"}, k=3)


def logsurv(th, menu, coords, sys):
    rr = np.full(len(coords), sys["r"]); pp = np.full(len(coords), sys["p"])
    S = np.stack([route_survival(th, m, coords, rr, pp, los=True) for m in menu])
    return np.log(np.clip(S, 1e-300, 1.0))            # [R, H]


def greedy_laydown(logsam, logman, sam_ok, man_ok, weight, ksam, kman):
    """Greedy typed BR: pick ksam SAM + kman MAN sites maximising sum_r weight[r]*(1 - surv[r])."""
    R = logsam.shape[0]
    cur = np.zeros(R)                                  # accumulated log-survival
    picks = {"sam": [], "man": []}
    budget = {"sam": ksam, "man": kman}
    pools = {"sam": (logsam, sam_ok), "man": (logman, man_ok)}
    for _ in range(ksam + kman):
        best, bt, bs = -1e18, None, None
        base = (weight * (1 - np.exp(cur))).sum()
        for typ in ("sam", "man"):
            if budget[typ] <= 0:
                continue
            lg, ok = pools[typ]
            for s in ok:
                if s in picks[typ]:
                    continue
                gain = (weight * (1 - np.exp(cur + lg[:, s]))).sum() - base
                if gain > best:
                    best, bt, bs = gain, typ, s
        picks[bt].append(bs); budget[bt] -= 1
        cur = cur + pools[bt][0][:, bs]
    return picks, cur                                  # cur = laydown log-survival per route


def eval_laydown(logsam, logman, sam_ok, man_ok, lane_idx, R, states, succ, ksam, kman):
    """Return the (A) static + (B) reactive numbers for one (ksam, kman) laydown size."""
    dmg_of = lambda cur: 1.0 - np.exp(cur) ** N
    # (A) static FP
    d = np.full(R, 1.0 / R); lay = []
    for it in range(200):
        _, cur = greedy_laydown(logsam, logman, sam_ok, man_ok, d, ksam, kman)
        lay.append(cur)
        c = np.mean([dmg_of(cu) for cu in lay], axis=0)
        d = d * it / (it + 1); d[int(c.argmin())] += 1.0 / (it + 1)
    dbar = d
    def worst(dn):
        _, cu = greedy_laydown(logsam, logman, sam_ok, man_ok, dn, ksam, kman)
        return float((dn * dmg_of(cu)).sum())
    v_static = worst(dbar)
    d_uni = np.zeros(R); d_uni[lane_idx] = 1.0 / len(lane_idx)
    exp = 1 - np.exp(np.minimum(logsam, logman)).min(axis=1)
    d_inv = np.zeros(R); d_inv[lane_idx] = 1 / np.clip(exp[lane_idx], 1e-6, None); d_inv /= d_inv.sum()
    bestk = min(worst(d_uni), worst(d_inv))
    for k in (2, 3, 4):
        for T in itertools.combinations(range(R), k):
            dk = np.zeros(R); dk[list(T)] = 1.0 / k
            bestk = min(bestk, worst(dk))
    # (B) reactive: ksam SAM fixed + kman MAN re-cue to the window
    picks, _ = greedy_laydown(logsam, logman, sam_ok, man_ok, dbar, ksam, kman)
    sam_cur = logsam[:, picks["sam"]].sum(axis=1) if picks["sam"] else np.zeros(R)
    Sn = len(states); cache = {}
    def man_cur(winset):
        key = frozenset(winset)
        if key not in cache:
            wv = np.zeros(R)
            for r in winset: wv[r] += 1.0
            cur = sam_cur.copy(); chosen = []
            for _ in range(kman):
                base = (wv * (1 - np.exp(cur))).sum(); best, bs = -1, None
                for s in man_ok:
                    if s in chosen: continue
                    g = (wv * (1 - np.exp(cur + logman[:, s]))).sum() - base
                    if g > best: best, bs = g, s
                chosen.append(bs); cur = cur + logman[:, bs]
            cache[key] = cur
        return cache[key]
    stepdmg = np.stack([dmg_of(man_cur(tuple(st))) for st in states])
    V = np.zeros(Sn)
    for _ in range(3000):
        Q = stepdmg + V[succ]; Vn = 0.5 * Q.min(axis=1) + 0.5 * V; Vd = Vn - Vn.mean()
        if np.abs(Vd - V).max() < 1e-12: break
        V = Vd
    hist_opt = float((stepdmg + V[succ]).min(axis=1).mean() - 0*V.mean())
    Q = stepdmg + V[succ]; hist_opt = float((Q.min(axis=1) - V).mean())
    inwin = np.zeros((Sn, R), bool)
    for k in range(W): inwin[np.arange(Sn), states[:, k]] = True
    def stat(D):
        pi = np.full(Sn, 1.0 / Sn)
        for _ in range(500):
            flow = pi[:, None] * D; nxt = np.zeros(Sn); np.add.at(nxt, succ.ravel(), flow.ravel())
            nxt = 0.5 * nxt + 0.5 * pi
            if np.abs(nxt - pi).max() < 1e-13: break
            pi = nxt
        return float((pi[:, None] * D * stepdmg).sum())
    iid_eq = stat(np.broadcast_to(dbar, (Sn, R)))
    def anti_mat(supp):
        m0 = np.zeros(R); m0[supp] = 1.0
        M = np.broadcast_to(m0, (Sn, R)).copy(); M[inwin] = 0.0
        s = M.sum(1, keepdims=True)
        return np.where(s > 1e-9, M / np.where(s > 1e-9, s, 1), np.broadcast_to(m0 / m0.sum(), M.shape))
    anti = min(stat(anti_mat(np.where(dbar > 1e-4)[0])), stat(anti_mat(np.array(lane_idx))))
    return dict(v_static=v_static, best_naive=bestk, hist_opt=hist_opt, iid_eq=iid_eq, anti=anti)


def dmg_of(cur):
    return 1.0 - np.exp(cur) ** N                      # mission damage per route


def main():
    t0 = time.time()
    th = load_vec_theatre("data/maps/theatre_kgd_gvardeysk_vec.json")
    game, menu, coords, rr, pp, S, lane_idx = build_theatre_game(th, K=1, n_lanes=14,
                                                                 n_terrain=12, standoff_km=4.0)
    R, H = len(menu), len(coords)
    terr = [th.classify(coords[h]) for h in range(H)]
    sam_ok = [h for h in range(H) if terr[h] in SAM["terr"]]
    man_ok = [h for h in range(H) if terr[h] in MAN["terr"]]
    logsam = logsurv(th, menu, coords, SAM)
    logman = logsurv(th, menu, coords, MAN)
    print(f"[hetero] R={R} sites={H} SAM-eligible={len(sam_ok)} MAN-eligible={len(man_ok)} "
          f"| build {time.time()-t0:.0f}s", flush=True)

    # ---- (A) STATIC game: fictitious play (defender mixes routes, attacker greedy-typed BR) ----
    rng = np.random.default_rng(0)
    d = np.full(R, 1.0 / R)
    lay_curs = []                                      # attacker's played laydowns (log-surv)
    dbar = np.zeros(R)
    for it in range(250):
        _, cur = greedy_laydown(logsam, logman, sam_ok, man_ok, d)   # attacker BR to d
        lay_curs.append(cur)
        c = np.mean([dmg_of(cu) for cu in lay_curs], axis=0)          # def cost vs atk mix
        br = int(c.argmin())
        d = d * it / (it + 1); d[br] += 1.0 / (it + 1)               # def FP average
        dbar = d
    _, cur_star = greedy_laydown(logsam, logman, sam_ok, man_ok, dbar)  # atk BR to dbar
    eq_val = float((dbar * dmg_of(cur_star)).max()) if False else float((dbar * dmg_of(cur_star)).sum())
    # value = attacker BR to the average defender mix
    v_static = float((dbar * dmg_of(cur_star)).sum())

    # naive static family, each scored by the attacker's greedy BR
    def worst(dn):
        _, cu = greedy_laydown(logsam, logman, sam_ok, man_ok, dn)
        return float((dn * dmg_of(cu)).sum())
    d_uni = np.zeros(R); d_uni[lane_idx] = 1.0 / len(lane_idx)
    exp = 1 - np.exp(np.minimum(logsam, logman)).min(axis=1)          # worst single-sys exposure
    d_inv = np.zeros(R); d_inv[lane_idx] = 1 / np.clip(exp[lane_idx], 1e-6, None); d_inv /= d_inv.sum()
    naive = {"uniform_lanes": worst(d_uni), "invrisk_lanes": worst(d_inv)}
    bestk = np.inf
    for k in (2, 3, 4, 5):
        for T in itertools.combinations(range(R), k):
            dk = np.zeros(R); dk[list(T)] = 1.0 / k
            bestk = min(bestk, worst(dk))
    naive["best_k_stack"] = bestk
    best_naive = min(naive.values())
    print(f"\n(A) STATIC heterogeneous game (reshuffle per episode, NON-reacting):")
    print(f"    equilibrium (FP) value      {v_static:.3f}")
    for k, v in naive.items():
        print(f"    naive {k:16s}    {v:.3f}  ({v/v_static:.2f}x eq)")
    print(f"    => best naive / eq = {best_naive/v_static:.2f}x  "
          f"({'TIE (naive suffices)' if best_naive/v_static < 1.25 else 'gap survives'})",
          flush=True)

    # ---- (B) REACTIVE: 2 SAMs fixed (best static pair), 3 MANPADS RE-CUE to the window ----
    # fix the SAM pair = the greedy SAM picks vs dbar
    picks, _ = greedy_laydown(logsam, logman, sam_ok, man_ok, dbar)
    sam_cur = logsam[:, picks["sam"]].sum(axis=1)     # fixed SAM log-survival per route
    states = np.array(list(itertools.product(range(R), repeat=W)))
    Sn = len(states)
    # per unique window-route-SET, greedy 3 MANPADS to cover those routes (predict repeat)
    man_cache = {}
    def man_cur_for(winset):
        key = frozenset(winset)
        if key not in man_cache:
            wv = np.zeros(R);
            for r in winset: wv[r] += 1.0
            cur = sam_cur.copy(); budget = MAN["k"]; chosen = []
            for _ in range(budget):
                base = (wv * (1 - np.exp(cur))).sum(); best, bs = -1, None
                for s in man_ok:
                    if s in chosen: continue
                    g = (wv * (1 - np.exp(cur + logman[:, s]))).sum() - base
                    if g > best: best, bs = g, s
                chosen.append(bs); cur = cur + logman[:, bs]
            man_cache[key] = cur
        return man_cache[key]
    stepdmg = np.zeros((Sn, R))
    for i, st in enumerate(states):
        stepdmg[i] = dmg_of(man_cur_for(tuple(st)))
    pows = R ** np.arange(W - 1, -1, -1)
    shifted = np.concatenate([states[:, 1:], np.zeros((Sn, 1), int)], axis=1)
    succ = (shifted @ pows)[:, None] + np.arange(R)[None, :]
    # history_opt (lazy RVI)
    V = np.zeros(Sn)
    for _ in range(4000):
        Q = stepdmg + V[succ]; Vn = 0.5 * Q.min(axis=1) + 0.5 * V
        Vd = Vn - Vn.mean()
        if np.abs(Vd - V).max() < 1e-12: break
        V = Vd
    hist_opt = float((stepdmg + V[succ]).min(axis=1).mean() - V.mean() + V.mean())
    Q = stepdmg + V[succ]; hist_opt = float((Q.min(axis=1) - V).mean())
    # stationary value of a route rule (lazy power iteration)
    def stat(distfn):
        D = np.stack([distfn(tuple(s)) for s in states])
        pi = np.full(Sn, 1.0 / Sn)
        for _ in range(600):
            flow = pi[:, None] * D; nxt = np.zeros(Sn)
            np.add.at(nxt, succ.ravel(), flow.ravel()); nxt = 0.5 * nxt + 0.5 * pi
            if np.abs(nxt - pi).max() < 1e-13: break
            pi = nxt
        return float((pi[:, None] * D * stepdmg).sum())
    inwin = np.zeros((Sn, R), bool)
    for k in range(W): inwin[np.arange(Sn), states[:, k]] = True
    iid_eq = stat(lambda s: dbar)
    def anti(supp):
        m0 = np.zeros(R); m0[supp] = 1.0
        def f(s):
            m = m0.copy()
            for r in s: m[r] = 0.0
            return m / m.sum() if m.sum() > 1e-9 else m0 / m0.sum()
        return f
    supp_eq = np.where(dbar > 1e-4)[0]
    anti_eq = stat(anti(supp_eq)); anti_lane = stat(anti(lane_idx))
    best_dyn = min(anti_eq, anti_lane)
    print(f"\n(B) REACTIVE (2 SAM fixed + 3 MANPADS re-cue to the last {W} routes):")
    print(f"    history_opt (dynamic optimum) {hist_opt:.3f}")
    print(f"    iid_eq (best static mix)      {iid_eq:.3f}  ({iid_eq/hist_opt:.2f}x opt)")
    print(f"    anti-repeat (naive dynamic)   {best_dyn:.3f}  ({best_dyn/hist_opt:.2f}x opt)")
    gap = min(iid_eq, best_dyn) / hist_opt
    print(f"    => best static/naive-dyn / opt = {gap:.2f}x  "
          f"({'CORRIDOR opens' if gap >= 1.25 else 'no corridor'})", flush=True)
    print(f"\n[done {time.time()-t0:.0f}s]")
    json.dump({"static": {"eq": v_static, "naive": naive},
               "reactive": {"hist_opt": hist_opt, "iid_eq": iid_eq, "anti": best_dyn}},
              open("models/runs/gen33_hetero_probe.json", "w"), indent=1)


if __name__ == "__main__":
    main()
