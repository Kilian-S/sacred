"""Export a gen32 DYNAMIC operations-map view: the trained policy + the adaptive enemy rolled
out over a serial on a real gated Kaliningrad->Gvardeysk threat laydown. Shows the enemy
re-aiming to the fleet's recent routes and the policy hedging against it, vs the naive rules.
Oracle/eval-only. Output: models/runs/gen32_dyn_view.json.
"""
import itertools
import json

import numpy as np
import torch

from scratch.gen32_theatre_hunt import TheatreBase, resample_field, DynTheatre
from src.agents.networks import featurize_state, node_index_map
from src.agents.sac import ProtagonistSAC, _clip_ea, _clip_x
from src.envs.aerial_theatre_vec import engagement_footprint, load_vec_theatre

FIELD, W, TAU = 4100, 3, 0.10
Q_REP, Q_FLEE, Q_AR = 0.6, 0.2, 0.3
CKPT, T = ("models/runs/gen32_confirm/seed10_ckpts/actor_ep15000.pt"), 80


def aim(g: DynTheatre, window):
    """The enemy's per-site aim distribution for a window (the 3-component doctrine)."""
    st = g.states[g.widx(tuple(window))]
    Vw = g.dmg[st].mean(axis=0)
    inwin = np.zeros(g.R, bool); inwin[list(window)] = True
    Zr = (Vw - Vw.max()) / TAU; Ar = np.exp(Zr); Ar /= Ar.sum()
    rflee = int((Ar @ g.dmg.T).argmin())
    ar = (~inwin).astype(float); ar /= ar.sum()
    Var = ar @ g.dmg
    Z = Q_REP * Vw + Q_FLEE * g.dmg[rflee] + Q_AR * Var
    A = np.exp((Z - Z.max()) / TAU); A /= A.sum()
    return A


def main():
    base = TheatreBase()
    th = load_vec_theatre("data/maps/theatre_kgd_gvardeysk_vec.json")
    field = resample_field(base.coords, FIELD)
    g = DynTheatre(base, field, W, TAU, Q_REP, Q_FLEE, Q_AR, build_env=True)
    env = g.env
    R = g.R

    # trained policy route distribution per window (encoder once, head per window)
    prot = ProtagonistSAC(node_in_dim=14, edge_in_dim=5, hidden_dim=64, num_layers=2, heads=4,
                          reward_scale=1.0, lr_actor=3e-4, autotune_alpha=True, alpha_init=1.0,
                          device="cpu", role_alpha=True, lr_alpha=5e-3, alpha_floor=0.20)
    prot.actor.follow_w = torch.nn.Parameter(torch.tensor(1.0))
    prot.actor.route_feat_w = torch.nn.Parameter(torch.zeros(3))
    prot.actor.route_feats = None
    prot.actor.load_state_dict(torch.load(CKPT))
    env.reset(); obs = env.observe()
    pyg = featurize_state(obs, 0); pyg.x = _clip_x(pyg.x, prot.node_in_dim)
    pyg.edge_attr = _clip_ea(pyg.edge_attr, prot.edge_in_dim)
    n2i = node_index_map(obs); active = n2i[obs["trucks"][0]["current_node"]]
    prot.actor.menu_routes = obs["menu_route_node_idx"]; prot.actor.eval()
    pol = {}
    with torch.no_grad():
        h = prot.actor.encoder(pyg.x, pyg.edge_index, pyg.edge_attr)

        def pol_dist(window):
            key = tuple(int(x) for x in window)
            if key not in pol:
                prot.actor.route_feats = g.feats(key)
                probs, _ = prot.actor.head(h, active, list(range(R)), torch.zeros(R))
                pol[key] = probs.numpy()
            return pol[key]

        # naive comparators
        exp = 1.0 - g.S.min(axis=1)
        d_eq = g.d_eq
        d_uni = np.zeros(R); d_uni[base.lane_idx] = 1.0 / len(base.lane_idx)

        def step_dmg(window, r):
            return float(g.stepdmg[g.widx(tuple(window))][r])

        def anti_dist(window):                       # blind anti-repeat over eq support
            m = d_eq.copy(); m[list(window)] = 0.0
            return m / m.sum() if m.sum() > 1e-9 else d_eq

        rng = np.random.default_rng(7)
        arms = {"policy": ("SACRED (trained)", pol_dist),
                "equilibrium": ("Static equilibrium", lambda w: d_eq),
                "uniform_lanes": ("Uniform lanes (naive)", lambda w: d_uni),
                "anti_repeat": ("Anti-repeat (naive+)", anti_dist)}
        rolls = {}
        for key, (label, distfn) in arms.items():
            win = list(rng.integers(R, size=W))
            steps, acc = [], 0.0
            for t in range(T):
                d = np.asarray(distfn(win), float)
                r = int(rng.choice(R, p=d / d.sum()))
                dmg = step_dmg(win, r)
                acc += dmg
                A = aim(g, win)
                top = [int(i) for i in np.argsort(A)[::-1][:6]]
                steps.append({"win": [int(x) for x in win], "dist": [round(float(x), 3) for x in d],
                              "route": r, "dmg": round(dmg, 3),
                              "aim": [[t2, round(float(A[t2]), 3)] for t2 in top],
                              "mean": round(acc / (t + 1), 3)})
                win = win[1:] + [r]
            rolls[key] = {"label": label, "steps": steps,
                          "final_mean": round(acc / T, 3)}

    # exact anchors for this field (cap / blind / optimum / policy)
    from scratch.gen32_theatre_hunt import rule_family
    rows = rule_family(g)
    cap = min(rows["iid_eq"], rows["static_localopt*fit"])
    best_blind = min(rows[k] for k in rows if k.startswith(("anti_", "rot_")))
    hist_opt = g.history_opt()
    with torch.no_grad():
        pol_mat = np.stack([pol_dist(w) for w in g.states])
    succ = g.succ; pi = np.full(len(g.states), 1.0 / len(g.states))
    for _ in range(1200):
        flow = pi[:, None] * pol_mat; nxt = np.zeros(len(g.states))
        np.add.at(nxt, succ.ravel(), flow.ravel()); nxt = 0.5 * nxt + 0.5 * pi
        if np.abs(nxt - pi).max() < 1e-13:
            pi = nxt; break
        pi = nxt
    pol_val = float((pi[:, None] * pol_mat * g.stepdmg).sum())

    used = sorted({site for r in rolls.values() for s in r["steps"] for site, _ in s["aim"]})
    vecd = json.load(open("data/maps/theatre_kgd_gvardeysk_vec.json"))
    foot = {str(hh): engagement_footprint(th, base.coords[hh], base.rr[hh], n_rays=56)
            for hh in used}
    out = dict(
        W=th.W, H=th.H, base=th.base.tolist(), target=th.target.tolist(),
        base_label=vecd["base"]["label"], target_label=vecd["target"]["label"],
        classes={k: [[[round(x, 2), round(y, 2)] for x, y in ring] for ring in v]
                 for k, v in vecd["classes"].items()},
        routes=[[[round(float(x), 2), round(float(y), 2)] for x, y in r[::2]]
                for r in base.menu],
        lane_idx=base.lane_idx,
        hazards=[[round(float(base.coords[hh][0]), 2), round(float(base.coords[hh][1]), 2),
                  round(float(base.rr[hh]), 2), round(float(field[hh]), 2)]
                 for hh in range(len(base.coords))],
        footprints=foot,
        field=FIELD, N=3,
        anchors=dict(cap=round(cap, 4), best_blind=round(best_blind, 4),
                     hist_opt=round(hist_opt, 4), policy=round(pol_val, 4),
                     eq_static=round(g.eq_static, 4)),
        rolls=rolls,
    )
    json.dump(out, open("models/runs/gen32_dyn_view.json", "w"))
    print("wrote gen32_dyn_view.json | field", FIELD, "routes", R, "sites", len(base.coords),
          "T", T)
    for k, v in rolls.items():
        print(f"  {k:14s} final mean mission-failure {v['final_mean']}")


if __name__ == "__main__":
    main()
