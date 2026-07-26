#!/usr/bin/env python3
"""gen39 step 3: SACRED on the concealment game (experiments/gen39_concealment.md; the pinned
narva cell K=3, cr 0.85, rm 0.7, table lethality). gen32 machinery throughout; what is new is the
INFORMATION STATE: the policy's state is (track window, set of teams seen so far this mission),
the reveal channel arrives as a per-route head column (threat of the SPOTTED teams), and memory
of spotted teams is whole-mission, reset per episode (Kilian's persistence rule).

Head columns per route: [public exposure (terrain-lethality worst case, field-blind),
recency (window frequency), known-threat (spotted teams' zone damage, masked by what THIS
mission has seen)]. `--blind` zeroes the known-threat column: the causal control AND the
sighted-vs-blind concealment measurement (one arm, two duties, per the pinned step-3 record).

Arms (--arm): llm | random | heuristic - the training population the enemy is drawn from each
episode. Test = held-out enemies (never trained against, all three families + the oracle-searched
best force) on pristine fields 6100-6105. Exact eval: the policy's (window, mask)-conditioned
route distribution, scored by backward induction over the mission (episodic, T=40).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path as _P

import numpy as np
import torch

from src.agents.networks import featurize_state, node_index_map
from src.agents.sac import ProtagonistSAC, _clip_ea, _clip_x
from src.agents.transition_builder import SMDPTransition
from src.envs.aerial_conceal import (ConcealBase, ConcealDyn, choose_force, resample_field)
from src.envs.aerial_theatre_env import TheatreEnv
from src.envs.aerial_theatre_vec import lateral_width, load_vec_theatre, terrain_v2

MAP, CR, RM, K = "narva", 0.85, 0.7, 3
PATH = "data/maps/theatre_%s_vec.json"
W, TAU, S_EP, N = 2, 0.10, 40, 3
DOC32 = dict(q_rep=0.6, q_flee=0.2, q_ar=0.3)
TRAIN_FIELDS = tuple(range(1000, 1016))
VAL_FIELDS = (3000, 3001, 3002, 3003)
TEST_FIELDS = (6100, 6101, 6102, 6103, 6104, 6105)


def _mm(x):
    r = x.max() - x.min()
    return (x - x.min()) / r if r > 0 else np.zeros_like(x)


def narva_base():
    sc = lateral_width(load_vec_theatre(PATH % MAP)) / lateral_width(
        load_vec_theatre(PATH % "kgd_gvardeysk"))
    t = terrain_v2(hidden_leth=1.0, conceal_reach=CR)
    base = ConcealBase(PATH % MAP, terrain=t, range_scale=sc * RM, spacing_km=2.0 * sc,
                       standoff_km=4.0 * sc, n_sites=200)
    S_pub = base.survival(base.pp_base)                 # terrain-lethality worst case: PUBLIC
    base.expo_pub = _mm((1.0 - S_pub ** N).max(axis=1))
    return base


# --- forces ------------------------------------------------------------------------------------

def place(force, base, pp):
    """The step-2 placer verbatim (one placer everywhere)."""
    from scratch.gen39_compose import place as _place
    return _place(force, base, pp)


def doctrines_of(force):
    from scratch.gen39_compose import doctrines_of as _d
    return _d(force)


class Inst:
    """One (field, enemy force) game; slimmed after build so 100+ fit in RAM."""

    def __init__(self, base, name, field, sites=None, doctrines=None, archetype=None):
        self.name, self.field = name, field
        pp = base.lethality(resample_field(base.coords, field), hidden_leth=1.0)
        if archetype is not None:                       # heuristic family: gen32 doctrine
            sites, g, _ = choose_force(base, pp, archetype, K, np.random.default_rng(field),
                                       w=W, tau=TAU, doctrine=DOC32)
        else:
            g = ConcealDyn(base, pp, np.asarray(sites, int), w=W, tau=TAU, doctrines=doctrines)
        self.g, self.R = g, g.R
        expose, perceived, _ = g._memory_tables()
        self.expose = expose                            # [R] bitmask of teams a route gives away
        self.known_cols = np.stack([_mm(perceived[m]) for m in range(1 << len(g.L))])
        self.M = 1 << len(g.L)
        self.expo_pub = base.expo_pub
        self.S_field = g.S                              # field-level, shared per field
        Sn = len(g.states)
        self.wf = np.zeros((Sn, self.R))                # window-frequency column per state
        for k in range(W):
            np.add.at(self.wf, (np.arange(Sn), g.states[:, k]), 1.0 / W)

    def refs(self):
        g = self.g
        sup = g.blind_supports()
        Sn = len(g.states)
        eqv = g.episodic(rule=lambda i, m, p,
                         Mx=np.broadcast_to(g.d_eq, (Sn, g.R)).copy(): Mx, T=S_EP)
        blind = min(g.episodic(rule=lambda i, m, p, Mx=np.asarray(g._anti(d), float): Mx, T=S_EP)
                    for d in sup.values())
        obs = min([blind] + [g.episodic_rule(d, anti_repeat=a, softness=s, topm=t, T=S_EP)
                             for d in sup.values() for a in (False, True)
                             for s, t in ((0.0, 0), (0.05, 0), (0.2, 0), (0.0, 2), (0.0, 3),
                                          (0.0, 5))])
        self.cap, self.blind_ref, self.obs_ref = float(eqv), float(blind), float(obs)
        self.opt = float(g.episodic(T=S_EP))
        return self

    def slim(self):
        g = self.g
        for attr in ("aim", "dmg", "dmg_j", "prior_j", "S", "perceived", "known"):
            if hasattr(g, attr):
                delattr(g, attr)
        return self

    def feats(self, window, mask):
        wf = np.zeros(self.R)
        for r in window:
            wf[r] += 1.0 / W
        return torch.tensor(np.stack([self.expo_pub, wf, self.known_cols[mask]], axis=1),
                            dtype=torch.float32)


CACHE = "models/runs/gen39_step3/pool_cache.json"


def _llm_split(forces_path):
    llm_all = []
    if forces_path and _P(forces_path).exists():
        for r in json.load(open(forces_path)):
            if r.get("force") and not r.get("errors") and \
                    len(r["force"].get("agents", [])) == K:
                llm_all.append((f"{r['model']}#{r['j']}", r["force"]))
    n_hold = max(2, len(llm_all) // 4)
    return (llm_all[:-n_hold], llm_all[-n_hold:]) if llm_all else ([], [])


def _rnd_force(seed):
    from scratch.gen39_compose import random_force
    return random_force(np.random.default_rng(seed))


def prep_cache(base, forces_path, cache=CACHE):
    """The expensive shared prep, run ONCE before the batch: heuristic laydowns (choose_force per
    field x archetype), the common val set, the held-out test cells and every oracle ref. All 12
    runs load this; the yardsticks are byte-identical across arms and seeds by construction."""
    _, llm_test = _llm_split(forces_path)
    out = {"heur_train": {}, "val": [], "test": []}
    for f in TRAIN_FIELDS:
        pp = base.lethality(resample_field(base.coords, f), hidden_leth=1.0)
        out["heur_train"][str(f)] = {
            kind: [int(x) for x in choose_force(base, pp, kind, K, np.random.default_rng(f),
                                                w=W, tau=TAU, doctrine=DOC32)[0]]
            for kind in ("open", "hidden", "mixed")}
        print(f"  [prep] train field {f} done", flush=True)
    for f in VAL_FIELDS:
        it = Inst(base, f"val{f}", f, archetype="mixed").refs()
        out["val"].append(dict(name=it.name, field=f, sites=[int(x) for x in it.g.L],
                               doctrines=None, cap=it.cap, blind=it.blind_ref,
                               obs=it.obs_ref, opt=it.opt))
        print(f"  [prep] val field {f} done", flush=True)
    for f in TEST_FIELDS:
        pp = base.lethality(resample_field(base.coords, f), hidden_leth=1.0)
        cell = []
        if llm_test:
            nm, fo = llm_test[f % len(llm_test)]
            cell.append((f"te{f}-llm({nm})", place(fo, base, pp), doctrines_of(fo)))
        rf = _rnd_force(8000 + f)
        cell.append((f"te{f}-rnd", place(rf, base, pp), doctrines_of(rf)))
        best = None
        for kind in ("open", "hidden", "mixed"):
            it = Inst(base, f"te{f}-oracle({kind})", f, archetype=kind)
            v = it.g.episodic(T=S_EP)
            if best is None or v > best[0]:
                best = (v, it, kind)
        rows = []
        for nm, sites, doc in cell:
            it = Inst(base, nm, f, sites=sites, doctrines=doc).refs()
            rows.append(dict(name=nm, field=f, sites=[int(x) for x in np.asarray(sites)],
                             doctrines=doc, cap=it.cap, blind=it.blind_ref, obs=it.obs_ref,
                             opt=it.opt))
        heur = Inst(base, f"te{f}-heur", f, archetype="mixed").refs()
        rows.append(dict(name=heur.name, field=f, sites=[int(x) for x in heur.g.L],
                         doctrines=None, cap=heur.cap, blind=heur.blind_ref,
                         obs=heur.obs_ref, opt=heur.opt))
        orc = best[1].refs()
        rows.append(dict(name=f"te{f}-oracle({best[2]})", field=f,
                         sites=[int(x) for x in orc.g.L], doctrines=None, cap=orc.cap,
                         blind=orc.blind_ref, obs=orc.obs_ref, opt=orc.opt))
        out["test"].append(rows)
        print(f"  [prep] test field {f} done ({len(rows)} enemies)", flush=True)
    _P(cache).parent.mkdir(parents=True, exist_ok=True)
    _P(cache).write_text(json.dumps(out))
    print(f"[prep] cache written: {cache}", flush=True)


def _from_cache(base, rec):
    it = Inst(base, rec["name"], rec["field"], sites=rec["sites"], doctrines=rec["doctrines"])
    it.cap, it.blind_ref, it.obs_ref, it.opt = rec["cap"], rec["blind"], rec["obs"], rec["opt"]
    return it.slim()


def build_pools(base, arm, forces_path, cache=CACHE):
    """Training population per arm + the SHARED cached val/test structure."""
    c = json.loads(_P(cache).read_text())
    llm_train, _ = _llm_split(forces_path)
    train = []
    pp_cache = {}
    for f in TRAIN_FIELDS:
        if arm == "heuristic":
            for kind in ("open", "hidden", "mixed"):
                train.append(Inst(base, f"tr{f}-{kind}", f,
                                  sites=c["heur_train"][str(f)][kind]).slim())
        elif arm == "llm":
            for i in range(min(4, len(llm_train))):
                nm, fo = llm_train[(f + i) % len(llm_train)]
                pp = pp_cache.setdefault(f, base.lethality(
                    resample_field(base.coords, f), hidden_leth=1.0))
                train.append(Inst(base, f"tr{f}-{nm}", f, sites=place(fo, base, pp),
                                  doctrines=doctrines_of(fo)).slim())
        else:
            for i in range(3):
                fo = _rnd_force(7000 + 10 * f + i)
                pp = pp_cache.setdefault(f, base.lethality(
                    resample_field(base.coords, f), hidden_leth=1.0))
                train.append(Inst(base, f"tr{f}-rnd{i}", f, sites=place(fo, base, pp),
                                  doctrines=doctrines_of(fo)).slim())
    val = [_from_cache(base, r) for r in c["val"]]
    test = [[_from_cache(base, r) for r in rows] for rows in c["test"]]
    return train, val, test


# --- exact policy eval -------------------------------------------------------------------------

def policy_value(prot, inst, env, blind=False):
    """EXACT policy value in ONE head call. The head applies route_feats as a pure additive
    logit shift (`logit shift = feats @ w`, networks.py): for the fixed graph encoding,
    probs(window, mask) = softmax(log p0 + F(window, mask) @ w) where p0 is the head's output
    with route_feats zeroed. The first version looped Sn x M head calls (151k forwards per
    eval); it ground the whole batch to a halt and is replaced by this closed form, verified
    numerically identical (tests/test_gen39_trainer_eval.py)."""
    env.reset()
    obs = env.observe()
    pyg = featurize_state(obs, 0).to(prot.device)
    pyg.x = _clip_x(pyg.x, prot.node_in_dim)
    pyg.edge_attr = _clip_ea(pyg.edge_attr, prot.edge_in_dim)
    n2i = node_index_map(obs)
    active = n2i[obs["trucks"][0]["current_node"]]
    prot.actor.menu_routes = obs["menu_route_node_idx"]
    prot.actor.eval()
    g, R = inst.g, inst.R
    with torch.no_grad():
        h = prot.actor.encoder(pyg.x, pyg.edge_index, pyg.edge_attr)
        prot.actor.route_feats = torch.zeros(R, 3)
        p0, _ = prot.actor.head(h, active, list(range(R)), torch.zeros(R))
        wvec = prot.actor.route_feat_w.detach().cpu().numpy().astype(float)
    prot.actor.train()
    logp0 = np.log(np.clip(p0.cpu().numpy().astype(float), 1e-300, None))
    kw = 0.0 if blind else wvec[2]
    L = (logp0[None, None, :] + wvec[0] * inst.expo_pub[None, None, :]
         + wvec[1] * inst.wf[:, None, :] + kw * inst.known_cols[None, :, :])
    L -= L.max(axis=2, keepdims=True)
    dists = np.exp(L)
    dists /= dists.sum(axis=2, keepdims=True)
    return float(g.episodic(rule=lambda idx, m, p: dists[:, m, :], T=S_EP))


def save_full_state(prot, rng, sortie, hist, path):
    """STATE-COMPLETE run state: nets + optimisers + alpha + replay buffer + every RNG. The
    buffered transitions' featurization caches are SHARED per-field graphs, which pickle once
    per unique object, so they stay in the save (stripping them would regrow private copies
    after resume: the measured memory crawl)."""
    import random as _rnd
    _P(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"prot": prot, "rng": rng.bit_generator.state, "np": np.random.get_state(),
                "py": _rnd.getstate(), "torch": torch.get_rng_state(),
                "sortie": sortie, "hist": hist}, path)


def load_full_state(path):
    import random as _rnd
    st = torch.load(path, weights_only=False)
    np.random.set_state(st["np"])
    _rnd.setstate(st["py"])
    torch.set_rng_state(st["torch"])
    return st


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=("llm", "random", "heuristic"), required=True)
    p.add_argument("--forces", default="models/runs/gen39_compose/forces_llm.json")
    p.add_argument("--sorties", type=int, default=8000)
    p.add_argument("--resume", default="", help="full-state file; continues the run losslessly")
    p.add_argument("--state-out", default="", help="where the final full state lands "
                   "(default: <json-out>.state.pt)")
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--head-term-lr", type=float, default=3e-2)
    p.add_argument("--ent-frac", type=float, default=0.5)
    p.add_argument("--alpha-floor", type=float, default=0.20)
    p.add_argument("--interception-loss", type=float, default=10.0)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--blind", action="store_true")
    p.add_argument("--prep", action="store_true", help="build the shared oracle cache and exit")
    p.add_argument("--json-out", default="")
    p.add_argument("--ckpt-dir", default="")
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    print(f"[gen39-t] building pools (arm={args.arm}, blind={args.blind})...", flush=True)
    t0 = time.time()
    base = narva_base()
    if args.prep:
        prep_cache(base, args.forces)
        return
    train, val, test = build_pools(base, args.arm, args.forces)
    envs = {}
    for it in train + val + [x for cell in test for x in cell]:
        if it.field not in envs:
            envs[it.field] = TheatreEnv(base.menu, it.g.game, it.S_field, N=N)
        del it.S_field
    # ONE shared featurized graph per (field, convoy): the graph is identical for every
    # transition on a field (route feats ride separately at the head), but the update path
    # memoises a PRIVATE copy on each transition, which grew ~1 GB per run as the buffer filled
    # and pinned the machine at the memory-compression threshold (measured 2026-07-26: the
    # 7 s/sortie crawl at every priority). Pre-attaching the shared graph at push time stores
    # each tensor once; exactness pinned by test_gen39_trainer_eval.
    pyg_cache = {}
    for f, env_ in envs.items():
        env_.reset()
        obs_ = env_.observe()
        for ci in range(N):
            pyg_cache[(f, ci)] = featurize_state(obs_, ci).to("cpu")
    print(f"[gen39-t] pool in {time.time() - t0:.0f}s: {len(train)} train games, "
          f"{len(val)} val, {len(test)} test cells x {len(test[0])} enemies", flush=True)
    for cell in test:
        for it in cell:
            print(f"    {it.name}: cap={it.cap:.4f} blind={it.blind_ref:.4f} "
                  f"obs={it.obs_ref:.4f} opt={it.opt:.4f}", flush=True)

    start_sortie, hist = 0, []
    if args.resume and _P(args.resume).exists():
        st = load_full_state(args.resume)
        prot = st["prot"]
        rng = np.random.default_rng()
        rng.bit_generator.state = st["rng"]
        start_sortie, hist = st["sortie"], st["hist"]
        print(f"[gen39-t] RESUMED at sortie {start_sortie} from {args.resume}", flush=True)
    else:
        prot = ProtagonistSAC(node_in_dim=14, edge_in_dim=5, hidden_dim=64, num_layers=2, heads=4,
                              reward_scale=1.0, lr_actor=3e-4, autotune_alpha=True, alpha_init=1.0,
                              device="cpu", role_alpha=True, lr_alpha=5e-3,
                              alpha_floor=args.alpha_floor)
        for net in (prot.actor, prot.q1, prot.q2, prot.target_q1, prot.target_q2):
            net.follow_w = torch.nn.Parameter(torch.tensor(1.0))
            net.route_feat_w = torch.nn.Parameter(torch.zeros(3))   # [public, recency, known]
            net.route_feats = None
        prot.actor_optimizer.add_param_group({"params": [prot.actor.follow_w]})
        prot.actor_optimizer.add_param_group({"params": [prot.actor.route_feat_w],
                                              "lr": args.head_term_lr})
        prot.critic_optimizer.add_param_group({"params": [prot.q1.follow_w, prot.q2.follow_w]})
        prot.critic_optimizer.add_param_group({"params": [prot.q1.route_feat_w,
                                                          prot.q2.route_feat_w],
                                               "lr": args.head_term_lr})

    def test_rows():
        rows = {}
        for cell in test:
            for it in cell:
                rows[it.name] = policy_value(prot, it, envs[it.field], blind=args.blind)
        cells = [float(np.mean([rows[it.name] for it in cell])) for cell in test]
        return rows, cells

    if args.sorties == 0:
        rows, cells = test_rows()
        print(f"[gen39-t] UNTRAINED cells: " + " ".join(f"{c:.4f}" for c in cells), flush=True)
        return

    sortie = start_sortie
    t0 = time.time()
    while sortie < args.sorties:
        it = train[int(rng.integers(len(train)))]
        env = envs[it.field]
        g = it.g
        window = tuple(int(x) for x in rng.integers(it.R, size=W))
        mask = 0
        ep = []
        for _ in range(S_EP):
            env.reset()
            f0 = it.feats(window, mask)
            if args.blind:
                f0 = f0.clone(); f0[:, 2] = 0.0
            steps = []
            leader_act = None
            for _ in range(N):
                ci = env.current_convoy()
                obs = env.observe()
                obs["menu_route_feats"] = f0
                amask = env.defender_action_mask()
                act = (prot.select_action(obs, amask)[ci] if ci == 0 else leader_act)
                if ci == 0:
                    leader_act = act
                env.route_convoy_by_index(int(act))
                steps.append((obs, ci, act, amask))
            r_lead = int(leader_act)
            iw = int(np.dot(window, g.pows))
            dmgv = float(g.stepdmg[iw, r_lead])
            ep.append((steps, -args.interception_loss * dmgv))
            mask |= int(it.expose[r_lead])
            window = tuple(list(window[1:]) + [r_lead])
            sortie += 1
        flat = []
        for si, (steps, rew) in enumerate(ep):
            for j, (obs, ci, act, amask) in enumerate(steps):
                last = j == len(steps) - 1
                flat.append((obs, ci, act, amask, rew if last else 0.0, 1 if last else 0,
                             si == len(ep) - 1 and last))
        for idx, (obs, ci, act, amask, rew, dt, done) in enumerate(flat):
            obs["target_entropy"] = (0.05 if ci != 0 and sortie > 250 else args.ent_frac) \
                * math.log(it.R)
            obs["alpha_group"] = 1 if (ci != 0 and sortie > 250) else 0
            if done or idx == len(flat) - 1:
                nstate = {}
            else:
                nobs, nci, nmask = flat[idx + 1][0], flat[idx + 1][1], flat[idx + 1][3]
                nstate = dict(nobs)
                nstate["active_truck"] = nci
                nstate["allowed_destinations"] = {"protagonist": {nci: list(nmask[nci])}}
            tr = SMDPTransition(
                agent="protagonist", state=obs, action={ci: act}, reward=rew,
                next_state=nstate, done=bool(done), elapsed_ticks=dt,
                action_mask={"protagonist": amask}, info={})
            tr.feature_cache = {"state": pyg_cache[(it.field, ci)]}
            if nstate:
                tr.feature_cache["next"] = pyg_cache[(it.field, nstate["active_truck"])]
            prot.replay_buffer.push(tr)
        for _ in range(S_EP):
            prot.update(args.batch_size)

        if sortie % args.eval_every < S_EP:
            vv = [policy_value(prot, it2, envs[it2.field], blind=args.blind) for it2 in val]
            va = float(np.mean([v / max(it2.cap, 1e-9) for v, it2 in zip(vv, val)]))
            rows, cells = test_rows()
            caps = [float(np.mean([it2.cap for it2 in cell])) for cell in test]
            obs_r = [float(np.mean([it2.obs_ref for it2 in cell])) for cell in test]
            below_cap = sum(1 for c, r in zip(cells, caps) if c < r)
            below_obs = sum(1 for c, r in zip(cells, obs_r) if c < r)
            fw = tuple(float(x) for x in prot.actor.route_feat_w.detach())
            hist.append(dict(sortie=sortie, val=va, cells=cells, rows=rows, fw=fw,
                             alpha=float(prot.alpha)))
            if args.ckpt_dir:
                _P(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
                torch.save(prot.actor.state_dict(),
                           str(_P(args.ckpt_dir) / f"actor_ep{sortie}.pt"))
            print(f"  sortie {sortie:6d}: VAL {va:.2f} | TEST below-cap {below_cap}/6 "
                  f"below-obs {below_obs}/6 mean {np.mean(cells):.4f} | "
                  f"rw[{fw[0]:.2f},{fw[1]:.2f},{fw[2]:.2f}] a{prot.alpha:.2f} | "
                  f"{time.time() - t0:5.0f}s", flush=True)

    if args.json_out:
        refs = {}
        for cell in test:
            for it in cell:
                refs[it.name] = dict(cap=it.cap, blind=it.blind_ref, obs=it.obs_ref, opt=it.opt)
        _P(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        _P(args.json_out).write_text(json.dumps(
            {"arm": args.arm, "seed": args.seed, "blind": args.blind, "refs": refs,
             "history": hist}, indent=1))
        print(f"  [written] {args.json_out}", flush=True)
    state_out = args.state_out or (args.json_out + ".state.pt" if args.json_out else "")
    if state_out:
        save_full_state(prot, rng, sortie, hist, state_out)
        print(f"  [state] {state_out}", flush=True)


if __name__ == "__main__":
    main()
