"""Profile the gen39 step-3 training loop: where does a sortie's CPU go, and does update cost
grow with replay-buffer fill? (SYSTEM.md: profile before claiming a cause.)"""
import cProfile, io, pstats, time, math
import numpy as np, torch
from scripts.train_gen39_conceal import (Inst, TheatreEnv, narva_base, build_pools, N, W, S_EP)
from src.agents.sac import ProtagonistSAC
from src.agents.transition_builder import SMDPTransition

torch.set_num_threads(1)
base = narva_base()
train, val, test = build_pools(base, "llm", "models/runs/gen39_compose/forces_llm.json")
it = train[0]
env = TheatreEnv(base.menu, it.g.game, it.S_field if hasattr(it, "S_field") else it.g.S, N=N)
prot = ProtagonistSAC(node_in_dim=14, edge_in_dim=5, hidden_dim=64, num_layers=2, heads=4,
                      reward_scale=1.0, lr_actor=3e-4, autotune_alpha=True, alpha_init=1.0,
                      device="cpu", role_alpha=True, lr_alpha=5e-3, alpha_floor=0.20)
for net in (prot.actor, prot.q1, prot.q2, prot.target_q1, prot.target_q2):
    net.follow_w = torch.nn.Parameter(torch.tensor(1.0))
    net.route_feat_w = torch.nn.Parameter(torch.zeros(3)); net.route_feats = None
rng = np.random.default_rng(0)

def one_episode(push=True):
    window = tuple(int(x) for x in rng.integers(it.R, size=W)); mask = 0
    for _ in range(S_EP):
        env.reset(); f0 = it.feats(window, mask); leader = None; steps = []
        for _ in range(N):
            ci = env.current_convoy(); obs = env.observe(); obs["menu_route_feats"] = f0
            am = env.defender_action_mask()
            a = prot.select_action(obs, am)[ci] if ci == 0 else leader
            if ci == 0: leader = a
            env.route_convoy_by_index(int(a)); steps.append((obs, ci, a, am))
        r = int(leader); iw = int(np.dot(window, it.g.pows))
        rew = -10.0 * float(it.g.stepdmg[iw, r])
        mask |= int(it.expose[r]); window = tuple(list(window[1:]) + [r])
        if push:
            for j, (obs, ci, a, am) in enumerate(steps):
                obs["target_entropy"] = 0.5 * math.log(it.R); obs["alpha_group"] = 0
                prot.replay_buffer.push(SMDPTransition(agent="protagonist", state=obs,
                    action={ci: a}, reward=rew if j == N - 1 else 0.0, next_state={},
                    done=j == N - 1, elapsed_ticks=1, action_mask={"protagonist": am}, info={}))

t = time.time(); one_episode(); one_episode()
print(f"rollout: {(time.time()-t)/2/S_EP*1000:.0f} ms/sortie (no updates)")
for fill, label in ((200, "buffer~200"), (3000, "buffer~3000")):
    while len(prot.replay_buffer) < fill: one_episode()
    t = time.time()
    for _ in range(40): prot.update(32)
    print(f"updates {label}: {(time.time()-t)/40*1000:.0f} ms/update -> "
          f"{(time.time()-t)/S_EP*40/40:.2f} s/sortie-equivalent")
pr = cProfile.Profile(); pr.enable()
for _ in range(20): prot.update(32)
pr.disable(); s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(12)
print("\n".join(s.getvalue().splitlines()[:22]))
