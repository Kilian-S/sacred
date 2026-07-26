"""gen39 step-3 trainer: the closed-form policy evaluation must equal the per-state head loop.

The head applies route_feats as an additive logit shift, so probs(window, mask) =
softmax(log p0 + F @ w). This pins that equivalence numerically (the 151k-forwards eval bug):
if the head's use of route_feats ever stops being a pure logit shift, this fires.
"""
import numpy as np
import torch

from src.agents.networks import featurize_state, node_index_map
from src.agents.sac import ProtagonistSAC, _clip_ea, _clip_x
from src.envs.aerial_conceal import ConcealBase, resample_field
from src.envs.aerial_theatre_vec import terrain_v2

KGD = "data/maps/theatre_kgd_gvardeysk_vec.json"


def test_shared_field_graph_equals_per_transition_featurization():
    """The pre-attached shared graph must equal what the update path's lambda would build for a
    transition on that field (the memory-crawl fix must not change a single tensor)."""
    from scripts.train_gen39_conceal import Inst, TheatreEnv, N
    base = ConcealBase(KGD, terrain=terrain_v2(), spacing_km=6.0, standoff_km=4.0)
    S_pub = base.survival(base.pp_base)
    from scripts.train_gen39_conceal import _mm
    base.expo_pub = _mm((1.0 - S_pub ** N).max(axis=1))
    pp = base.lethality(resample_field(base.coords, 5100))
    inst = Inst(base, "t", 5100, sites=np.where(~base.concealed)[0][:3])
    env = TheatreEnv(base.menu, inst.g.game, inst.S_field, N=N)
    env.reset()
    shared = {ci: featurize_state(env.observe(), ci) for ci in range(N)}
    env.reset()
    for ci in range(N):
        obs = env.observe()
        obs["menu_route_feats"] = inst.feats((0, 1), 0)     # the per-serial extras ride along
        obs["target_entropy"] = 1.0
        rebuilt = featurize_state(obs, ci)
        assert torch.equal(shared[ci].x, rebuilt.x)
        assert torch.equal(shared[ci].edge_index, rebuilt.edge_index)
        assert torch.equal(shared[ci].edge_attr, rebuilt.edge_attr)
        env.route_convoy_by_index(0)


def test_closed_form_policy_eval_matches_the_head_loop():
    from scripts.train_gen39_conceal import Inst, TheatreEnv, N
    base = ConcealBase(KGD, terrain=terrain_v2(), spacing_km=6.0, standoff_km=4.0)
    S_pub = base.survival(base.pp_base)
    from scripts.train_gen39_conceal import _mm
    base.expo_pub = _mm((1.0 - S_pub ** N).max(axis=1))
    pp = base.lethality(resample_field(base.coords, 5100))
    L = np.concatenate([np.where(~base.concealed)[0][:2], np.where(base.concealed)[0][:1]])
    inst = Inst(base, "t", 5100, sites=L)
    env = TheatreEnv(base.menu, inst.g.game, inst.S_field, N=N)

    torch.manual_seed(0)
    prot = ProtagonistSAC(node_in_dim=14, edge_in_dim=5, hidden_dim=32, num_layers=2, heads=2,
                          reward_scale=1.0, device="cpu")
    prot.actor.follow_w = torch.nn.Parameter(torch.tensor(1.0))
    prot.actor.route_feat_w = torch.nn.Parameter(torch.tensor([-0.7, -2.1, 1.3]))
    prot.actor.route_feats = None

    env.reset()
    obs = env.observe()
    pyg = featurize_state(obs, 0)
    pyg.x = _clip_x(pyg.x, prot.node_in_dim)
    pyg.edge_attr = _clip_ea(pyg.edge_attr, prot.edge_in_dim)
    active = node_index_map(obs)[obs["trucks"][0]["current_node"]]
    prot.actor.menu_routes = obs["menu_route_node_idx"]
    prot.actor.eval()
    g, R = inst.g, inst.R
    with torch.no_grad():
        h = prot.actor.encoder(pyg.x, pyg.edge_index, pyg.edge_attr)
        prot.actor.route_feats = torch.zeros(R, 3)
        p0, _ = prot.actor.head(h, active, list(range(R)), torch.zeros(R))
        wvec = prot.actor.route_feat_w.detach().numpy().astype(float)
        logp0 = np.log(np.clip(p0.numpy().astype(float), 1e-300, None))
        idx = list(range(0, len(g.states), 37))          # a spread of window states
        for m in range(inst.M):
            for i in idx:
                fe = inst.feats(g.states[i], m)
                prot.actor.route_feats = fe
                slow, _ = prot.actor.head(h, active, list(range(R)), torch.zeros(R))
                Lg = (logp0 + wvec[0] * inst.expo_pub + wvec[1] * inst.wf[i]
                      + wvec[2] * inst.known_cols[m])
                Lg -= Lg.max()
                fast = np.exp(Lg); fast /= fast.sum()
                assert np.allclose(slow.numpy(), fast, atol=1e-5), (i, m)
