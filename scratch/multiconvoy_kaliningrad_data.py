"""Extract the REAL Kaliningrad geometry + multi-convoy oracle strategies for the live visualiser.
Full road network (for the map) + the candidate routes / vulnerabilities / equilibrium strategies of
a multi-convoy interdiction instance on the actual graph."""
import json
import sys

import numpy as np

from src.baselines.interdiction_oracle import build_route_set, edges_of_route, length_band_vulnerability
from src.baselines.multiconvoy_oracle import objective_matrix, solve_multiconvoy
from src.envs.multiconvoy_interdiction import make_multiconvoy_env

OD = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("110", "135")
N = int(sys.argv[3]) if len(sys.argv) > 3 else 3
K, BAND = 1, (0.15, 0.95)

env = make_multiconvoy_env(od=OD, N=N, K=K, edge_vuln_band=BAND, objective="mission")
game = env.game
sol = solve_multiconvoy(game, N, "mission")
occs, M = objective_matrix(game, N, "mission")
det_i = int(np.argmin(M.max(axis=1))); br_det = int(M[det_i].argmax())

Genv = env.graph_env.graph
nodes = {str(n): [round(float(d["x"]), 6), round(float(d["y"]), 6)] for n, d in Genv.nodes(data=True)}
edges = [[str(u), str(v)] for u, v in Genv.edges()]

routes_rs = build_route_set(env.graph, OD[0], OD[1], 0, "w")
cand = set().union(*(edges_of_route(r) for r in routes_rs))
vuln = length_band_vulnerability(env.graph, cand, band=BAND, weight="w")

data = {
    "nodes": nodes, "edges": edges, "base": OD[0], "fob": OD[1], "N": N, "K": K,
    "routes": [list(r) for r in game.routes],
    "route_edges": [[sorted(e) for e in re] for re in game.route_edges],
    "isets": [{"edge": sorted(iset[0]), "vuln": round(float(vuln[iset[0]]), 4)}
              for iset in game.interdiction_sets],
    "occupancies": [[int(x) for x in o] for o in sol.occupancies],
    "defender_mixed": [round(float(x), 4) for x in sol.defender_strategy],
    "attacker_eq": [round(float(x), 4) for x in sol.attacker_strategy],
    "det_occupancy": [int(x) for x in occs[det_i]], "attacker_br_det": br_det,
    "loss_det": round(float(sol.loss_det), 3), "loss_mixed": round(float(sol.loss_mixed), 3),
}
print(f"OD {OD} N={N}: routes={game.n_routes} loss_det={data['loss_det']} "
      f"loss_mixed={data['loss_mixed']} gap={data['loss_det']-data['loss_mixed']:.3f} "
      f"det_occ={data['det_occupancy']}")
print("route lengths (hops):", [len(r) for r in game.routes])
print("route vulns:", [round(max(vuln[frozenset(e)] for e in re), 2) for re in game.route_edges])
print(f"graph: {len(nodes)} nodes, {len(edges)} edges")
out = "/private/tmp/claude-501/-Users-kilian-Kilian-ICL-Thesis-code-sacred/3614da47-f0d2-4cbf-8bff-4bff1d7b87b2/scratchpad/multiconvoy_kali_data.json"
open(out, "w").write(json.dumps(data))
print("[written]", out)
