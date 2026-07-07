"""Compute a clean toy multi-convoy interdiction instance + its oracle strategies, dumped as JSON
for the interactive visualiser. Toy graph: base B -> 4 disjoint corridors -> FOB F, soft interception
(short corridor = safe, long = exposed), N=3 convoys, K=1 interdictor, mission-failure objective."""
import json

import networkx as nx
import numpy as np

from src.baselines.interdiction_oracle import (
    build_interdiction_game, build_route_set, edges_of_route,
    length_band_vulnerability, survival_intercept_fn)
from src.baselines.multiconvoy_oracle import objective_matrix, solve_multiconvoy

POS = {"B": (0.0, 0.0), "F": (12.0, 0.0),
       "n1": (6.0, 3.4), "n2": (6.0, 1.15), "n3": (6.0, -1.15), "n4": (6.0, -3.4)}
EDGES = [("B", "n1", 4.0), ("n1", "F", 4.0),      # route 1: least exposed
         ("B", "n2", 5.0), ("n2", "F", 5.0),      # route 2
         ("B", "n3", 6.0), ("n3", "F", 6.0),      # route 3
         ("B", "n4", 7.0), ("n4", "F", 7.0)]      # route 4: most exposed
N, K = 3, 1
_BAND = (0.40, 0.72)

G = nx.Graph()
for u, v, d in EDGES:
    G.add_edge(u, v, w=float(d))
routes = build_route_set(G, "B", "F", 0, "w")
cand = set().union(*(edges_of_route(r) for r in routes))
vuln = length_band_vulnerability(G, cand, band=_BAND, weight="w")
game = build_interdiction_game(G, "B", "F", K, k_extra=0, weight="w",
                               intercept_fn=survival_intercept_fn(vuln))
sol = solve_multiconvoy(game, N, "mission")
occs, M = objective_matrix(game, N, "mission")
det_i = int(np.argmin(M.max(axis=1)))
br_det = int(M[det_i].argmax())

data = {
    "pos": {k: list(v) for k, v in POS.items()},
    "edges": [[u, v] for u, v, _ in EDGES],
    "base": "B", "fob": "F", "N": N, "K": K,
    "routes": [list(r) for r in game.routes],
    "route_edges": [[sorted(e) for e in re] for re in game.route_edges],
    "isets": [{"edge": sorted(iset[0]), "vuln": float(vuln[iset[0]])}
              for iset in game.interdiction_sets],
    "occupancies": [[int(x) for x in o] for o in sol.occupancies],
    "defender_mixed": [float(x) for x in sol.defender_strategy],
    "attacker_eq": [float(x) for x in sol.attacker_strategy],
    "det_occupancy": [int(x) for x in occs[det_i]],
    "attacker_br_det": br_det,
    "loss_det": round(float(sol.loss_det), 3),
    "loss_mixed": round(float(sol.loss_mixed), 3),
}
print(f"routes={game.n_routes} isets={len(game.interdiction_sets)} "
      f"loss_det={data['loss_det']} loss_mixed={data['loss_mixed']} gap={data['loss_det']-data['loss_mixed']:.3f}")
print("det_occupancy (convoys per route):", data["det_occupancy"])
print("route vulns (max edge per route):",
      [round(max(vuln[frozenset(e)] for e in re), 2) for re in game.route_edges])
out = "/private/tmp/claude-501/-Users-kilian-Kilian-ICL-Thesis-code-sacred/3614da47-f0d2-4cbf-8bff-4bff1d7b87b2/scratchpad/multiconvoy_toy_data.json"
open(out, "w").write(json.dumps(data))
print("[written]", out)
