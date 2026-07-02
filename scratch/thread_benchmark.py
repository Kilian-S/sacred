"""Find the optimal torch CPU thread count for SACRED's hot path (no training-dynamics change).

The bottleneck is the GATv2 encoder forward+backward (~67% of update() time per the profile).
This times one batched encoder pass over a workload matched to the real run (32 graphs x 290
nodes x ~412 edges) under several `torch.set_num_threads` settings. For a tiny graph, more
threads can cost more than they save, so the default (= all cores) may not be optimal.
"""

import os
import statistics
import time

import torch

from src.agents.networks import GATv2Encoder

N_GRAPHS = 32      # batch_size
N_NODES = 290      # Kaliningrad OSM
N_EDGES = 412
NODE_DIM, EDGE_DIM, HIDDEN = 9, 2, 64


def make_batch(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    xs, eis, eas, off = [], [], [], 0
    for _ in range(N_GRAPHS):
        x = torch.randn(N_NODES, NODE_DIM, generator=g)
        s = torch.randint(0, N_NODES, (N_EDGES,), generator=g)
        d = torch.randint(0, N_NODES, (N_EDGES,), generator=g)
        ei = torch.stack([torch.cat([s, d]), torch.cat([d, s])]) + off
        ea = torch.randn(ei.size(1), EDGE_DIM, generator=g)
        xs.append(x); eis.append(ei); eas.append(ea); off += N_NODES
    return torch.cat(xs, 0), torch.cat(eis, 1), torch.cat(eas, 0)


def main():
    x, ei, ea = make_batch()
    enc = GATv2Encoder(NODE_DIM, EDGE_DIM, hidden_dim=HIDDEN, num_layers=2, heads=4)

    def run_once():
        enc.zero_grad()
        h = enc(x, ei, ea)
        h.pow(2).mean().backward()

    cores = os.cpu_count() or 8
    candidates = sorted({1, 2, 4, 6, 8, cores})
    candidates = [n for n in candidates if n <= cores]
    print(f"cpu_count={cores}, default torch threads={torch.get_num_threads()}, "
          f"workload={N_GRAPHS}x{N_NODES} nodes\n")

    results = {}
    for n in candidates:
        torch.set_num_threads(n)
        for _ in range(5):   # warmup
            run_once()
        ts = []
        for _ in range(25):  # timed
            t0 = time.perf_counter(); run_once(); ts.append(time.perf_counter() - t0)
        med = statistics.median(ts) * 1000
        results[n] = med
        print(f"threads={n:2d}: median={med:7.2f} ms/iter")

    best = min(results, key=results.get)
    base = results.get(torch.get_num_threads(), results[cores])
    print(f"\nfastest: threads={best} ({results[best]:.2f} ms); "
          f"speedup vs all-cores ({cores}): {results[cores] / results[best]:.2f}x")


if __name__ == "__main__":
    main()
