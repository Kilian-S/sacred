#!/usr/bin/env python3
"""Visualise the extract_city pipeline at several consolidation TOLERANCES in one shot.

Downloads the drive network and applies the arterial-highway filter ONCE (the expensive
network step), then sweeps `--tolerances`, consolidating intersections at each and rendering
every resulting network into a single side-by-side comparison PNG (node/edge counts in each
panel title) so you can pick the tolerance that lands near the Kaliningrad reference (290/706).

Same bbox convention as scripts/extract_city.py: --bbox north,south,east,west (reordered
internally to the osmnx 2.x (west,south,east,north) order). Needs live OSM/Overpass access, so
run it on a networked machine via the `!` prefix, e.g.:

    ! .venv/bin/python scratch/mapgen/tolerance_sweep.py \
        --bbox 54.4046,54.3318,18.7195,18.6198 \
        --tolerances 20,30,40,50,60 --out scratch/mapgen/gdansk_tolsweep.png

Then run extract_city.py once at the chosen tolerance to write the geojson.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: save PNG, no display
import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox

ARTERIAL = {"primary", "secondary", "tertiary", "trunk", "motorway",
            "primary_link", "secondary_link", "tertiary_link", "trunk_link", "motorway_link"}


def arterial_filter(G):
    """Keep only arterial-highway edges; drop isolated nodes (identical to extract_city)."""
    keep = []
    for u, v, k, d in G.edges(keys=True, data=True):
        hw = d.get("highway", "")
        hw = hw if isinstance(hw, list) else [hw]
        if any(h in ARTERIAL for h in hw):
            keep.append((u, v, k))
    G = G.edge_subgraph(keep).copy()
    G.remove_nodes_from(list(nx.isolates(G)))
    return G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", required=True, help="north,south,east,west (decimal degrees)")
    ap.add_argument("--tolerances", default="20,30,40,50,60",
                    help="comma-separated consolidation tolerances in metres")
    ap.add_argument("--no-filter", action="store_true", help="skip the arterial-highway filter")
    ap.add_argument("--node-size", type=float, default=6.0)
    ap.add_argument("--out", default="scratch/mapgen/tolerance_sweep.png")
    args = ap.parse_args()

    n, s, e, w = (float(x) for x in args.bbox.split(","))
    tols = [float(x) for x in args.tolerances.split(",")]

    print(f"[sweep] downloading drive network for bbox N{n} S{s} E{e} W{w} ...", flush=True)
    G = ox.graph_from_bbox(bbox=(w, s, e, n), network_type="drive", simplify=True)  # osmnx 2.x order
    print(f"  raw: {G.number_of_nodes()} nodes / {G.number_of_edges()} edges")

    if not args.no_filter:
        G = arterial_filter(G)
        print(f"  arterial-filtered: {G.number_of_nodes()} nodes / {G.number_of_edges()} edges")

    Gp = ox.project_graph(G)  # consolidate needs a projected graph; project once

    # Build the panel list: the unconsolidated base, then one graph per tolerance.
    base_label = "arterial base" if not args.no_filter else "full drive base"
    panels = [(f"{base_label} (unconsolidated)", Gp)]
    rows = [("(none)", Gp.number_of_nodes(), Gp.number_of_edges())]
    for tol in tols:
        try:
            Gc = ox.consolidate_intersections(Gp, tolerance=tol, rebuild_graph=True, dead_ends=False)
            panels.append((f"tolerance {tol:g} m", Gc))
            rows.append((f"{tol:g} m", Gc.number_of_nodes(), Gc.number_of_edges()))
        except Exception as ex:  # never let one bad tolerance kill the sweep
            print(f"  [tol {tol:g}m FAILED] {type(ex).__name__}: {ex}")

    # Grid layout.
    ncols = min(3, len(panels))
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5.0, nrows * 5.0))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, (title, Gx) in zip(axes, panels):
        ox.plot_graph(Gx, ax=ax, show=False, close=False, bgcolor="white",
                      node_color="#d62728", node_size=args.node_size,
                      edge_color="#444444", edge_linewidth=0.7)
        ax.set_title(f"{title}\n{Gx.number_of_nodes()} nodes / {Gx.number_of_edges()} edges",
                     fontsize=12)
    for ax in axes[len(panels):]:  # hide any empty cells
        ax.axis("off")

    fig.suptitle(f"Consolidation-tolerance sweep  (Kaliningrad ref: 290 nodes / 706 edges)",
                 fontsize=14)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[sweep] wrote {out}")

    print("\n  tolerance |  nodes | edges(directed)")
    print("  ----------+--------+----------------")
    for tol, nn, ne in rows:
        flag = "  <- near Kaliningrad 290" if 250 <= nn <= 450 else ""
        print(f"  {tol:>8} | {nn:>6} | {ne:>6}{flag}")


if __name__ == "__main__":
    main()
