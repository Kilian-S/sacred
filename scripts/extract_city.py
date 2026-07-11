#!/usr/bin/env python3
"""Extract an OSM drive network into the geojson format
`src/utils/graph_utils.load_osm_graph_and_demands` consumes, replicating the EXACT Kaliningrad
pipeline (`scratch/mapgen/export_geojson.py` + the `plot_filter.py` arterial filter) so a
second-city ZST test (A2) is built the same way as the training graph:

  1. download the drive network for a BBOX (osmnx / Overpass; needs network) with simplify=True;
  2. FILTER to arterial highways only (primary/secondary/tertiary/trunk/motorway + links) and drop
     isolated nodes -- the main node-count reducer (residential/service streets removed);
  3. CONSOLIDATE intersections at `--tolerance` metres (default 30, as Kaliningrad; dead_ends=False);
  4. export nodes (osmid + [lon,lat]) and edges (u, v, length[m]) geojson.

Kaliningrad reference: after these steps, 290 nodes / 706 edges. Tune --bbox and --tolerance so the
printed node count lands in a comparable, processable range (~250-450).

Run on a machine with network (e.g. via the `!` prefix in Claude Code):
    ! .venv/bin/python scripts/extract_city.py --place "Kyiv, Ukraine" \
        --bbox 50.52,50.38,30.66,30.40 --tolerance 30 --out data/maps/kyiv
Then A2 consumes it unchanged:
    PYTHONPATH=. .venv/bin/python scratch/a2_graph_transfer.py <generalist_actor.pt> \
        --nodes data/maps/kyiv/nodes.geojson --edges data/maps/kyiv/edges.geojson --tag kyiv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ARTERIAL = {"primary", "secondary", "tertiary", "trunk", "motorway",
            "primary_link", "secondary_link", "tertiary_link", "trunk_link", "motorway_link"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--place", default=None, help="place name (used only if --bbox omitted)")
    ap.add_argument("--bbox", default=None, help="north,south,east,west (decimal degrees) -- STRONGLY "
                    "recommended for big cities; Kaliningrad's region was pre-cropped")
    ap.add_argument("--tolerance", type=float, default=30.0, help="intersection-consolidation metres")
    ap.add_argument("--no-filter", action="store_true", help="skip the arterial-highway filter")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import networkx as nx
    import osmnx as ox

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if args.bbox:
        n, s, e, w = (float(x) for x in args.bbox.split(","))
        print(f"[extract_city] downloading drive network for bbox N{n} S{s} E{e} W{w} ...", flush=True)
        # osmnx 2.x expects bbox=(left, bottom, right, top) = (west, south, east, north);
        # the CLI takes the intuitive north,south,east,west order and we reorder here.
        G = ox.graph_from_bbox(bbox=(w, s, e, n), network_type="drive", simplify=True)
    elif args.place:
        print(f"[extract_city] downloading drive network for {args.place!r} (whole place; use "
              f"--bbox to crop) ...", flush=True)
        G = ox.graph_from_place(args.place, network_type="drive")
    else:
        ap.error("give --bbox or --place")
    print(f"  raw: {G.number_of_nodes()} nodes / {G.number_of_edges()} edges")

    # 2. arterial-highway filter (drop residential/service; remove isolated nodes)
    if not args.no_filter:
        keep = []
        for u, v, k, d in G.edges(keys=True, data=True):
            hw = d.get("highway", "")
            hw = hw if isinstance(hw, list) else [hw]
            if any(h in ARTERIAL for h in hw):
                keep.append((u, v, k))
        G = G.edge_subgraph(keep).copy()
        G.remove_nodes_from(list(nx.isolates(G)))
        print(f"  arterial-filtered: {G.number_of_nodes()} nodes / {G.number_of_edges()} edges")

    # 3. project + consolidate at tolerance m + back to lat/lon (Kaliningrad: tol 30, dead_ends False).
    # tolerance <= 0 means NO consolidation (keep the raw arterial nodes): osmnx's
    # consolidate_intersections rebuilds to an empty-edge graph at tolerance 0, and the graph is
    # already lat/lon out of graph_from_bbox, so we just skip straight to length + export.
    if args.tolerance > 0:
        Gp = ox.project_graph(G)
        Gc = ox.consolidate_intersections(Gp, rebuild_graph=True, tolerance=args.tolerance, dead_ends=False)
        Gll = ox.project_graph(Gc, to_latlong=True)
    else:
        Gll = G
    # lengths AFTER re-projecting to lat/long: add_edge_lengths assumes degree coordinates, and
    # calling it on the projected (metre) graph produced ~1e7 m lengths (2026-07-10 bug, repaired
    # post hoc by scratch/repair_map_lengths.py for the first three cities).
    Gll = ox.distance.add_edge_lengths(Gll)
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(Gll)
    print(f"  consolidated @ {args.tolerance}m: {len(nodes_gdf)} nodes / {len(edges_gdf)} edges "
          f"(Kaliningrad ref: 290 / 706)")

    id_map = {osmid: i for i, osmid in enumerate(nodes_gdf.index)}
    n_feats = [{"type": "Feature", "properties": {"osmid": id_map[osmid]},
                "geometry": {"type": "Point", "coordinates": [float(r.geometry.x), float(r.geometry.y)]}}
               for osmid, r in nodes_gdf.iterrows()]
    e_feats = [{"type": "Feature",
                "properties": {"u": id_map[u], "v": id_map[v], "key": int(k),
                               "length": float(r.get("length", 100.0))},
                "geometry": {"type": "LineString", "coordinates": list(r.geometry.coords)}}
               for (u, v, k), r in edges_gdf.iterrows()]
    (out / "nodes.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": n_feats}))
    (out / "edges.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": e_feats}))
    print(f"[extract_city] wrote {len(n_feats)} nodes / {len(e_feats)} edges to {out}")
    if not 200 <= len(n_feats) <= 500:
        print(f"  NOTE: {len(n_feats)} nodes is outside the comparable ~250-450 band; adjust "
              f"--bbox (smaller = fewer) or --tolerance (larger = fewer) and re-run.")


if __name__ == "__main__":
    main()
