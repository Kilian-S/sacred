#!/usr/bin/env python3
"""Extract an OSM drive network for a city into the geojson format
`src/utils/graph_utils.load_osm_graph_and_demands` consumes (nodes: osmid + [lon,lat]; edges:
u, v, length[m]), using the SAME 30m intersection consolidation as
`data/maps/kaliningrad_simplified_30m` so a second-city ZST test (A2) is comparable to the
training graph.

Needs network (osmnx / Overpass). Run on a machine with network access, e.g.:
    ! .venv/bin/python scripts/extract_city.py "Klaipeda, Lithuania" data/maps/klaipeda

Then the A2 harness runs unchanged:
    PYTHONPATH=. .venv/bin/python scratch/a2_graph_transfer.py <generalist_actor.pt> \
        --nodes data/maps/klaipeda/nodes.geojson --edges data/maps/klaipeda/edges.geojson --tag klaipeda
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    place, out_dir = sys.argv[1], Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    import osmnx as ox

    print(f"[extract_city] downloading drive network for {place!r} ...", flush=True)
    G = ox.graph_from_place(place, network_type="drive")
    # match the kaliningrad_simplified_30m pipeline: project, consolidate intersections at 30 m,
    # back to lat/lon. add_edge_lengths ensures 'length' (metres) exists post-consolidation.
    Gp = ox.project_graph(G)
    Gc = ox.consolidate_intersections(Gp, tolerance=30, rebuild_graph=True, dead_ends=True)
    Gc = ox.distance.add_edge_lengths(Gc)
    Gll = ox.project_graph(Gc, to_latlong=True)
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(Gll)

    # nodes: reindex 0..n-1 so ids are compact strings (as the kaliningrad export)
    id_map = {osmid: i for i, osmid in enumerate(nodes_gdf.index)}
    n_feats = []
    for osmid, row in nodes_gdf.iterrows():
        pt = row.geometry
        n_feats.append({"type": "Feature",
                        "properties": {"osmid": id_map[osmid]},
                        "geometry": {"type": "Point", "coordinates": [float(pt.x), float(pt.y)]}})
    e_feats = []
    for (u, v, k), row in edges_gdf.iterrows():
        length = float(row.get("length", 100.0))
        e_feats.append({"type": "Feature",
                        "properties": {"u": id_map[u], "v": id_map[v], "key": int(k),
                                       "length": length},
                        "geometry": {"type": "LineString",
                                     "coordinates": list(row.geometry.coords)}})
    (out_dir / "nodes.geojson").write_text(json.dumps(
        {"type": "FeatureCollection", "features": n_feats}))
    (out_dir / "edges.geojson").write_text(json.dumps(
        {"type": "FeatureCollection", "features": e_feats}))
    print(f"[extract_city] wrote {len(n_feats)} nodes, {len(e_feats)} edges to {out_dir}", flush=True)
    print(f"[extract_city] (Kaliningrad reference: 290 nodes / 706 edges after 30m consolidation)")


if __name__ == "__main__":
    main()
