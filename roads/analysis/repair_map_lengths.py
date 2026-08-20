"""Repair the edge lengths of extract_city.py outputs (2026-07-10 bug: add_edge_lengths was called
on the PROJECTED graph, so metre coordinates were treated as degrees -> lengths ~1e7 m). Recomputes
each edge's length as the haversine sum along its LineString geometry (lon/lat), preserving the
broken value as `length_raw` for audit. Idempotent (skips files whose median length is already sane).

Run: .venv/bin/python analysis/repair_map_lengths.py data/maps/gdansk data/maps/east_london data/maps/istanbul
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

R_EARTH = 6_371_000.0


def hav(lon1, lat1, lon2, lat2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R_EARTH * math.asin(math.sqrt(a))


for d in sys.argv[1:]:
    f = Path(d) / "edges.geojson"
    gj = json.loads(f.read_text())
    lens = [float(x["properties"].get("length", 0.0)) for x in gj["features"]]
    med = sorted(lens)[len(lens) // 2]
    if med < 100_000:
        print(f"{d}: median length {med:.0f} m looks sane; skipping")
        continue
    for x in gj["features"]:
        coords = x["geometry"]["coordinates"]
        L = sum(hav(*coords[i], *coords[i + 1]) for i in range(len(coords) - 1))
        x["properties"]["length_raw"] = x["properties"].get("length")
        x["properties"]["length"] = round(L, 1)
    f.write_text(json.dumps(gj))
    new = sorted(float(x["properties"]["length"]) for x in gj["features"])
    print(f"{d}: repaired {len(gj['features'])} edges; length m: min {new[0]:.0f} "
          f"med {new[len(new)//2]:.0f} max {new[-1]:.0f}")
