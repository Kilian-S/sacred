#!/usr/bin/env python3
"""gen39: re-aim the Fulda corridor (Kilian 2026-07-25).

The fetched corridor ran Point Alpha -> Frankfurt. Kilian wants it the other way and started
SOUTH-WEST of Frankfurt, so that the direct flight path crosses the city's urban core: the run is
now RHEIN-MAIN SW -> POINT ALPHA, i.e. resupply from the rear area up to the inner-German border
observation post, threading the Frankfurt conurbation on the way out.

The start is placed on the Frankfurt -> Point Alpha bearing, 12 km back from the city centre, so
Frankfurt sits ON the straight line rather than beside it. Urban ground blocks line of sight, so
the city is the corridor's first and largest sight-line wall.

Idempotent: run it as often as you like. The map JSONs live under data/ (gitignored), so this
script is the committed record of the change.

    PYTHONPATH=. python scratch/gen39_retarget_fulda.py
"""
from __future__ import annotations

import json

import numpy as np

PATH = "data/maps/theatre_fulda_vec.json"
BACK_KM = 12.0                      # how far back along the bearing to start
NEW_BASE_LABEL = "RHEIN-MAIN SW"
NEW_TARGET_LABEL = "POINT ALPHA"


def main():
    d = json.load(open(PATH))
    fra = np.array(d["base"]["xy_km"] if d["base"]["label"] == "FRANKFURT"
                   else d["target"]["xy_km"], dtype=float)
    alpha = np.array(d["target"]["xy_km"] if d["target"]["label"] == "POINT ALPHA"
                     else d["base"]["xy_km"], dtype=float)
    if d["base"]["label"] == NEW_BASE_LABEL:
        fra = np.array(d.get("waypoint_frankfurt_km", fra), dtype=float)

    u = (alpha - fra) / np.linalg.norm(alpha - fra)
    start = fra - BACK_KM * u
    start[0] = float(np.clip(start[0], 0.6, d["W_km"] - 0.6))
    start[1] = float(np.clip(start[1], 0.6, d["H_km"] - 0.6))

    d["base"] = {"label": NEW_BASE_LABEL, "xy_km": [round(float(start[0]), 3),
                                                    round(float(start[1]), 3)]}
    d["target"] = {"label": NEW_TARGET_LABEL, "xy_km": [round(float(alpha[0]), 3),
                                                        round(float(alpha[1]), 3)]}
    d["waypoint_frankfurt_km"] = [round(float(fra[0]), 3), round(float(fra[1]), 3)]
    d["retargeted"] = ("gen39 2026-07-25: start SW of Frankfurt so the direct path crosses the "
                       "urban core; Point Alpha is the destination")
    json.dump(d, open(PATH, "w"))

    off = np.linalg.norm(np.cross(u, fra - start))          # perpendicular miss distance
    print(f"base   {d['base']['label']:16s} {d['base']['xy_km']}")
    print(f"target {d['target']['label']:16s} {d['target']['xy_km']}")
    print(f"Frankfurt at {d['waypoint_frankfurt_km']}, {BACK_KM:.0f} km along the run, "
          f"{off:.2f} km off the straight line")
    print(f"corridor {np.linalg.norm(alpha - start):.1f} km")


if __name__ == "__main__":
    main()
