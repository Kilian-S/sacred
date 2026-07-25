"""Does the 2 km candidate raster miss small cover patches? Measure, per map."""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"): os.environ[v]="1"
import numpy as np
from shapely.geometry import Point
from src.envs.aerial_conceal import ConcealBase
from src.envs.aerial_theatre_vec import lateral_width, load_vec_theatre, terrain_v2
P="data/maps/theatre_%s_vec.json"
ref=lateral_width(load_vec_theatre(P%"kgd_gvardeysk"))

def parts(th,cls):
    out=[]
    for p in th.polys.get(cls,[]):
        out += list(p.geoms) if hasattr(p,"geoms") else [p]
    return [g for g in out if g.area>0]

for m in ("kgd_gvardeysk","ukraine","narva","fulda"):
    th=load_vec_theatre(P%m); sc=lateral_width(th)/ref
    b=ConcealBase(P%m, terrain=terrain_v2(), range_scale=sc, spacing_km=2.0*sc, standoff_km=4.0*sc)
    print(f"\n=== {m}  (grid {2.0*sc:.1f} km, {b.H} candidates) ===")
    for cls in ("forest","urban"):
        gs=parts(th,cls)
        if not gs: continue
        idx=[i for i,c in enumerate(b.cls) if c==cls]
        pts=[Point(*b.coords[i]) for i in idx]
        hit=[]; area_hit=0.0
        for g in gs:
            h=any(g.covers(p) for p in pts)
            hit.append(h); area_hit += g.area if h else 0.0
        tot=sum(g.area for g in gs)
        med=np.median([g.area for g in gs]); w=np.median([np.sqrt(g.area) for g in gs])
        print(f"  {cls:7s} {len(gs):5d} patches, total {tot:7.1f} km2, median patch {med:6.3f} km2 "
              f"(~{w:.2f} km across)")
        print(f"          patches WITH a candidate point: {sum(hit):4d}/{len(gs)} "
              f"({100*sum(hit)/len(gs):3.0f}%)  but they hold {100*area_hit/tot:3.0f}% of the area")
        big=[g for g in gs if g.area >= (2.0*sc)**2]
        bh=sum(1 for g in big if any(g.covers(p) for p in pts))
        print(f"          patches at least one grid-cell in size: {len(big)}, of which sampled "
              f"{bh} ({100*bh/max(len(big),1):3.0f}%)")
