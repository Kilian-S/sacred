"""Does area-stratified sampling fix the cover bias, and does it change the concealment verdict?"""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"): os.environ[v]="1"
import collections, numpy as np, time
from shapely.geometry import Point
from src.envs.aerial_conceal import ConcealBase, ConcealDyn, resample_field
from src.envs.aerial_theatre_vec import _class_parts, lateral_width, load_vec_theatre, terrain_v2
P="data/maps/theatre_%s_vec.json"; ref=lateral_width(load_vec_theatre(P%"kgd_gvardeysk"))
DOC=dict(q_rep=0.6,q_flee=0.2,q_ar=0.3)

def cover_area_seen(th, base, cls):
    gs=_class_parts(th,cls)
    if not gs: return float('nan'), 0
    pts=[Point(*base.coords[i]) for i,c in enumerate(base.cls) if c==cls]
    seen=sum(g.area for g in gs if any(g.covers(p) for p in pts))
    return 100*seen/sum(g.area for g in gs), len(pts)

for m in ("kgd_gvardeysk","ukraine"):
    th=load_vec_theatre(P%m); sc=lateral_width(th)/ref
    print(f"\n=== {m} ===")
    for strat in (0, 200, 400):
        t0=time.time()
        b=ConcealBase(P%m, terrain=terrain_v2(hidden_leth=1.0, conceal_reach=0.85),
                      range_scale=sc, spacing_km=2.0*sc, standoff_km=4.0*sc, stratified=strat)
        f_area,f_n=cover_area_seen(th,b,"forest"); u_area,u_n=cover_area_seen(th,b,"urban")
        print(f"  stratified={strat:4d}: H={b.H:4d} sites (forest {f_n:3d}, urban {u_n:3d}), "
              f"forest AREA reachable {f_area:3.0f}%, urban {u_area:3.0f}%  [build {time.time()-t0:.0f}s]")
        if strat in (0,400):
            pp=b.lethality(resample_field(b.coords,5100), hidden_leth=1.0)
            thr=b.threat_rank(pp); res={}
            for kind,pool in (("open",np.where(~b.concealed)[0]),("hidden",np.where(b.concealed)[0])):
                L=pool[np.argsort(-thr[pool])[:3]]
                g=ConcealDyn(b,pp,L,w=2,same_class=True,**DOC)
                sup=g.blind_supports()
                blind=min(g.episodic(rule=lambda i,mm,p,M=np.asarray(g._anti(d),float):M,T=40) for d in sup.values())
                obs=min([blind]+[g.episodic_rule(d,anti_repeat=a,softness=s,topm=t,T=40)
                                 for d in sup.values() for a in (False,True)
                                 for s,t in ((0.0,0),(0.05,0),(0.2,0),(0.0,2),(0.0,3),(0.0,5))])
                res[kind]=(g.episodic(T=40), obs)
            print(f"      -> optimum  open {res['open'][0]:.4f}  hidden {res['hidden'][0]:.4f}"
                  f"   ratio {res['hidden'][0]/max(res['open'][0],1e-9):.2f}")
            print(f"      -> observing open {res['open'][1]:.4f}  hidden {res['hidden'][1]:.4f}"
                  f"   ratio {res['hidden'][1]/max(res['open'][1],1e-9):.2f}")
