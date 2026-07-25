"""Is the remaining cover gap a SAMPLING limit or a TERRAIN fact? Push the budget until it saturates."""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"): os.environ[v]="1"
import numpy as np, time
from shapely.geometry import Point
from src.envs.aerial_conceal import ConcealBase, ConcealDyn, resample_field
from src.envs.aerial_theatre_vec import _class_parts, lateral_width, load_vec_theatre, terrain_v2
P="data/maps/theatre_%s_vec.json"; ref=lateral_width(load_vec_theatre(P%"kgd_gvardeysk"))
DOC=dict(q_rep=0.6,q_flee=0.2,q_ar=0.3)
for m in ("kgd_gvardeysk","ukraine"):
    th=load_vec_theatre(P%m); sc=lateral_width(th)/ref
    gs=_class_parts(th,"forest"); tot=sum(g.area for g in gs)
    print(f"\n=== {m}: {len(gs)} forest patches, {tot:.0f} km2 total, "
          f"largest {max(g.area for g in gs):.1f} km2, weapon radius {2.98*sc:.1f} km ===")
    big=[g for g in gs if g.area >= np.pi*(2.98*sc)**2]
    print(f"    patches at least as big as one weapon footprint: {len(big)} "
          f"({100*sum(g.area for g in big)/tot:.0f}% of forest area)")
    for strat in (400, 1200, 2400):
        t0=time.time()
        b=ConcealBase(P%m, terrain=terrain_v2(hidden_leth=1.0, conceal_reach=0.85),
                      range_scale=sc, spacing_km=2.0*sc, standoff_km=4.0*sc, stratified=strat)
        pts=[Point(*b.coords[i]) for i,c in enumerate(b.cls) if c=="forest"]
        seen=100*sum(g.area for g in gs if any(g.covers(p) for p in pts))/tot
        pp=b.lethality(resample_field(b.coords,5100), hidden_leth=1.0)
        thr=b.threat_rank(pp); res={}
        for kind,pool in (("open",np.where(~b.concealed)[0]),("hidden",np.where(b.concealed)[0])):
            L=pool[np.argsort(-thr[pool])[:3]]
            g_=ConcealDyn(b,pp,L,w=2,same_class=True,**DOC); sup=g_.blind_supports()
            blind=min(g_.episodic(rule=lambda i,mm,p,M=np.asarray(g_._anti(d),float):M,T=40) for d in sup.values())
            obs=min([blind]+[g_.episodic_rule(d,anti_repeat=a,softness=s,topm=t,T=40)
                             for d in sup.values() for a in (False,True)
                             for s,t in ((0.0,0),(0.05,0),(0.2,0),(0.0,2),(0.0,3),(0.0,5))])
            res[kind]=(g_.episodic(T=40), obs)
        print(f"  strat={strat:5d}: H={b.H:5d} forest sites={len(pts):4d} area seen={seen:3.0f}%  "
              f"opt ratio={res['hidden'][0]/max(res['open'][0],1e-9):.2f}  "
              f"obs ratio={res['hidden'][1]/max(res['open'][1],1e-9):.2f}  [{time.time()-t0:.0f}s]")
