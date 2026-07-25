import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"): os.environ[v]="1"
import numpy as np
from src.envs.aerial_conceal import ConcealBase, ConcealDyn, resample_field
from src.envs.aerial_theatre_vec import lateral_width, load_vec_theatre, terrain_v2
P="data/maps/theatre_%s_vec.json"
ref=lateral_width(load_vec_theatre(P%"kgd_gvardeysk"))
print('BEST optimum over 6 laydowns (top-K open / top-K any / 4 random), K=3, so no single greedy')
print('laydown choice can drive the comparison.\n')
print(f'{"map":14s} {"forest opaque":>13s} {"forest clear":>12s} {"x":>6s} | '
      f'{"sightlines blocked":>18s}')
for m in ("kgd_gvardeysk","ukraine","narva","fulda"):
    th=load_vec_theatre(P%m); sc=lateral_width(th)/ref
    res={}
    for los in (True,False):
        base=ConcealBase(P%m, terrain=terrain_v2(conceal_reach=0.43, forest_los=los),
                         range_scale=sc, spacing_km=2.0*sc, standoff_km=4.0*sc)
        pp=base.lethality(resample_field(base.coords,5100),hidden_leth=1.0)
        thr=base.threat_rank(pp); op=np.where(~base.concealed)[0]
        cands=[op[np.argsort(-thr[op])[:3]], np.argsort(-thr)[:3]]
        rng=np.random.default_rng(7)
        cands += [rng.choice(base.H,size=3,replace=False) for _ in range(4)]
        res[los]=max(ConcealDyn(base,pp,L,w=2).episodic(T=40) for L in cands)
        if los:
            # how often is a site that is IN RANGE of a route leg nevertheless blind?
            inr = base.expo.sum()          # (route, site) pairs actually engaged, LOS applied
            base2=ConcealBase(P%m, terrain=terrain_v2(conceal_reach=0.43, forest_los=False),
                              range_scale=sc, spacing_km=2.0*sc, standoff_km=4.0*sc)
            blocked = 1.0 - inr/max(base2.expo.sum(),1)
    print(f'{m:14s} {res[True]:13.5f} {res[False]:12.5f} {res[False]/max(res[True],1e-9):6.1f}x | '
          f'{100*blocked:17.0f}%')
