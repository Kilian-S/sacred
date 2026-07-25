"""With the terrain leak removed, is there ANY setting where hiding pays? kgd, 3 fields."""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"): os.environ[v]="1"
import numpy as np
from src.envs.aerial_conceal import ConcealBase, ConcealDyn, resample_field
from src.envs.aerial_theatre_vec import terrain_v2
P="data/maps/theatre_kgd_gvardeysk_vec.json"; F=(5100,5101,5102); DOC=dict(q_rep=0.6,q_flee=0.2,q_ar=0.3)

def score(base, seed, hl, K, kind):
    pp=base.lethality(resample_field(base.coords,seed), hidden_leth=hl)
    thr=base.threat_rank(pp)
    pool=np.where(base.concealed)[0] if kind=="hidden" else np.where(~base.concealed)[0]
    if len(pool)<K: return None
    L=pool[np.argsort(-thr[pool])[:K]]
    g=ConcealDyn(base,pp,L,w=2,same_class=True,**DOC)
    sup=g.blind_supports()
    blind=min(g.episodic(rule=lambda i,m,p,M=np.asarray(g._anti(d),float):M,T=40) for d in sup.values())
    obs=min([blind]+[g.episodic_rule(d,anti_repeat=a,softness=s,topm=t,T=40)
                     for d in sup.values() for a in (False,True)
                     for s,t in ((0.0,0),(0.05,0),(0.2,0),(0.0,2),(0.0,3),(0.0,5))])
    return g.episodic(T=40), obs

print(f'{"K":>2s} {"reach":>5s} {"leth":>5s} | {"open opt":>9s} {"hid opt":>8s} {"ratio":>6s} | '
      f'{"open obs":>9s} {"hid obs":>8s} {"ratio":>6s}')
for K in (3,6):
    for cr in (0.65,0.85,1.0):
        for hl in (0.6,1.0):
            base=ConcealBase(P, terrain=terrain_v2(hidden_leth=1.0, conceal_reach=cr), range_scale=1.0)
            got={k:[score(base,s,hl,K,k) for s in F] for k in ("open","hidden")}
            if any(x is None for v in got.values() for x in v): continue
            oo=np.median([x[0] for x in got["open"]]); ho=np.median([x[0] for x in got["hidden"]])
            ob=np.median([x[1] for x in got["open"]]); hb=np.median([x[1] for x in got["hidden"]])
            print(f'{K:2d} {cr:5.2f} {hl:5.1f} | {oo:9.4f} {ho:8.4f} {ho/max(oo,1e-9):6.2f} | '
                  f'{ob:9.4f} {hb:8.4f} {hb/max(ob,1e-9):6.2f}')
