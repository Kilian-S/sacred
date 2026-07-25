"""Why are fulda/narva dead? Measure the geometry the game actually turns on."""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ[v]="1"
import numpy as np
from src.envs.aerial_conceal import ConcealBase, ConcealDyn, resample_field
from src.envs.aerial_theatre_vec import lateral_width, load_vec_theatre, terrain_v2
P="data/maps/theatre_%s_vec.json"
ref=lateral_width(load_vec_theatre(P%"kgd_gvardeysk"))
print(f'{"map":14s} {"corridor":>8s} {"lat width":>9s} {"scale":>6s} {"open reach":>10s} '
      f'{"route sep":>9s} {"reach/sep":>9s} {"routes 1 team can touch":>23s}')
for m in ("kgd_gvardeysk","ukraine","narva","fulda"):
    th=load_vec_theatre(P%m); sc=lateral_width(th)/ref
    base=ConcealBase(P%m, terrain=terrain_v2(conceal_reach=0.43), range_scale=sc,
                     spacing_km=2.0*sc, standoff_km=4.0*sc)
    corridor=float(np.linalg.norm(th.target-th.base))
    u=(th.target-th.base)/corridor; n=np.array([-u[1],u[0]])
    # lateral offset of each route: max perpendicular deviation from the straight line
    off=[float(np.max(np.abs((r-th.base)@n))) for r in base.menu]
    spread=float(np.percentile(off,90))                      # how wide the menu really is
    sep=2*spread/max(len(base.menu)-1,1)
    reach=3.5*sc
    # how many routes does ONE emplaced team actually threaten?
    pp=base.lethality(resample_field(base.coords,5100),hidden_leth=1.0)
    dmg=1.0-base.survival(pp)**3
    touch=[int((dmg[:,j]>0.02*dmg.max()).sum()) for j in np.argsort(-base.threat_rank(pp))[:20]]
    print(f'{m:14s} {corridor:7.0f}k {lateral_width(th):8.0f}k {sc:6.2f} {reach:9.1f}k '
          f'{sep:8.1f}k {reach/max(sep,1e-9):9.1f} {np.median(touch):7.0f} of {len(base.menu)}')

print("\nfraction of routes a K=3 best-open laydown leaves COMPLETELY untouched:")
for m in ("kgd_gvardeysk","ukraine","narva","fulda"):
    th=load_vec_theatre(P%m); sc=lateral_width(th)/ref
    base=ConcealBase(P%m, terrain=terrain_v2(conceal_reach=0.43), range_scale=sc,
                     spacing_km=2.0*sc, standoff_km=4.0*sc)
    pp=base.lethality(resample_field(base.coords,5100),hidden_leth=1.0)
    thr=base.threat_rank(pp); op=np.where(~base.concealed)[0]
    L=op[np.argsort(-thr[op])[:3]]
    g=ConcealDyn(base,pp,L,w=2)
    per=g.stepdmg.mean(axis=0)                       # expected damage per route
    free=(per<0.02*per.max()).sum()
    print(f'  {m:14s} {free:2d}/{g.R} routes are free   best route {per.min():.5f}'
          f'   worst {per.max():.5f}   optimum {g.episodic(T=40):.5f}')
