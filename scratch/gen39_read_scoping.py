#!/usr/bin/env python3
"""gen39 step 1 read-out (see experiments/gen39_concealment.md RESULTS).

    PYTHONPATH=. python scratch/gen39_read_scoping.py
"""
import json, numpy as np, collections
rows=json.load(open('models/runs/gen39_screen2.json'))
P=lambda r:r['persistent']
med=lambda v: float(np.median(v)) if len(v) else float('nan')
mp=lambda r: r['tag'].split('x')[0]

print('=== WHY IS FULDA (and narva) DEGENERATE? ===')
print(f'{"map":14s} {"R":>4s} {"H":>5s} {"phi":>5s} {"opt":>8s} {"cap":>8s} | {"opt<5e-3":>8s} {"cap>0.9":>8s}')
for m in ('kgd_gvardeysk','ukraine','narva','fulda'):
    s=[r for r in rows if mp(r)==m]
    lo=sum(1 for r in s if P(r)['opt']<5e-3); hi=sum(1 for r in s if P(r)['cap']>0.90)
    print(f'{m:14s} {med([r["R"] for r in s]):4.0f} {med([r["H"] for r in s]):5.0f} {med([r["phi"] for r in s]):5.2f} '
          f'{med([P(r)["opt"] for r in s]):8.5f} {med([P(r)["cap"] for r in s]):8.4f} | {100*lo/len(s):7.0f}% {100*hi/len(s):7.0f}%')

print('\n  fulda by K and kind (opt):')
for K in (1,2,3,4,6):
    print(f'   K{K}: ' + '  '.join(f'{k}={med([P(r)["opt"] for r in rows if mp(r)=="fulda" and r["K"]==K and r["kind"]==k]):.5f}'
                                   for k in ('open','mixed','hidden','random')))

print('\n=== SECTION 3 REDONE, NON-DEGENERATE CELLS ONLY, PER MAP ===')
print(f'{"map":14s} {"reach":>5s} | {"n open":>6s} {"n hid":>6s} | {"vs omniscient":>13s} {"vs observing":>12s}')
for m in ('kgd_gvardeysk','ukraine','narva'):
    for cr in (0.43,0.65,0.85):
        so=[r for r in rows if mp(r)==m and r['kind']=='open' and abs(r['conceal_reach']-cr)<1e-6
            and r['K']>=2 and not P(r)['degenerate']]
        sh=[r for r in rows if mp(r)==m and r['kind']=='hidden' and abs(r['conceal_reach']-cr)<1e-6
            and r['K']>=2 and not P(r)['degenerate']]
        if len(so)<10 or len(sh)<10: 
            print(f'{m:14s} {cr:5.2f} | {len(so):6d} {len(sh):6d} |  (too few hidden cells survive)')
            continue
        print(f'{m:14s} {cr:5.2f} | {len(so):6d} {len(sh):6d} | '
              f'{med([P(r)["opt"] for r in sh])/med([P(r)["opt"] for r in so]):13.2f} '
              f'{med([P(r)["revealed"] for r in sh])/med([P(r)["revealed"] for r in so]):12.2f}')

print('\n  NOTE: hidden laydowns are DROPPED as degenerate far more often than open ones, so any')
print('  hidden-vs-open ratio computed only on survivors is SELECTED. Survival rates:')
for m in ('kgd_gvardeysk','ukraine','narva','fulda'):
    print(f'   {m:14s} ' + '  '.join(f'{k}={100*sum(1 for r in rows if mp(r)==m and r["kind"]==k and not P(r)["degenerate"])/max(1,sum(1 for r in rows if mp(r)==m and r["kind"]==k)):3.0f}%'
                                     for k in ('open','mixed','hidden','random')))
print('\n  Honest form: score every laydown INCLUDING the ones a knowing defender walks around,')
print('  since "walk-around-able" is a real property of a hidden force, not a missing datum:')
print(f'{"map":14s} {"reach":>5s} | {"vs omniscient":>13s} {"vs observing":>12s}')
for m in ('kgd_gvardeysk','ukraine','narva'):
    for cr in (0.43,0.65,0.85):
        so=[r for r in rows if mp(r)==m and r['kind']=='open' and abs(r['conceal_reach']-cr)<1e-6 and r['K']>=2]
        sh=[r for r in rows if mp(r)==m and r['kind']=='hidden' and abs(r['conceal_reach']-cr)<1e-6 and r['K']>=2]
        print(f'{m:14s} {cr:5.2f} | {med([P(r)["opt"] for r in sh])/max(med([P(r)["opt"] for r in so]),1e-9):13.2f} '
              f'{med([P(r)["revealed"] for r in sh])/med([P(r)["revealed"] for r in so]):12.2f}')
