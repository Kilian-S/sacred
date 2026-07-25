#!/usr/bin/env python3
"""gen39 step 1 read-out (see experiments/gen39_concealment.md RESULTS).

    PYTHONPATH=. python scratch/gen39_read_screen.py
"""
import json, numpy as np, collections
rows=json.load(open('models/runs/gen39_screen2.json'))
P=lambda r:r['persistent']; F=lambda r:r['forgetful']
med=lambda v: float(np.median(v)) if len(v) else float('nan')
mp=lambda r: r['tag'].split('x')[0]
print(f'cells={len(rows)}  maps={sorted(set(map(mp,rows)))}\n')

print('=== 1. IS THERE ROOM FOR A LEARNED POLICY? (vs the omniscient optimum) ===')
print(f'{"map":14s} {"n":>5s} {"real game":>9s} {"G1 cap/opt":>10s} {"G2 rules/opt":>12s}')
for m in ('kgd_gvardeysk','ukraine','narva','fulda','ALL'):
    s=[r for r in rows if m=='ALL' or mp(r)==m]; ok=[r for r in s if not P(r)['degenerate']]
    print(f'{m:14s} {len(s):5d} {100*len(ok)/len(s):8.0f}% {med([P(r)["G1"] for r in ok]):10.2f} '
          f'{med([P(r)["G2"] for r in ok]):12.2f}')
ok=[r for r in rows if not P(r)['degenerate']]
g=[r for r in ok if P(r)['G1']>=2.0 and P(r)['G2']>=1.25]
print(f'\ngates (G1>=2.0, G2>=1.25) pass on {len(g)}/{len(ok)} real cells = {100*len(g)/len(ok):.0f}% '
      f'({100*len(g)/len(rows):.0f}% of the whole grid)')

print('\n=== 2. DOES CONCEALMENT DENY THE CHANNEL? (same defender, with vs without sight) ===')
print(f'{"enemy sits in":14s} {"n":>5s} {"blind":>7s} {"observing":>9s} {"sight worth":>11s}')
for k in ('open','mixed','random','hidden'):
    s=[r for r in rows if r['kind']==k and not P(r)['degenerate']]
    print(f'{k:14s} {len(s):5d} {med([P(r)["blind"] for r in s]):7.4f} {med([P(r)["revealed"] for r in s]):9.4f} '
          f'{med([P(r)["G_conceal"] for r in s]):10.2f}x')

print('\n=== 3. SHOULD THE ENEMY HIDE? (hidden as a share of open, K>=2) ===')
print(f'{"reach":>6s} | {"vs omniscient":>13s} {"vs observing":>12s}')
for cr in (0.43,0.65,0.85):
    o={k: (med([P(r)['opt'] for r in rows if r['kind']==k and abs(r['conceal_reach']-cr)<1e-6 and r['K']>=2]),
           med([P(r)['revealed'] for r in rows if r['kind']==k and abs(r['conceal_reach']-cr)<1e-6 and r['K']>=2]))
       for k in ('open','hidden')}
    print(f'{cr:6.2f} | {o["hidden"][0]/o["open"][0]:13.2f} {o["hidden"][1]/o["open"][1]:12.2f}')

print('\n=== 4. WHERE IS THE GAME REAL AND THE GAP WIDE? (operating point) ===')
print(f'{"map":14s} {"K":>2s} | {"real":>5s} {"G1":>5s} {"G2":>5s} {"sight":>6s} {"phi":>5s}')
cand=[]
for m in ('kgd_gvardeysk','ukraine','narva','fulda'):
    for K in (1,2,3,4,6):
        s=[r for r in rows if mp(r)==m and r['K']==K]
        ok=[r for r in s if not P(r)['degenerate']]
        if len(ok)<0.6*len(s): continue
        g1,g2 = med([P(r)['G1'] for r in ok]), med([P(r)['G2'] for r in ok])
        gc = med([P(r)['G_conceal'] for r in ok if r['kind'] in ('open','mixed')])
        cand.append((m,K,100*len(ok)/len(s),g1,g2,gc,med([r['phi'] for r in ok])))
for c in sorted(cand, key=lambda c:-(c[4]*c[5])):
    print(f'{c[0]:14s} {c[1]:2d} | {c[2]:4.0f}% {c[3]:5.2f} {c[4]:5.2f} {c[5]:5.2f}x {c[6]:5.2f}')

print('\n=== 5. MEMORY: what did the faithful form change? (paired, same cell) ===')
pk=[r for r in rows if not P(r)['degenerate'] and not F(r)['degenerate']]
for lab,f in (('omniscient optimum',lambda r:(F(r)['opt'],P(r)['opt'])),
              ('best blind rule',   lambda r:(F(r)['blind'],P(r)['blind'])),
              ('best observing rule',lambda r:(F(r)['revealed'],P(r)['revealed'])),
              ('G2 (rules/opt)',    lambda r:(F(r)['G2'],P(r)['G2']))):
    a=[f(r)[0] for r in pk]; b=[f(r)[1] for r in pk]
    print(f'  {lab:22s} window {med(a):7.4f} -> whole-mission {med(b):7.4f}  ({med(b)/med(a):.2f}x)')

print('\n=== 6. CROSS-CHECK: the SYMMETRIC-FOREST pair (archived; a property of those artefacts) ===')
try:
    a2=json.load(open('models/runs/gen39_screen2_symforest.json'))
    s1=json.load(open('models/runs/gen39_screen_symforest.json'))
    key=lambda r:(r['tag'],r['seed'],r['hidden_leth'],r['K'],r['kind'])
    m1={key(r):r for r in s1}
    n=d=0; worst=0.0
    for r in a2:
        o=m1.get(key(r))
        if o is None: continue
        n+=1
        for x,y in ((o['opt'],r['forgetful']['opt']),(o['cap'],r['forgetful']['cap']),
                    (o['blind'],r['forgetful']['blind']),(o['revealed'],r['forgetful']['revealed'])):
            worst=max(worst,abs(x-y)); d+=(x!=y)
    print(f'  {n} cells overlap; {d} value mismatches; largest absolute difference {worst:.2e}')
except FileNotFoundError:
    print('  (archived symmetric-forest artefacts not present)')
