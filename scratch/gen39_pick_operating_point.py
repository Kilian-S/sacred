import json, numpy as np, itertools
rows=json.load(open('models/runs/gen39_screen2.json'))
P=lambda r:r['persistent']; med=lambda v: float(np.median(v)) if len(v) else float('nan')
mp=lambda r: r['tag'].split('x')[0]
def rmof(r): return float(r['tag'].split('x')[1].split('cr')[0])
HL=sorted({r['hidden_leth'] for r in rows})   # derived from the data, never hardcoded
out=[]
for m in ('kgd_gvardeysk','ukraine','narva','fulda'):
    for K in (1,2,3,4,6):
        for cr in (0.43,0.65,0.85):
            for rm in (0.7,1.0,1.3):
                for hl in HL:
                    s=[r for r in rows if mp(r)==m and r['K']==K and r['hidden_leth']==hl
                       and abs(r['conceal_reach']-cr)<1e-6 and abs(rmof(r)-rm)<1e-9]
                    if len(s)<12: continue
                    ok=[r for r in s if not P(r)['degenerate']]
                    om=[r for r in s if r['kind']=='open']; hd=[r for r in s if r['kind']=='hidden']
                    if not om or not hd: continue
                    real=len(ok)/len(s)
                    g1=med([P(r)['G1'] for r in ok]) if ok else 0
                    g2=med([P(r)['G2'] for r in ok]) if ok else 0
                    sight=med([P(r)['G_conceal'] for r in ok if r['kind'] in ('open','mixed')]) if ok else 0
                    omni=med([P(r)['opt'] for r in hd])/max(med([P(r)['opt'] for r in om]),1e-9)
                    obs=med([P(r)['revealed'] for r in hd])/max(med([P(r)['revealed'] for r in om]),1e-9)
                    out.append(dict(m=m,K=K,cr=cr,rm=rm,hl=hl,real=real,g1=g1,g2=g2,sight=sight,
                                    omni=omni,obs=obs))
sel=[c for c in out if c['real']>=0.90 and c['g1']>=2.0 and c['g2']>=2.0 and c['sight']>=1.4]
sel.sort(key=lambda c:(abs(c['omni']-1.0)-0.5*(c['obs']-1.0)))
print(f'{"map":14s} {"K":>2s} {"cr":>5s} {"rm":>4s} {"hl":>4s} | {"real":>5s} {"G1":>5s} {"G2":>5s} '
      f'{"sight":>6s} | {"hid/open omni":>13s} {"hid/open obs":>12s}')
for c in sel[:14]:
    print(f'{c["m"]:14s} {c["K"]:2d} {c["cr"]:5.2f} {c["rm"]:4.1f} {c["hl"]:4.1f} | {100*c["real"]:4.0f}% '
          f'{c["g1"]:5.2f} {c["g2"]:5.2f} {c["sight"]:5.2f}x | {c["omni"]:13.2f} {c["obs"]:12.2f}')
print(f'\n{len(sel)} cells meet real>=90%, G1>=2, G2>=2, sight>=1.4x  (of {len(out)} swept)')
