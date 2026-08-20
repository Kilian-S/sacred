#!/usr/bin/env python3
"""gen39 step 1 read-out (see experiments/gen39_concealment.md RESULTS).

    PYTHONPATH=. python analysis/gen39_read_operating_point.py
"""
import json, numpy as np
rows=json.load(open('models/runs/gen39_screen2.json'))
P=lambda r:r['persistent']; med=lambda v: float(np.median(v)) if len(v) else float('nan')
mp=lambda r: r['tag'].split('x')[0]
print('CANDIDATE OPERATING POINTS on the primary theatre (kgd), K=3\n')
print(f'{"cr":>5s} {"rm":>4s} {"hl":>4s} | {"real":>5s} {"G1":>5s} {"G2":>5s} {"sight":>6s} | '
      f'{"hid/open (obs)":>14s} {"hid/open (omni)":>15s}')
for cr in (0.43,0.65,0.85):
    for rm in (0.7,1.0,1.3):
        for hl in (0.4,0.6,0.8,1.0):
            s=[r for r in rows if mp(r)=='kgd_gvardeysk' and r['K']==3 and r['hidden_leth']==hl
               and abs(r['conceal_reach']-cr)<1e-6 and f'x{rm}' in r['tag']]
            if not s: continue
            ok=[r for r in s if not P(r)['degenerate']]
            om=[r for r in s if r['kind']=='open']; hd=[r for r in s if r['kind']=='hidden']
            print(f'{cr:5.2f} {rm:4.1f} {hl:4.1f} | {100*len(ok)/len(s):4.0f}% '
                  f'{med([P(r)["G1"] for r in ok]):5.2f} {med([P(r)["G2"] for r in ok]):5.2f} '
                  f'{med([P(r)["G_conceal"] for r in ok if r["kind"] in ("open","mixed")]):5.2f}x | '
                  f'{med([P(r)["revealed"] for r in hd])/med([P(r)["revealed"] for r in om]):14.2f} '
                  f'{med([P(r)["opt"] for r in hd])/max(med([P(r)["opt"] for r in om]),1e-9):15.2f}')
