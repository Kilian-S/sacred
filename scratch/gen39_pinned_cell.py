import json, numpy as np
rows=json.load(open('models/runs/gen39_screen2.json'))
P=lambda r:r['persistent']; med=lambda v: float(np.median(v)) if len(v) else float('nan')
sel=[r for r in rows if r['tag']=='kgd_gvardeyskx1.0cr0.85' and r['K']==3 and r['hidden_leth']==0.4]
print(f'PINNED CELL: kgd_gvardeysk, K=3, conceal_reach 0.85, range x1.0, hidden_leth 0.4  (n={len(sel)})\n')
print(f'{"laydown":8s} {"seed":>5s} | {"optimum":>8s} {"static cap":>10s} {"blind":>7s} {"observing":>9s} '
      f'{"G1":>5s} {"G2":>5s} {"sight":>6s}')
for k in ('open','mixed','hidden','random'):
    for r in [x for x in sel if x['kind']==k]:
        p=P(r)
        print(f'{k:8s} {r["seed"]:5d} | {p["opt"]:8.4f} {p["cap"]:10.4f} {p["blind"]:7.4f} '
              f'{p["revealed"]:9.4f} {p["G1"]:5.2f} {p["G2"]:5.2f} {p["G_conceal"]:5.2f}x')
print()
om=[r for r in sel if r['kind']=='open']; hd=[r for r in sel if r['kind']=='hidden']
print(f'hidden/open vs omniscient  {med([P(r)["opt"] for r in hd])/med([P(r)["opt"] for r in om]):.2f}')
print(f'hidden/open vs observing   {med([P(r)["revealed"] for r in hd])/med([P(r)["revealed"] for r in om]):.2f}')
print(f'mission-length curve (optimum per serial, open laydowns):')
print('   ' + '  '.join(f'T{t}={med([r["opt_curve"][t] for r in om]):.4f}' for t in ('10','20','40','80')))
