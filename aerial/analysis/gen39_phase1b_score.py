"""gen39 Phase 1b: score the 61-force population per model, on BOTH yardsticks, and report the
axis Phase 1a identified: threat vs PERFECT play (the curriculum predictor) alongside threat vs
the observing rule (what step 2 scored). Also: terrain posture mix and doctrine diversity."""
import json, os, collections
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v,"1")
import numpy as np
from pathlib import Path
from src.envs.aerial_conceal import resample_field
from analysis.gen39_compose import narva_base, place, doctrines_of, score_force, FIELDS, K

_CTX={}
def _init():
    _CTX["base"],_,_ = narva_base()
def _task(spec):
    j, force, field = spec
    base=_CTX["base"]
    pp=base.lethality(resample_field(base.coords, field), hidden_leth=1.0)
    o,b,ob,cov = score_force(base, pp, place(force, base, pp), doctrines_of(force))
    return j, field, (o, ob, cov)

if __name__ == "__main__":
    recs=[r for r in json.load(open("models/runs/gen39_compose/forces_big.json"))
          if r["force"] and not r["errors"] and len(r["force"]["agents"])==K]
    specs=[(i, r["force"], f) for i,r in enumerate(recs) for f in FIELDS]
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(9, initializer=_init) as P:
        out={}
        for j,f,v in P.imap_unordered(_task, specs, chunksize=6): out.setdefault(j,[]).append(v)
    rows=[]
    for j,r in enumerate(recs):
        a=np.array(out[j])
        terr=[x["emplacement_zone"]["terrain"] for x in r["force"]["agents"]]
        rows.append(dict(model=r["model"], j=r["j"], opt=float(np.median(a[:,0])),
                         obs=float(np.median(a[:,1])), cover=float(np.median(a[:,2])),
                         terr=terr, hidden=sum(t in ("forest","urban") for t in terr)/K))
    Path("models/runs/gen39_phase1b_scores.json").write_text(json.dumps(rows, indent=1))
    print(f'{"model":16s} {"n":>3s} {"vs PERFECT (curriculum axis)":>29s} {"vs observing":>13s} {"cover share":>12s}')
    for m in ("llama-3.3-70b","qwen3-27b"):
        s=[r for r in rows if r["model"]==m]
        print(f'{m:16s} {len(s):3d} {np.median([r["opt"] for r in s]):29.5f} '
              f'{np.median([r["obs"] for r in s]):13.4f} {np.mean([r["hidden"] for r in s]):11.0%}')
        top=sorted(s, key=lambda r:-r["opt"])[:5]
        print(f'    top-5 by irreducible threat: ' + " ".join(f'{r["opt"]:.4f}' for r in top)
              + f'  (their cover share {np.mean([r["hidden"] for r in top]):.0%})')
    allr=sorted(rows, key=lambda r:-r["opt"])
    print(f'\npopulation spread vs perfect play: max {allr[0]["opt"]:.4f}  '
          f'p75 {np.percentile([r["opt"] for r in rows],75):.4f}  median '
          f'{np.median([r["opt"] for r in rows]):.5f}  min {allr[-1]["opt"]:.5f}')
    print(f'heuristic reference (oracle-placed gen32, phase 1a): 0.0215 vs perfect play')
    strong=[r for r in rows if r["opt"]>=0.005]
    print(f'forces with irreducible threat >= 0.005: {len(strong)}/{len(rows)} '
          f'({collections.Counter(r["model"] for r in strong)})')
    print(f'correlation(cover share, irreducible threat) = '
          f'{np.corrcoef([r["hidden"] for r in rows],[r["opt"] for r in rows])[0,1]:+.2f}')
