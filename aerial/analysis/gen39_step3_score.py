"""gen39 step 3: score the 12-run batch against the PRE-REGISTERED clauses (ledger, launch
record). Validation-selected checkpoint per run; per-cell values; no averaging-away."""
import json, glob, numpy as np
from pathlib import Path
O = Path("models/runs/gen39_step3")

def load(tag, seed):
    p = O / f"{tag}_seed{seed}.json"
    return json.loads(p.read_text()) if p.exists() else None

def sel(run):
    """validation-selected checkpoint = lowest VAL ratio; returns (sortie, per-cell means)."""
    h = run["history"]
    best = min(h, key=lambda r: r["val"])
    return best["sortie"], np.array(best["cells"], float), best

ARMS = ["llm", "random", "heuristic", "llmblind"]
sel_rows = {}
print(f'{"arm":10s} {"seed":>4s} {"sel@":>6s} {"VAL":>5s} | per-cell held-out damage')
for a in ARMS:
    for s in (0, 1, 2):
        r = load(a, s)
        if not r: continue
        srt, cells, rec = sel(r)
        sel_rows[(a, s)] = cells
        print(f'{a:10s} {s:4d} {srt:6d} {rec["val"]:5.2f} | ' + " ".join(f"{c:.4f}" for c in cells))

# reference rows per cell (cap = static equilibrium, obs = best observing rule, opt = optimum)
r0 = load("llm", 0)
names = sorted(r0["refs"])
cells_ref = {}
for i, cellnames in enumerate([[n for n in names if f"te{6100+i}" in n] for i in range(6)]):
    cells_ref[i] = dict(
        cap=np.mean([r0["refs"][n]["cap"] for n in cellnames]),
        obs=np.mean([r0["refs"][n]["obs"] for n in cellnames]),
        opt=np.mean([r0["refs"][n]["opt"] for n in cellnames]))
print("\nper-cell reference rows (mean over the cell's 4 enemies):")
print("cell  " + "  ".join(f"{i}" for i in range(6)))
for k in ("cap", "obs", "opt"):
    print(f"{k:5s} " + " ".join(f"{cells_ref[i][k]:.4f}" for i in range(6)))

print("\n=== PRIMARY: llm below BOTH controls, >=4/6 cells AND pooled, >=2/3 seeds ===")
seeds_pass = 0
for s in (0, 1, 2):
    L, R, H = sel_rows[("llm", s)], sel_rows[("random", s)], sel_rows[("heuristic", s)]
    vs_r = int((L < R).sum()); vs_h = int((L < H).sum())
    both = int(((L < R) & (L < H)).sum())
    pooled = L.mean() < R.mean() and L.mean() < H.mean()
    ok = both >= 4 and pooled
    seeds_pass += ok
    print(f'  seed {s}: llm {L.mean():.4f} | random {R.mean():.4f} (beats {vs_r}/6) | '
          f'heuristic {H.mean():.4f} (beats {vs_h}/6) | both {both}/6 pooled {pooled} -> '
          f'{"PASS" if ok else "FAIL"}')
print(f'  VERDICT: {seeds_pass}/3 seeds -> PRIMARY {"PASS" if seeds_pass >= 2 else "FAIL"}')

print("\n=== pooled arm means (validation-selected) ===")
for a in ARMS:
    v = [sel_rows[(a, s)].mean() for s in (0, 1, 2) if (a, s) in sel_rows]
    print(f'  {a:10s} {np.mean(v):.4f}  (seeds: ' + " ".join(f"{x:.4f}" for x in v) + ")")

print("\n=== REPORTED: sighted vs blind (the concealment channel's worth to a trained policy) ===")
sight = np.mean([sel_rows[("llm", s)].mean() for s in (0, 1, 2)])
blind = np.mean([sel_rows[("llmblind", s)].mean() for s in (0, 1, 2)])
print(f'  sighted {sight:.4f} vs blinded {blind:.4f} -> channel worth {blind/sight:.2f}x')
for s in (0, 1, 2):
    a, b = sel_rows[("llm", s)], sel_rows[("llmblind", s)]
    print(f'  seed {s}: sighted {a.mean():.4f} blind {b.mean():.4f} ({b.mean()/a.mean():.2f}x), '
          f'blind-beaten-by-sighted on {int((a<b).sum())}/6 cells')

print("\n=== REPORTED: vs the static cap and the best observing rule (per cell, pooled seeds) ===")
capv = np.array([cells_ref[i]["cap"] for i in range(6)])
obsv = np.array([cells_ref[i]["obs"] for i in range(6)])
for a in ARMS:
    M = np.mean([sel_rows[(a, s)] for s in (0, 1, 2)], axis=0)
    print(f'  {a:10s} below-cap {int((M<capv).sum())}/6  below-observing-rule {int((M<obsv).sum())}/6'
          f'  mean/cap {np.mean(M/capv):.2f}x  mean/obs {np.mean(M/obsv):.2f}x')
