"""Scores the gen39 step-5 batch against its pre-registered clauses, one validation-selected
checkpoint per run, reported per cell."""
import json, numpy as np
from pathlib import Path
O = Path("models/runs/gen39_step5")
ARMS = ["llm16", "local16", "random16", "tuned"]

def sel(arm, s):
    r = json.loads((O / f"{arm}_seed{s}.json").read_text())
    b = min(r["history"], key=lambda h: h["val"])
    return b, r["refs"]

rows = {}
print(f'{"arm":10s} {"seed":>4s} {"sel@":>6s} {"VAL":>5s} | per-cell held-out damage')
for a in ARMS:
    for s in (0, 1, 2):
        b, _ = sel(a, s)
        rows[(a, s)] = np.array(b["cells"], float)
        print(f'{a:10s} {s:4d} {b["sortie"]:6d} {b["val"]:5.2f} | '
              + " ".join(f"{c:.4f}" for c in rows[(a, s)]))

print("\n=== pooled arm means (validation-selected) ===")
pooled = {}
for a in ARMS:
    v = [rows[(a, s)].mean() for s in (0, 1, 2)]
    pooled[a] = float(np.mean(v))
    print(f'  {a:10s} {pooled[a]:.4f}   (seeds ' + ' '.join(f'{x:.4f}' for x in v) + ')')

print("\n=== PRIMARY: llm16 below TUNED on >=4/6 cells AND pooled, on 2/2 seeds ===")
ok = 0
for s in (0, 1):
    L, T = rows[("llm16", s)], rows[("tuned", s)]
    c = int((L < T).sum()); p = L.mean() < T.mean()
    good = c >= 4 and p
    ok += good
    print(f'  seed {s}: llm16 {L.mean():.4f} vs tuned {T.mean():.4f}, beats {c}/6, pooled {p}'
          f' -> {"PASS" if good else "FAIL"}')
print(f'  VERDICT: {ok}/3 seeds -> PRIMARY {"PASS" if ok >= 2 else "FAIL"}')

print("\n=== SECONDARY: llm16 vs the matched-budget search controls ===")
for ctrl in ("local16", "random16"):
    tot = 0
    for s in (0, 1, 2):
        L, C = rows[("llm16", s)], rows[(ctrl, s)]
        c = int((L < C).sum()); p = L.mean() < C.mean()
        tot += (c >= 4 and p)
        print(f'  vs {ctrl} seed {s}: {L.mean():.4f} vs {C.mean():.4f}, beats {c}/6 '
              f'-> {"PASS" if (c>=4 and p) else "FAIL"}')
    print(f'  vs {ctrl}: {tot}/3 seeds')

print("\n=== AMBIGUITY TRIGGER (pinned pre-launch: any pair within 10% pooled, or a 1-1 split) ===")
amb = []
for i, a in enumerate(ARMS):
    for b in ARMS[i+1:]:
        d = abs(pooled[a] - pooled[b]) / max(pooled[a], pooled[b])
        if d < 0.10:
            amb.append(f"{a}~{b} ({100*d:.1f}%)")
print("  pairs within 10%:", ", ".join(amb) if amb else "none")
print(f'  -> third seed {"WARRANTED" if amb else "not warranted"}')

print("\n=== step-3 comparison (the negative this tested) ===")
print(f'  step 3: llm 0.1079 / random 0.1520 / heuristic 0.0915 (weak test set)')
print(f'  step 5: ' + " / ".join(f"{a} {pooled[a]:.4f}" for a in ARMS) + "  (STRONG test set)")
_, refs = sel("llm16", 0)
names = sorted(refs)
caps = np.array([np.mean([refs[n]["cap"] for n in names if f"te{6100+i}" in n]) for i in range(6)])
obsv = np.array([np.mean([refs[n]["obs"] for n in names if f"te{6100+i}" in n]) for i in range(6)])
print("\n=== vs the oracle rows (pooled seeds, per cell) ===")
for a in ARMS:
    M = np.mean([rows[(a, s)] for s in (0, 1, 2)], axis=0)
    print(f'  {a:10s} below-cap {int((M<caps).sum())}/6  below-observing-rule '
          f'{int((M<obsv).sum())}/6  mean/cap {np.mean(M/caps):.2f}x')
