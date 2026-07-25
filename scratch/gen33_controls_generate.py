"""gen33 METRIC 3 CONTROLS: generate forces under doctored briefs (live workbench, concurrent).

- scrambled: terrain class LABELS permuted in the brief by a 3-cycle (field->open->forest->field)
  in BOTH the composition line and the physics table, so every property the model reads is
  attached to the wrong label while the statistics are preserved. Emitted zones are resolved on
  the TRUE map: if the model reasons from terrain properties, its forces must DEGRADE materially.
- renamed: the real theatre name replaced by a neutral codename (no other change): forces must
  NOT change materially (no memorised-geography dependence).

kgd, both phases, both models, n=8 per cell per control (the same population size as the main
run). Saved under models/runs/gen33_forces_controls/.
"""
import importlib.util
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

_spec = importlib.util.spec_from_file_location("g33gen", "scripts/gen33_generate_force.py")
g33 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g33)

from src.envs.aerial_theatre_vec import lateral_width, load_vec_theatre  # noqa: E402
from src.redforce import serialise_theatre  # noqa: E402

MODELS = ("llama-3.3-70b", "qwen3-27b")
CYCLE = {"field": "open", "open": "forest", "forest": "field"}


def scramble(text):
    def sub(m):
        return "\0" + CYCLE[m.group(0)] + "\0"
    t = re.sub(r"\b(field|open|forest)\b", sub, text)
    return t.replace("\0", "")


def rename(text, th_name):
    return text.replace(th_name, "CORRIDOR ALPHA")


if __name__ == "__main__":
    t0 = time.time()
    a = SimpleNamespace(provider="openai", base="http://100.88.32.88:8080/v1",
                        key="iits-local-key", seed=0, max_tokens=3000, temperature=0.7,
                        timeout=900, n=8)
    lat_ref = lateral_width(load_vec_theatre(g33.THEATRES["kgd"]))
    ctx = g33.build_ctx("kgd", g33.THEATRES["kgd"], lat_ref)
    th_name = ctx["th"].name
    tasks = []
    for control in ("scrambled", "renamed"):
        for phase in ("single", "coordinated"):
            K = 1 if phase == "single" else 3
            system, user = serialise_theatre(ctx["th"], phase=phase, K=K,
                                             range_scale=ctx["scale"])
            u = scramble(user) if control == "scrambled" else rename(user, th_name)
            for model in MODELS:
                for j in range(a.n):
                    tasks.append({"model": model, "theatre": f"kgd-{control}", "phase": phase,
                                  "j": j, "K": K, "system": system, "user": u, "ctx": ctx})
    print(f"[controls] {len(tasks)} tasks", flush=True)
    records = []
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = [ex.submit(g33.run_task, t, a) for t in tasks]
        for f in as_completed(futs):
            r = f.result()
            records.append(r)
            print(f"  [{r['model']} {r['theatre']} {r['phase']} #{r['index']}] "
                  f"{'ok' if r.get('valid') else 'FAIL'} {r['latency_s']}s "
                  f"mode={r.get('mode', '-')}", flush=True)
    g33.summarise_and_save(records, SimpleNamespace(provider="openai"), None)
    Path("models/runs/gen33_forces_controls").mkdir(parents=True, exist_ok=True)
    groups = {}
    for r in records:
        groups.setdefault((r["model"], r["theatre"], r["phase"]), []).append(r)
    for (model, name, phase), rs in groups.items():
        valid = [r for r in rs if r.get("valid")]
        out = {"theatre": name, "phase": phase, "model": model, "n_valid": len(valid),
               "n_total": len(rs),
               "forces": [{"index": r["index"], "mode": r["mode"], "latency_s": r["latency_s"],
                           "force": r["force"], "resolved": r["resolved"]} for r in valid]}
        Path("models/runs/gen33_forces_controls",
             f"force_{model}_{name}_{phase}.json").write_text(json.dumps(out, indent=1))
    print(f"[done {time.time()-t0:.0f}s]")
