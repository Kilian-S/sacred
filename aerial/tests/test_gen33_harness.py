"""The concurrent red-force generation harness: task fan-out and fallback parsing. The script is
loaded by path because ``scripts`` is not a package, and only the dry path is exercised."""
import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from src.redforce import dry_force

_spec = importlib.util.spec_from_file_location(
    "gen33_generate_force",
    Path(__file__).resolve().parents[1] / "scripts" / "gen33_generate_force.py")
g33 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g33)


def test_fallback_extraction_and_validation():
    force = dry_force(K=3, seed=3, coordinated=True)
    wrapped = ("<think>reasoning first</think>Here is the force:\n```json\n"
               + json.dumps(force) + "\n```")
    got = g33._extract_json(wrapped)
    assert got == force
    assert g33.validate_force(got) == []
    assert g33.validate_force({"agents": [{}]})       # a broken force reports problems


def test_concurrent_dry_tasks_validate_and_resolve():
    from src.envs.aerial_theatre_vec import lateral_width, load_vec_theatre
    lat_ref = lateral_width(load_vec_theatre(g33.THEATRES["kgd"]))
    ctx = g33.build_ctx("kgd", g33.THEATRES["kgd"], lat_ref)
    a = SimpleNamespace(provider="dry", seed=0)
    tasks = [{"model": ("dry-a", "dry-b")[j % 2], "theatre": "kgd", "phase": "coordinated",
              "j": j, "K": 3, "system": "", "user": "", "ctx": ctx} for j in range(4)]
    with ThreadPoolExecutor(max_workers=4) as ex:
        recs = list(ex.map(lambda t: g33.run_task(t, a), tasks))
    assert all(r["valid"] for r in recs)
    assert all(len(r["resolved"]["sites"]) == 3 for r in recs)
    assert all(r["latency_s"] >= 0 for r in recs)
