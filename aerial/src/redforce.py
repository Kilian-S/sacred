"""gen33_llm_adversary: the LLM red-force I/O contract.

- FORCE_SCHEMA: the frozen structured-output schema (guided JSON; the LLM emits a whole force).
- serialise_theatre: real terrain + physics table + mission + doctrine -> a prose brief.
- dry_force: a schema-valid synthetic force for pipeline validation without a live model.
- resolve_force_to_sites: map an emitted force onto concrete candidate sites + doctrine params
  (PROVISIONAL enemy semantics; the exact phase-2 joint-doctrine mapping is pinned by an oracle
  screen during the build, per the ledger). Firepower is terrain-set here, never force-set.
"""
from __future__ import annotations

import numpy as np

from src.envs.aerial_theatre_vec import TERRAIN, load_vec_theatre  # noqa: F401 (re-export path)

ARCHETYPES = ["sniper_overwatch", "ambusher", "anticipator", "blocker", "forward_picket"]
REGIONS = ["near_base", "mid_corridor", "near_target_standoff", "chokepoint"]
EMPLACE_TERRAIN = [k for k, v in TERRAIN.items() if v["emplace"] and k != "open"]  # field, forest

# The frozen guided-JSON schema. team_id/team_role carry the phase-2 coordination; phase-1 forces
# omit them (a single agent). decisiveness/memory bin the gen31 tau/window.
FORCE_SCHEMA = {
    "type": "object",
    "properties": {
        "agents": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "archetype": {"type": "string", "enum": ARCHETYPES},
                    "emplacement_zone": {
                        "type": "object",
                        "properties": {
                            "terrain": {"type": "string", "enum": EMPLACE_TERRAIN + ["open"]},
                            "region": {"type": "string", "enum": REGIONS},
                        },
                        "required": ["terrain", "region"],
                    },
                    "doctrine": {
                        "type": "object",
                        "properties": {
                            "punish_pattern": {"type": "number"},
                            "anticipate_flight": {"type": "number"},
                            "hold_static": {"type": "number"},
                        },
                        "required": ["punish_pattern", "anticipate_flight", "hold_static"],
                    },
                    "decisiveness": {"type": "string", "enum": ["decisive", "balanced", "hedged"]},
                    "memory": {"type": "integer", "minimum": 1, "maximum": 3},
                    "terrain_preference": {"type": "string", "enum": ["concealment", "reach"]},
                    "risk": {"type": "string", "enum": ["forward", "deep"]},
                    "team_id": {"type": "integer", "minimum": 0},
                    "team_role": {"type": "string",
                                  "enum": ["bait", "block", "cover", "anchor"]},
                    "rationale": {"type": "string"},
                },
                "required": ["archetype", "emplacement_zone", "doctrine", "rationale"],
            },
        }
    },
    "required": ["agents"],
}

TAU_BIN = {"decisive": 0.05, "balanced": 0.10, "hedged": 0.20}


def force_schema(terrain: dict | None = None) -> dict:
    """FORCE_SCHEMA for the table in force. The module constant's terrain enum was computed from
    the v1 table at import, so under v2 a model could never choose URBAN although v2 makes it
    emplaceable (and it is the strongest cover class). gen39 step 2 must pass its terrain table
    here; the default returns the frozen v1 schema unchanged (gen33's contract)."""
    if terrain is None:
        return FORCE_SCHEMA
    import copy
    s = copy.deepcopy(FORCE_SCHEMA)
    empl = [k for k, v in terrain.items() if v["emplace"] and k != "open"]
    s["properties"]["agents"]["items"]["properties"]["emplacement_zone"]["properties"][
        "terrain"]["enum"] = empl + ["open"]
    return s


def terrain_composition(th, n=44) -> dict:
    xs = np.linspace(1, th.W - 1, n)
    ys = np.linspace(1, th.H - 1, n)
    cnt: dict = {}
    for x in xs:
        for y in ys:
            c = th.classify((x, y))
            cnt[c] = cnt.get(c, 0) + 1
    tot = sum(cnt.values())
    return {k: 100.0 * v / tot for k, v in sorted(cnt.items(), key=lambda kv: -kv[1])}


def _physics_table_text(range_scale: float, terrain: dict | None = None) -> str:
    """The ground truth handed to the model. HIDING and SIGHT-BLOCKING are stated SEPARATELY.

    gen33's brief conflated them ("it conceals you (blocks line of sight)") off the single `los`
    flag, which (a) described a concealment mechanic the simulator did not implement and (b) is
    not what either flag means under v2. `reveal` is whether engaging gives your position away;
    `los` is whether the ground masks sight lines crossing it. Woodland now does the first and
    not the second. Disclosed in the gen33 ledger's terrain-mismatch appendix."""
    rows = []
    for k, v in (terrain or TERRAIN).items():
        hides = v.get("reveal", True) is False
        if v["emplace"]:
            tail = (" You stay HIDDEN here: engaging does not give your position away."
                    if hides else " Engaging here GIVES YOUR POSITION AWAY to the drones.")
            if v["los"]:
                tail += " This ground also masks sight lines crossing it."
            rows.append(f"  - {k}: your weapons reach {v['r_km'] * range_scale:.1f} km, "
                        f"lethality {v['p_max']:.2f} at the centre.{tail}")
        else:
            note = "masks sight lines crossing it" if v["los"] else "open, no cover"
            rows.append(f"  - {k}: you CANNOT emplace here ({note}).")
    return "\n".join(rows)


def serialise_theatre(th, phase: str = "single", K: int = 1, range_scale: float = 1.0,
                      terrain: dict | None = None) -> tuple:
    """Return (system, user) messages briefing the LLM to design the red force. Physics shown.

    terrain: the table actually in force. Defaults to v1 so gen33's briefs reproduce verbatim;
    gen39 passes terrain_v2(...) so the model is briefed on the world it is playing in."""
    comp = terrain_composition(th)
    comp_txt = ", ".join(f"{k} {v:.0f}%" for k, v in comp.items())
    coordinated = phase == "coordinated"
    system = (
        "You are the air-defence planner for a RED force opposing an enemy drone resupply mission. "
        "Reading the terrain and your doctrine, you design where your air-defence teams sit and how "
        "they behave. You know your own weapons (the physics table is given). Emit ONLY the "
        "structured force."
    )
    user = f"""THEATRE: {th.name}, a {th.W:.0f} x {th.H:.0f} km corridor.
The enemy flies {3} drones together (a stacked flight) from base ({th.base[0]:.0f},{th.base[1]:.0f} km)
to target ({th.target[0]:.0f},{th.target[1]:.0f} km). Their mission FAILS if you down at least one
drone. Terrain along the corridor: {comp_txt}.

YOUR WEAPONS (fixed by the ground you sit on; you do not choose their power):
{_physics_table_text(range_scale, terrain)}

THE CONTEST IS DYNAMIC. The enemy watches nothing you place, but it reacts to its own recent
routes: it is a habitual flier that also breaks its habit in the obvious way. A single fixed
ambush is trivially flown around; scattering teams naively is punished because the enemy anticipates
naive avoidance. Value comes from WHERE you concentrate (terrain that channels or conceals) and HOW
each team behaves over time: punish the enemy's repeated pattern, anticipate where a pattern-punished
flier escapes to, or hold a chokepoint.

YOUR DOCTRINE PER TEAM (a mix on the simplex, summing to ~1):
  - punish_pattern: aim where the enemy just flew (exploit habit).
  - anticipate_flight: aim where a habit-punished enemy will obviously flee to.
  - hold_static: sit on a fixed high-value spot regardless.

TASK: design a force of {K} air-defence team{'s' if K > 1 else ''}."""
    if coordinated:
        user += (
            "\nThey act as a COORDINATED group: give each a team_role (bait forces the enemy off "
            "the direct line, block covers the diversion, cover/anchor hold complementary ground) "
            "and a shared team_id, so the group's coverage is complementary, not redundant."
        )
    else:
        user += "\nA single team: choose its terrain, region, and doctrine."
    return system, user


def dry_force(K: int = 1, seed: int = 0, coordinated: bool = False) -> dict:
    """A schema-valid synthetic force (pipeline validation without a live model)."""
    rng = np.random.default_rng(seed)
    agents = []
    for i in range(K):
        arch = ARCHETYPES[i % len(ARCHETYPES)]
        d = rng.dirichlet([2, 1, 1])
        a = {
            "archetype": arch,
            "emplacement_zone": {"terrain": EMPLACE_TERRAIN[i % len(EMPLACE_TERRAIN)],
                                 "region": REGIONS[i % len(REGIONS)]},
            "doctrine": {"punish_pattern": round(float(d[0]), 2),
                         "anticipate_flight": round(float(d[1]), 2),
                         "hold_static": round(float(d[2]), 2)},
            "decisiveness": ["decisive", "balanced", "hedged"][i % 3],
            "memory": int(rng.integers(1, 4)),
            "terrain_preference": "concealment" if arch == "ambusher" else "reach",
            "risk": "forward" if arch == "forward_picket" else "deep",
            "rationale": f"dry synthetic {arch}",
        }
        if coordinated:
            a["team_id"] = 0
            a["team_role"] = ["bait", "block", "cover", "anchor"][i % 4]
        agents.append(a)
    return {"agents": agents}


def _region_mask(th, coords, region: str) -> np.ndarray:
    """Boolean mask over candidate sites in the given corridor region (along-axis thirds; the
    chokepoint is the narrowest lateral band of emplaceable terrain, approximated by mid-corridor)."""
    v = th.target - th.base
    u = v / (np.linalg.norm(v) + 1e-9)
    along = (coords - th.base) @ u
    span = float(v @ u)
    frac = np.clip(along / (span + 1e-9), 0, 1)
    if region == "near_base":
        return frac < 0.34
    if region == "near_target_standoff":
        return frac > 0.66
    return (frac >= 0.34) & (frac <= 0.66)      # mid_corridor and chokepoint


def resolve_force_to_sites(force: dict, th, coords, cls, exposure) -> dict:
    """PROVISIONAL: map each agent to a candidate SITE (best-exposure site of its terrain in its
    region) and doctrine params (normalised simplex, tau from decisiveness, window from memory).
    Returns {sites:[idx], doctrine:[(q_rep,q_flee,q_eq,tau,w)], validity notes}. The support of the
    emitted force is these sites; the enemy plays the gen31 doctrine over that support. The exact
    phase-2 joint semantics are pinned by an oracle screen before the live scoring is trusted."""
    cls = list(cls)
    sites, doctrine, notes = [], [], []
    for a in force["agents"]:
        terr = a["emplacement_zone"]["terrain"]
        region = a["emplacement_zone"]["region"]
        elig = np.array([i for i, c in enumerate(cls)
                         if c == terr or (terr == "open" and c in ("open", "field"))], dtype=int)
        if len(elig) == 0:
            elig = np.arange(len(cls))
            notes.append(f"{terr} not available; fell back to any terrain")
        rmask = _region_mask(th, coords, region)
        pool = elig[rmask[elig]] if rmask[elig].any() else elig
        site = int(pool[np.argmax(exposure[pool])])         # best-exposure site in the pool
        sites.append(site)
        d = a["doctrine"]
        tot = d["punish_pattern"] + d["anticipate_flight"] + d["hold_static"] or 1.0
        q = (d["punish_pattern"] / tot, d["anticipate_flight"] / tot, d["hold_static"] / tot)
        tau = TAU_BIN.get(a.get("decisiveness", "balanced"), 0.10)
        w = int(a.get("memory", 2))
        doctrine.append((q[0], q[1], q[2], tau, w))
    return {"sites": sites, "doctrine": doctrine, "notes": notes}
