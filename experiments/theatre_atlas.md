# Theatre atlas: the four scored aerial theatres

- **status: REFERENCE (2026-07-25).** Kilian's 2026-07-25 decision limits the aerial line to these four theatres. Regenerate with `PYTHONPATH=. python scratch/theatre_atlas.py`; the maps themselves live under `data/maps/` and are gitignored, so this file is the committed record of what they contain. Oracle-only: no training, no model calls.
- **Companion:** `experiments/gen39_concealment.md` (terrain table v2 and the concealment mechanic these statistics are being reviewed for).

## At a glance

| theatre | box (km) | corridor | lateral width | range scale | sites (v2) | concealed |
|---|---|---|---|---|---|---|
| kgd_gvardeysk | 45 x 20 | KALININGRAD -> GVARDEYSK, 35.5 km | 28.2 km | 1.00 | 200 | **26%** |
| ukraine | 46 x 90 | DNIPRO -> ZAPORIZHZHIA, 69.9 km | 57.5 km | 2.04 | 946 | **12%** |
| narva | 107 x 61 | KOHTLA-JARVE -> KINGISEPP, 75.6 km | 64.0 km | 2.27 | 1,160 | **61%** |
| fulda | 124 x 112 | POINT ALPHA -> FRANKFURT, 109.9 km | 163.1 km | 5.79 | 3,431 | **51%** |

*Range scale* is the weapon-range multiplier the game applies so that the fraction of the corridor a team covers is comparable across maps, taken relative to Kaliningrad (lateral width 28.2 km). A 2.5 km system on Kaliningrad and a 14.5 km system on Fulda contest the same share of the width.

## kgd_gvardeysk

Kaliningrad -> Gvardeysk along the Pregolya. The reference theatre: a short, narrow corridor with the city as a line-of-sight wall at the mouth and open farmland beyond. Every banked aerial number was produced here.

| property | value |
|---|---|
| base | KALININGRAD at (5.9, 12.2) km |
| target | GVARDEYSK at (40.7, 5.4) km |
| box | 45.188 x 19.881 km (898 km2) |
| bounding box (lon/lat) | W 20.420, S 54.600, E 21.120, N 54.780 |
| projection | not recorded (pre-2026-07-22 format) |
| corridor length | 35.5 km |
| lateral width | 28.2 km |
| range scale | 1.00 (Kaliningrad = 1.00) |
| load time | 0.0 s |

**Terrain by area** (59778 samples at 0.12 km, priority order as the game classifies):

| class | share of area | polygons | ring vertices |
|---|---|---|---|
| open | 40.3% | - | - |
| field | 30.5% | 388 | 4,690 |
| forest | 15.8% | 102 | 1,635 |
| urban | 9.0% | 299 | 2,653 |
| water | 4.4% | 68 | 926 |

**Candidate emplacement sites** (2.0 km grid, 4.0 km terminal standoff):

| table | total | by class |
|---|---|---|
| v1 (banked) | 185 | field 63, forest 38, open 84 |
| v2 (gen39) | 200 | field 63, forest 38, open 84, urban 15 |

## ukraine

Dnipro -> Zaporizhzhia along the river. A long north-south corridor, farmland dominated, heavily built up at both ends: the longest run of the four.

| property | value |
|---|---|
| base | DNIPRO at (16.3, 79.9) km |
| target | ZAPORIZHZHIA at (25.6, 10.7) km |
| box | 45.884 x 90.248 km (4141 km2) |
| bounding box (lon/lat) | W 34.800, S 47.750, E 35.450, N 48.550 |
| projection | EPSG:32636 |
| corridor length | 69.9 km |
| lateral width | 57.5 km |
| range scale | 2.04 (Kaliningrad = 1.00) |
| load time | 0.3 s |

**Terrain by area** (60200 samples at 0.26 km, priority order as the game classifies):

| class | share of area | polygons | ring vertices |
|---|---|---|---|
| field | 44.6% | 2,569 | 20,002 |
| open | 36.2% | - | - |
| urban | 7.6% | 1,681 | 11,274 |
| water | 7.1% | 221 | 2,734 |
| forest | 4.5% | 1,342 | 12,003 |

**Candidate emplacement sites** (2.0 km grid, 4.0 km terminal standoff):

| table | total | by class |
|---|---|---|
| v1 (banked) | 871 | field 441, forest 34, open 396 |
| v2 (gen39) | 946 | field 441, forest 34, open 396, urban 75 |

## narva

Kohtla-Jarve -> Kingisepp across the Narva river border. Forest dominated with a natural river pinch, and the Gulf of Finland closing the north flank.

| property | value |
|---|---|
| base | KOHTLA-JARVE at (4.5, 27.9) km |
| target | KINGISEPP at (80.1, 25.6) km |
| box | 106.809 x 60.873 km (6502 km2) |
| bounding box (lon/lat) | W 27.200, S 59.150, E 29.100, N 59.680 |
| projection | EPSG:32635 |
| corridor length | 75.6 km |
| lateral width | 64.0 km |
| range scale | 2.27 (Kaliningrad = 1.00) |
| load time | 0.2 s |

**Terrain by area** (59940 samples at 0.33 km, priority order as the game classifies):

| class | share of area | polygons | ring vertices |
|---|---|---|---|
| forest | 43.6% | 1,169 | 18,241 |
| open | 23.2% | - | - |
| sea | 22.5% | 1 | 542 |
| field | 5.9% | 538 | 6,152 |
| water | 3.5% | 137 | 2,952 |
| urban | 1.3% | 462 | 3,226 |

**Candidate emplacement sites** (2.0 km grid, 4.0 km terminal standoff):

| table | total | by class |
|---|---|---|
| v1 (banked) | 1140 | field 80, forest 692, open 368 |
| v2 (gen39) | 1160 | field 80, forest 692, open 368, urban 20 |

## fulda

Point Alpha -> Frankfurt, the Fulda Gap. The Cold War invasion axis, flanked by the forested Vogelsberg and Rhoen uplands. Much the largest and the most detailed.

| property | value |
|---|---|
| base | POINT ALPHA at (105.8, 87.7) km |
| target | FRANKFURT at (16.8, 23.3) km |
| box | 123.882 x 111.733 km (13842 km2) |
| bounding box (lon/lat) | W 8.450, S 49.900, E 10.200, N 50.900 |
| projection | EPSG:32632 |
| corridor length | 109.9 km |
| lateral width | 163.1 km |
| range scale | 5.79 (Kaliningrad = 1.00) |
| load time | 4.2 s |

**Terrain by area** (60114 samples at 0.48 km, priority order as the game classifies):

| class | share of area | polygons | ring vertices |
|---|---|---|---|
| forest | 43.1% | 4,748 | 81,082 |
| field | 24.7% | 39,593 | 284,294 |
| open | 24.0% | - | - |
| urban | 7.9% | 3,370 | 36,600 |
| water | 0.4% | 223 | 4,313 |
| alpine | 0.0% | 9 | 77 |

**Candidate emplacement sites** (2.0 km grid, 4.0 km terminal standoff):

| table | total | by class |
|---|---|---|
| v1 (banked) | 3159 | field 853, forest 1487, open 819 |
| v2 (gen39) | 3431 | field 853, forest 1487, open 819, urban 272 |

## Notes for the terrain review

- **The four maps form a gradient in exactly the variable table v2 introduces.** The share of candidate sites that sit on CONCEALED ground (forest or urban, the classes that do not give themselves away when they engage) runs ukraine 12%, kgd 26%, fulda 51%, narva 61%. That is not four repetitions of one theatre: it is a designed axis running from a corridor where hiding is barely available to one where most of the ground conceals, and it is the natural held-out structure for the concealment act.
- **Fulda's range scale is the outlier and needs a decision.** Scaling by lateral width to keep the coverage fraction comparable turns a 2.5 km system into a 14.5 km one there, which is a different weapon class in everything but name. Either accept the abstraction and say so, or cap the scale and accept that Fulda is a lower-coverage theatre. This must be settled before any cross-map claim.
- **The exact interdiction matrix does not survive multi-team games on the big maps.** It is routes x C(sites, K). At K=1 that is fine everywhere (Fulda: 3,431 columns), but K=3 on Fulda is C(3431,3), about 6.7 billion columns. The gen33 scoring semantics, where a force induces a soft site prior rather than a hard K-subset, scales fine; the exact-LP path does not. Any multi-team screen on Narva or Fulda uses the soft-prior semantics or a coarser site grid, and says which.
- All four maps are covered by the terrain table: every polygon class present is one the game models, so the loader drops nothing. (The four maritime theatres that were dropped from the line carry `land`, `island` and `coast` layers that the table does not model, which is why they were unusable for anything scored.)
- Narva is the only one of the four with a `sea` layer, and it is a single polygon: the Gulf of Finland closing the north flank. It is non-emplaceable and does not block line of sight.
- Alpine appears in the Narva and Fulda fetches as an empty or near-empty layer, so high-terrain walls are not in play on any of the four.
- Under table v2 the urban class becomes emplaceable, which is where the extra sites in the v2 row come from; the forest and urban sites are the concealed ones that do not reveal themselves when they engage.
- Fulda is much the largest and most detailed map and is correspondingly the most expensive to load and to build games on; budget for that before using it in a sweep.

