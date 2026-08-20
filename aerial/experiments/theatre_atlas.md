# Theatre atlas: the four scored aerial theatres

Reference record, 2026-07-25. The aerial line uses these four theatres. Regenerate with `PYTHONPATH=. python analysis/theatre_atlas.py`. The map files live under `data/maps/` and are not committed, so this file is the record of what they contain. Oracle only: no training, no model calls.

## At a glance

| theatre | box (km) | corridor | lateral width | range scale | sites (v2) | concealed |
|---|---|---|---|---|---|---|
| kgd_gvardeysk | 45 x 20 | KALININGRAD -> GVARDEYSK, 35.5 km | 28.2 km | 1.00 | 200 | 26% |
| ukraine | 46 x 90 | DNIPRO -> ZAPORIZHZHIA, 69.9 km | 57.5 km | 2.04 | 946 | 12% |
| narva | 107 x 61 | KOHTLA-JARVE -> KINGISEPP, 75.6 km | 64.0 km | 2.27 | 1,160 | 61% |
| fulda | 124 x 112 | RHEIN-MAIN SW -> POINT ALPHA, 121.9 km | 163.1 km | 5.79 | 3,428 | 51% |

Range scale is the weapon-range multiplier applied so that the fraction of the corridor a team covers is comparable across maps, taken relative to Kaliningrad (lateral width 28.2 km). A 2.5 km system on Kaliningrad and a 14.5 km system on Fulda contest the same share of the width.

Concealed is the share of v2 candidate sites on forest or urban ground, the classes that do not reveal themselves when they engage. The ordering runs ukraine 12%, kgd 26%, fulda 51%, narva 61%.

Candidate emplacement sites are built on a 2.0 km grid with a 4.0 km terminal standoff. Table v1 admits field, forest and open; table v2 adds urban, which accounts for the difference between the two rows.

## kgd_gvardeysk

Kaliningrad to Gvardeysk along the Pregolya. A short, narrow corridor with the city as a line-of-sight wall at the mouth and open farmland beyond.

| property | value |
|---|---|
| base | KALININGRAD at (5.9, 12.2) km |
| target | GVARDEYSK at (40.7, 5.4) km |
| box | 45.188 x 19.881 km (898 km2) |
| bounding box (lon/lat) | W 20.420, S 54.600, E 21.120, N 54.780 |
| projection | not recorded (pre-2026-07-22 format) |
| corridor length | 35.5 km |
| lateral width | 28.2 km |
| range scale | 1.00 |
| load time | 0.1 s |

Terrain by area, 59778 samples at 0.12 km, in the priority order the game classifies:

| class | share of area | polygons | ring vertices |
|---|---|---|---|
| open | 40.3% | - | - |
| field | 30.5% | 388 | 4,690 |
| forest | 15.8% | 102 | 1,635 |
| urban | 9.0% | 299 | 2,653 |
| water | 4.4% | 68 | 926 |

| site table | total | by class |
|---|---|---|
| v1 | 185 | field 63, forest 38, open 84 |
| v2 | 200 | field 63, forest 38, open 84, urban 15 |

## ukraine

Dnipro to Zaporizhzhia along the river. A long north-south corridor, farmland dominated, built up at both ends, the longest run of the four.

| property | value |
|---|---|
| base | DNIPRO at (16.3, 79.9) km |
| target | ZAPORIZHZHIA at (25.6, 10.7) km |
| box | 45.884 x 90.248 km (4141 km2) |
| bounding box (lon/lat) | W 34.800, S 47.750, E 35.450, N 48.550 |
| projection | EPSG:32636 |
| corridor length | 69.9 km |
| lateral width | 57.5 km |
| range scale | 2.04 |
| load time | 0.4 s |

Terrain by area, 60200 samples at 0.26 km:

| class | share of area | polygons | ring vertices |
|---|---|---|---|
| field | 44.6% | 2,569 | 20,002 |
| open | 36.2% | - | - |
| urban | 7.6% | 1,681 | 11,274 |
| water | 7.1% | 221 | 2,734 |
| forest | 4.5% | 1,342 | 12,003 |

| site table | total | by class |
|---|---|---|
| v1 | 871 | field 441, forest 34, open 396 |
| v2 | 946 | field 441, forest 34, open 396, urban 75 |

## narva

Kohtla-Jarve to Kingisepp across the Narva river border. Forest dominated, with a river pinch and the Gulf of Finland closing the north flank.

| property | value |
|---|---|
| base | KOHTLA-JARVE at (4.5, 27.9) km |
| target | KINGISEPP at (80.1, 25.6) km |
| box | 106.809 x 60.873 km (6502 km2) |
| bounding box (lon/lat) | W 27.200, S 59.150, E 29.100, N 59.680 |
| projection | EPSG:32635 |
| corridor length | 75.6 km |
| lateral width | 64.0 km |
| range scale | 2.27 |
| load time | 0.2 s |

Terrain by area, 59940 samples at 0.33 km:

| class | share of area | polygons | ring vertices |
|---|---|---|---|
| forest | 43.6% | 1,169 | 18,241 |
| open | 23.2% | - | - |
| sea | 22.5% | 1 | 542 |
| field | 5.9% | 538 | 6,152 |
| water | 3.5% | 137 | 2,952 |
| urban | 1.3% | 462 | 3,226 |

| site table | total | by class |
|---|---|---|
| v1 | 1140 | field 80, forest 692, open 368 |
| v2 | 1160 | field 80, forest 692, open 368, urban 20 |

Narva is the only one of the four with a `sea` layer, a single polygon covering the Gulf of Finland. It is non-emplaceable and does not block line of sight.

## fulda

Point Alpha to Frankfurt, the Fulda Gap, flanked by the Vogelsberg and Rhoen uplands. The largest and most detailed of the four, and correspondingly the most expensive to load and to build games on.

| property | value |
|---|---|
| base | RHEIN-MAIN SW at (7.0, 16.2) km |
| target | POINT ALPHA at (105.8, 87.7) km |
| box | 123.882 x 111.733 km (13842 km2) |
| bounding box (lon/lat) | W 8.450, S 49.900, E 10.200, N 50.900 |
| projection | EPSG:32632 |
| corridor length | 121.9 km |
| lateral width | 163.1 km |
| range scale | 5.79 |
| load time | 4.6 s |

Terrain by area, 60114 samples at 0.48 km:

| class | share of area | polygons | ring vertices |
|---|---|---|---|
| forest | 43.1% | 4,748 | 81,082 |
| field | 24.7% | 39,593 | 284,294 |
| open | 24.0% | - | - |
| urban | 7.9% | 3,370 | 36,600 |
| water | 0.4% | 223 | 4,313 |
| alpine | 0.0% | 9 | 77 |

| site table | total | by class |
|---|---|---|
| v1 | 3150 | field 853, forest 1482, open 815 |
| v2 | 3428 | field 853, forest 1482, open 815, urban 278 |

## Scaling notes

- The exact interdiction matrix is routes x C(sites, K). At K=1 it is tractable everywhere (Fulda, 3,431 columns), while K=3 on Fulda is C(3431,3), about 6.7 billion columns. The gen33 scoring semantics, in which a force induces a soft site prior rather than a hard K-subset, scales; the exact LP path does not. Multi-team screens on Narva or Fulda use the soft-prior semantics or a coarser site grid, and state which.
- Every polygon class present on the four maps is one the game models, so the loader drops nothing.
- Alpine appears on Narva and Fulda as an empty or near-empty layer, so high-terrain walls are not in play on any of the four.
