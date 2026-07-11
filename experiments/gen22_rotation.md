# Generation: gen22_rotation (item 2.3: leave-one-city-out - hold out ISTANBUL)

- **status: PRE-REGISTERED 2026-07-11 (expansion item 2.3); chained after F2+vanilla; binding now.**

## Why

gen16 held out one city (Gdansk). Insurance against "you picked the easy hold-out": rotate the
hold-out to **Istanbul** - the structurally most distant city (1266 nodes, the mega-city arterial
grid, vs the Baltic/London organic meshes). NOT the full leave-one-out rotation (deliberately not
funded); the single most-informative rotation cell.

## Design

gen16 recipe EXACTLY, but train on **Kaliningrad + East London + Gdansk**, hold out **Istanbul**
entirely; 3 seeds, pool-seed 0, 12000 sorties, eval-every 500, same pre-registered bars as gen16.

## Decision reading (PRE-REGISTERED, same as gen16)

> **PASS:** held-out-Istanbul best-checkpoint mean ratio (select-on-train) <= 2.0 AND < the
> random-init reference AND beats loss_det on >= 4/6 ODs. Anchor: gen16 Gdansk 1.677/1.733.
> Istanbul is the harder hold-out (most distant structure), so a mean up to ~1.9-2.0 still passes;
> the point is that transfer holds to whichever city is held out, not just Gdansk.

## RESULT (to be appended)
