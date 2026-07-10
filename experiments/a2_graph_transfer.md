# A2: graph-geometry transfer (the graph-agnostic ZST claim; held-out GRAPH)

- **status: PRE-REGISTERED 2026-07-10 (expansion programme); auto-fires after A1 (gen15) via the
  post-A1 eval chain; EVAL-ONLY. Binding now.**

## Question

Does the A1 generalist (trained on sampled ODs of the 30m-simplified Kaliningrad graph) transfer
zero-shot to a STRUCTURALLY DIFFERENT graph, scored against that graph's own oracle equilibria?

## Honest scope (recorded up front)

A true SECOND OSM CITY requires network access to pull (osmnx present, but this session has no
network: `overpass-api.de` unreachable). The available structurally-distinct graph is
**`data/maps/kaliningrad_original`** (the UNSIMPLIFIED export: different node set, different
topology, different edge lengths, different route menus from the same road network). Transferring
the generalist from the 30m-simplified graph to the original graph is a genuine graph-geometry
transfer test (different node ids, different candidate routes, the featurise cache now keyed to
handle coexisting graphs), reported PRECISELY as "held-out graph geometry", NOT as "second city".
A true second-city row is left as a one-command follow-up (`load_osm_graph_and_demands` + the
`zst_transfer`/generalist evaluators are city-agnostic) for whenever Kilian's machine has network;
this ledger's harness runs it unchanged on any geojson pair.

## Design

Sample held-out ODs on the original graph by the same screen (deg>=3, 3-6 base routes, k8 menus,
R in [10,14]); evaluate the frozen generalist's exact fleet occupancy distribution per OD under
that OD's oracle BR; report the mean best-checkpoint-actor ratio to equilibrium, vs (a) a
random-init net and (b) shortest-path, on the NEW graph.

## Decision reading (PRE-REGISTERED)

> **Transfer:** generalist mean ratio on the held-out graph < random-init AND every OD's absolute
> exploitability < its loss_det. **Strong:** mean ratio <= 2.0 (comparable to A1's in-graph
> held-out). Honest expectation: weaker than A1's same-graph transfer (the generalist has never
> seen this graph's geometry), reported as measured; a partial result still evidences graph-agnostic
> structure in the mixed-strategy concept.

## RESULT (2026-07-10): NEGATIVE, and the negative is the finding

Transfer of the Kaliningrad-30m generalist to the UNSIMPLIFIED Kaliningrad-original graph (denser,
different node set, different construction: NO arterial filter, NO 30m consolidation), 6 held-out
ODs, single-checkpoint exact evaluation:

| | mean ratio to eq | beats loss_det |
|---|---|---|
| generalist | 2.40x | 0/6 |
| random-init reference | 2.41x | - |

> **The generalist TIES a random-init network (2.40 vs 2.41) and beats no held-out OD.** Its
> learned edge has VANISHED on the different graph.

**Control (same-graph sanity, single-checkpoint):** on its OWN training graph's held-out ODs the
generalist DOES keep a consistent edge over random-init (e.g. 72-42 2.29 vs 2.45; 103-27 1.68 vs
1.87; 66-230 2.15 vs 2.40 - a ~0.15-0.2x margin per single checkpoint; the A1 headline 1.59x uses
the TAP, which is better). So the collapse is specifically CROSS-GRAPH, not an evaluator artefact.

**The finding (important, direction-changing):** a generalist trained on ONE source graph transfers
across OD PAIRS (A1: 1.59x, passing) but NOT across GRAPHS (this: ~random). Its GNN has only ever
seen one graph's structure, so a structurally different graph is out-of-distribution for the
encoder; the transferable head features (cost/vulnerability) cannot compensate for an OOD GNN base.
This directly predicts that Kaliningrad-ONLY -> Kyiv zero-shot would likely also fail, and it
reframes the second-city plan: the honest route to a cross-CITY ZST claim is a **multi-graph
generalist** (train on several cities, hold one out), not single-source transfer. Recorded for
Kilian's second-city decision; the original graph is an EXTREME shift (also a construction-pipeline
difference), so a same-pipeline Kyiv is a milder test, but the mechanism (single-source GNN = graph-
overfit) is the operative risk.

(Caveat: single-checkpoint eval understates vs TAP; but both arms are single-checkpoint, so the
tie-with-random conclusion is footing-fair.)
