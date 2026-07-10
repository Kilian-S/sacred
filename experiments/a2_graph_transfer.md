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

## RESULT (to be appended by the post-A1 chain)
