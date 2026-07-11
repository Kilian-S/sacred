# Generation: gen20_f2 (F2: the ONE clean learned-interdictor co-evolution attempt; Obj-1's antagonist AGENT)

- **status: PRE-REGISTERED 2026-07-11 (expansion item 1; Kilian's autonomous launch authority).
  HARD GATE: ONE attempt, no chase, whatever the outcome.**
- **git SHA:** the commit landing this ledger + `scripts/train_f2.py`.

## Why

Obj-1 promises an "environment-altering antagonist AGENT"; in EVERY banked positive result the
antagonist is the ORACLE best response, not a learned agent. The campaign (gen03/04) found learned
congestion adversaries weaker than random, but that evidence is (a) pre-node-ordering-fix and (b)
on the flat congestion game. After gen10's fix removed both confounds, exactly one clean attempt on
the interdiction game is scientifically due. This closes the Obj-1 verbatim gap either way: a
positive (co-evolution works) or the honest oracle-bounded sentence with a measured basis.

## Design

A LEARNED antagonist (a menu-select SAC policy over the K=1 interdiction sets = candidate edges,
scored by its edges' node embeddings through its own GNN) replaces the oracle BR as the SPARRING
PARTNER; it co-evolves simultaneously with the gen14 fleet-route defender (defender minimises,
antagonist maximises mission-failure; both learn each sortie). Instance 35-159 (the post-fix
headline), N=3, K=1, fleet-route defender config VERBATIM from gen14 (role alphas, floor 0.20,
ent-frac 0.5), so the defender's best-checkpoint TAP is directly comparable to the oracle-trained
headline **0.256** (gen14 n=10). 3000 sorties, eval-every 200, seeds {0,1,2}, 3-parallel.

**EVALUATION STAYS ORACLE-BR PORTFOLIO-MAX in every row** (the gen07 portfolio-max lesson: a weak
learned attacker must never flatter the defender). The defender's reported exploitability is always
under the oracle best-response interdictor, never under the learned antagonist.

## Decision metric (PRE-REGISTERED)

Primary = the defender's exact best-checkpoint occupancy exploitability under the ORACLE BR,
mean over 3 seeds. Anchors: oracle-trained ref 0.256 (gen14), ALNS 0.699, eq 0.206.

> **PASS (Obj-1 gets a learned-agent positive):** best-ckpt expl <= 0.356 (within 0.10 of the
> oracle-trained 0.256) on >= 2/3 seeds AND < ALNS 0.699 (beats the deterministic-class optimum).
> **STRONG:** <= 0.306 (within 0.05). **FAIL:** above 0.356 -> the honest oracle-bounded sentence
> for Obj-1, with the measured learned-antagonist strength reported.

Secondary (the "why", reported regardless): the **learned antagonist strength ratio** = its
exploitation of the defender / the oracle BR's exploitation of the same defender (1.0 = as strong
as the oracle interdictor; the campaign's "learned adversaries are weak" finding, now measured
post-fix in the interdiction game). Smoke (60 sorties) already read ~0.29x.

## Command (pinned; via `scratch/gen20_f2.sh`, 3 seeds 3-parallel)

```bash
PYTHONPATH=. .venv/bin/python scripts/train_f2.py --od 35-159 --N 3 --K 1 --k-extra 8 \
  --band 0.15,0.95 --sorties 3000 --eval-every 200 --seed $S --threads 3 \
  --json-out models/runs/gen20_f2/seed$S.json --ckpt-dir models/runs/gen20_f2/seed${S}_ckpts
```

## RESULT (to be appended)
