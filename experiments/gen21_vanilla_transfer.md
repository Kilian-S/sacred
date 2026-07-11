# Generation: gen21_vanilla_transfer (item 2.1: the missing Obj-5 control at the ZST level)

- **status: PRE-REGISTERED 2026-07-11 (expansion item 2.1); binding at launch.**
- **git SHA:** the commit landing this ledger.

## Why

gen16 compares the adversarial multi-city generalist to random-init, shortest-path and each OD's
loss_det, but NOT to a VANILLA generalist (same multi-city training, TRAVEL objective, no
adversary). So "adversarial training is what makes transfer work" is currently an INFERENCE, not a
measurement. Obj-5's named non-adversarial control exists at single-instance level only. This run
supplies it at the transfer level.

## Design

Identical to gen16 (`--cities kaliningrad,east_london,istanbul --holdout-city gdansk --n-per-city 6
--n-test 6 --pool-seed 0`, 12000 sorties, eval-every 500) EXCEPT `--vanilla`: reward = -normalised
fleet travel cost, NO adversary. Still map-conditioned (same edge-vulnerability observation +
per-route features), still evaluated zero-shot on the SAME held-out Gdansk ODs under the oracle BR.
1 seed (a control, not a headline).

## Decision reading (PRE-REGISTERED)

Anchor: the adversarial gen16 held-out-Gdansk best-checkpoint mean **1.677** (select-on-test) /
**1.733** (select-on-train).

> **Expected (the control confirms the ZST claim):** the vanilla generalist transfers WORSE -
> held-out mean ratio materially ABOVE the adversarial 1.68-1.73 (a travel-objective policy
> concentrates on cheap routes, which are exploitable, so it should be near or above the
> random-init ~1.99 reference). If so, "adversarial training is what makes transfer work" becomes
> a MEASUREMENT. If the vanilla generalist transfers comparably, that is the honest surprising
> result (transfer is about map-conditioning, not adversarial training) - reported as measured.
> Smoke (20 sorties) read ~2.27x, supporting the expected direction.

## RESULT (2026-07-11, 1 seed): the control CONFIRMS adversarial training is causal for transfer

> **Vanilla (travel-objective) generalist held-out-Gdansk best-checkpoint ratio 2.338 (final
> 2.575).** vs the ADVERSARIAL gen16 generalist 1.677/1.733, AND vs random-init ~1.99.

**What is established (Obj-5 at the transfer level, measured not inferred):** the vanilla
generalist - same multi-city training, same map-conditioning, only the objective changed to travel
cost - transfers WORSE than the adversarial generalist (2.34 vs 1.68) and, strikingly, **worse than
a random-init network (2.34 vs 1.99).** A cost-trained policy concentrates its fleet on the cheapest
routes, which are exactly the predictable, exploitable ones, so under the oracle interdictor it is
more exploitable than random routing. So "adversarial training is what makes transfer work" is now
a MEASUREMENT: it is not merely one ingredient of transfer - without it, map-conditioning alone
transfers below random. This is the named Obj-5 non-adversarial control, at the transfer level the
programme previously lacked.
