# SACRED

**SACRED** (Soft Actor-Critic Robust Evolutionary Deep RL) is an MSc thesis project (Imperial College
London) framing contested convoy routing as an adversarial security game: a SAC dispatcher routes
convoys base -> FOB while a hidden, committing interdictor places ambushes, and the only defence is
anticipation + unpredictable (mixed-strategy) routing, validated against a computable minimax
equilibrium.

**Current direction (2026-07-07): multi-convoy interdiction** with soft (probabilistic) interception
and a loss-averse (mission-failure) objective, where adversarially-trained randomised routing
provably beats both a non-adversarial SAC and a coordinating classical metaheuristic (ALNS). The
single-convoy shared-edge result (B2-P3) is the banked, proven headline; multi-convoy is the
extension that meets all five research objectives.

Start here: `HANDOVER.md`, then `REDESIGN_INTERDICTION.md` (design, §10 = multi-convoy), `ROADMAP.md`
(plan, Phase M), and `experiments/gen08_interdiction.md` (the live ledger). Historical campaign
record: `CONTEXT.md`, `SACRED_PROGRESS.md`, `experiments/gen0*.md`.

