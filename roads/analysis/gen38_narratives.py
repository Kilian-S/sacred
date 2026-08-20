#!/usr/bin/env python3
"""gen38 intelligence-assessment narratives (committed verbatim, authored 2026-07-24 BEFORE any
LLM call; the enemy TYPE is never named). 4 per member; behaviourally worded; some deliberately
NOT keyword-obvious so the keyword baseline must miss what reasoning can catch.

The five members (gen34, normative defs in analysis/gen34_family_probe.py:member_fns):
  reactive     - softmax-BR to our RECENT realised-route counts (counters where we have been)
  sharp        - the same but near-argmax: decisive, concentrated on our single dominant route
  anticipatory - BR to the ANTI-REPEAT prediction: pre-aims where we will likely go NEXT
                 (expects us to switch off our recent routes)
  doctrine     - window-INDEPENDENT: a fixed counter to our overall posture, ignores any sortie
  scattergun   - uniform/indiscriminate: assets spread thinly with no focus
"""

NARRATIVES = {
    "reactive": [
        "Every time we lean on a corridor for a few runs, their ambushes show up on it the next "
        "day. They are consistently a step behind us, punishing the routes we have leaned on "
        "most recently.",
        "Pattern analysis: enemy emplacements track our trailing activity. Wherever our convoys "
        "concentrated over the last several sorties, that is where they mass next. They respond "
        "to what we did, not to what we might do.",
        "The opposing commander is a counter-puncher. He watches our recent movement, waits, and "
        "sets his positions to answer the pattern we have just shown. Give him a habit and he "
        "will exploit it a beat later.",
        "Their emplacements lag our operations by about a cycle: heavy where we were heavy, thin "
        "where we were thin, always keyed to our recent route mix rather than any forecast.",
    ],
    "sharp": [
        "This enemy is decisive to the point of rigidity. He identifies our single most-used "
        "route and throws almost everything onto it, with little hedging elsewhere. Concentrated, "
        "committed, all-in on our dominant lane.",
        "Ruthless and narrow: they do not spread to cover our options, they pick our busiest "
        "corridor and saturate it hard. If one route dominates our traffic, expect an "
        "overwhelming, tightly-focused response there.",
        "The adversary commits with unusual conviction to countering our top route only. Where a "
        "more cautious foe would hedge across several of our lanes, this one concentrates almost "
        "all of its assets against our single heaviest path.",
        "Intelligence describes an aggressive, low-hesitation opponent who answers our most "
        "frequent movement with a near-total commitment, accepting blind spots everywhere else "
        "to hit our main artery as hard as possible.",
    ],
    "anticipatory": [
        "Their reconnaissance is excellent and forward-looking. They do not chase where we have "
        "been; they pre-position where we are likely to go next, on the assumption that we will "
        "rotate off our recently-used routes.",
        "This enemy thinks a move ahead. Knowing we tend to avoid repeating ourselves, he sets "
        "ambushes on the fresh routes we have NOT used lately, waiting for us to switch onto "
        "exactly those.",
        "The opposing staff anticipates our variation. Rather than covering our current pattern, "
        "they forecast the routes we will pivot to when we change things up, and they are already "
        "waiting there.",
        "Forward-leaning and predictive: their emplacements sit on the lanes we have been "
        "neglecting, betting we are about to freshen our routing and rotate onto them next.",
    ],
    "doctrine": [
        "This is a by-the-book, inflexible opponent. He has studied our overall force posture and "
        "committed to a fixed ambush plan against our general tendencies. He does not adjust "
        "sortie to sortie; the plan is set.",
        "Doctrinaire and static: their positions reflect a standing appreciation of our typical "
        "route distribution, laid down in advance and held constant regardless of what any single "
        "run does.",
        "The adversary fights a plan, not a feed. Their emplacements answer our long-run posture "
        "as a whole and stay put; real-time movement of ours does not move them.",
        "Rigid planners. They set their coverage once, against our aggregate habits, and do not "
        "react to individual sorties or recent shifts, trusting a fixed doctrinal counter to our "
        "overall pattern.",
    ],
    "scattergun": [
        "Frankly the enemy looks disorganised. Their assets are spread thin and scattered across "
        "the whole sector with no discernible focus, priority, or pattern we can detect.",
        "No coherent scheme: emplacements appear almost at random over the area, evenly thin, "
        "with nothing to suggest they are keyed to our routes, our habits, or any forecast.",
        "Indiscriminate coverage. Rather than concentrate anywhere meaningful, they dilute across "
        "everything, giving a uniform low-density presence with no identifiable main effort.",
        "Their laydown reads as noise: a little everywhere, nothing anywhere, no focus on our "
        "corridors or our tendencies. Effectively a blanket of thin, unfocused coverage.",
    ],
}

MEMBERS = ["reactive", "sharp", "anticipatory", "doctrine", "scattergun"]

DOCTRINE_BRIEF = {
    "reactive": "REACTIVE: positions counter our recently-used routes (lags our pattern by a cycle).",
    "sharp": "SHARP: decisively concentrates almost all assets on our single most-used route.",
    "anticipatory": "ANTICIPATORY: pre-aims at routes we have NOT used lately, expecting us to "
                    "rotate onto fresh ones next.",
    "doctrine": "DOCTRINE: a fixed plan against our overall posture; does not react to any sortie.",
    "scattergun": "SCATTERGUN: spreads assets uniformly/indiscriminately with no focus.",
}

# naive keyword control: first matching keyword wins (order matters); reasonable-analyst lookup.
KEYWORD_TABLE = [
    ("scattergun", ["random", "scattered", "indiscriminate", "noise", "blanket", "disorganis",
                    "uniform", "no focus", "no discernible", "diluted", "thin, unfocused"]),
    ("anticipatory", ["anticipat", "forecast", "predict", "next", "forward-look", "forward-lean",
                      "pre-position", "pivot", "will go", "have not used", "not used lately",
                      "neglect"]),
    ("doctrine", ["doctrin", "fixed", "static", "by-the-book", "inflexible", "rigid", "standing",
                  "set once", "plan, not a feed", "aggregate", "overall posture"]),
    ("sharp", ["decisive", "concentrat", "single most", "all-in", "saturate", "ruthless",
               "committed", "overwhelming", "heaviest", "main artery", "top route"]),
    ("reactive", ["recent", "counter-punch", "step behind", "trailing", "lag", "we have been",
                  "where we were", "responds to what we did", "leaned on"]),
]


def keyword_classify(text: str) -> str:
    t = text.lower()
    for typ, kws in KEYWORD_TABLE:
        if any(k in t for k in kws):
            return typ
    return "reactive"  # default guess


def all_labelled():
    return [(typ, i, n) for typ in MEMBERS for i, n in enumerate(NARRATIVES[typ])]
