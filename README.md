# Prisoner

A campaign simulator: which ad should each person see next, when all you keep is campaign totals?

Each person sees up to K of your A ad variants (no repeats) and converts at most once. The strategies only store
aggregate counts (impressions and conversions per cohort × ad), never anything about an individual:

| Strategy | Knows the cohort? | What it does |
|---|---|---|
| Rotate at random | no | baseline, no learning |
| Crossed rotation, then exploit | yes | test phase where each cohort loops through the ads from its own offset (a Latin-square rotation), then each cohort gets its best ads |
| Cohort bandit | yes | Thompson sampling per cohort |
| Learn overall best | no (blind) | Thompson sampling on overall results |
| Follow the lead | no (blind) | prisoners-puzzle idea: next ad chosen from what converted after the previous ad (and whether it was clicked) |
| Perfect knowledge | yes | knows every true rate; the upper limit |

Default run (5,000 people, 10 cohorts, 20 ads, 5 per person, 1% ordinary / up to 8% best, 10 campaigns):

| Strategy | Conversions | vs random |
|---|---|---|
| Crossed rotation, then exploit | ~640 | +74% |
| Cohort bandit | ~590 | +60% |
| Learn overall best | ~460 | +26% |
| Follow the lead | ~430 | +17% |
| Rotate at random | ~370 | — |
| Perfect knowledge | ~735 | +100% |

For blind advertising: learning the overall best ads helps, and helps a lot when cohorts share taste (`--overlap`).
Following the lead doesn't beat it, because one missed ad says little about what a person wants next. The big gain
comes from any grouping signal, and that can be non-personal context (placement, content, time, device, coarse region).

`index.html` is the interactive version (open it in a browser or serve with GitHub Pages).

```bash
python -m prisoners                      # campaign comparison with defaults
python -m prisoners --overlap 0.8        # cohorts mostly like the same ads
python -m prisoners --audience 50000 --runs 3
```

## The original puzzle

`puzzle.html` and `python -m prisoners puzzle` cover the [100 prisoners puzzle](https://en.wikipedia.org/wiki/100_prisoners_problem) itself, reframed as a marketing search problem.

| Puzzle | Marketing |
|---|---|
| prisoner | a campaign team, one per audience segment |
| box | a channel the team can test |
| number in the box | the segment that channel actually converts |
| 50 openings | each team's test budget |
| everyone must find their number | the launch needs every segment covered |

Two strategies are compared:

- **random**: each team tests channels at random, independently.
- **chain** ("follow the lead"): start with the channel labelled for your segment. Its result names another segment, so test that segment's channel next, and keep going.

Both give each team the same ~50% hit rate. Chain correlates the outcomes: teams win together or lose together. That lifts the all-or-nothing success rate from ~10⁻³⁰ to ~31%. It is also fragile. A small share of misleading results (`noise`) breaks the chains.

`puzzle.html` has sliders for teams, test budget, data noise and the success threshold.

## Python engine

Stdlib only, Python 3.10+.

```bash
python -m prisoners puzzle                        # classic puzzle, both strategies
python -m prisoners puzzle -k 60 --noise 0.05     # bigger budget, 5% misleading leads
python -m unittest discover -s tests -t .
```

Use it from code:

```python
from prisoners.campaign import Settings, compare
res = compare(Settings(audience=5000, overlap=0.5, runs=5, seed=1))
{k: r.mean for k, r in res.items()}

from prisoners import simulate, exact_p_all      # the puzzle
simulate(n=100, k=50, strategy="chain", trials=5000, seed=1).p_all
```
