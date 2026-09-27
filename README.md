# Prisoner

Live: https://hazz3000.github.io/Prisoner/

## IP circles (`index.html`, `python -m prisoners`)

A single-product campaign. The audience is IP addresses grouped into circles (subnets, areas); neighbouring
circles behave alike, and showing an IP again too soon works less well (fatigue). With a fixed daily impression
budget, which way of rotating through the circles gets the most sales, and wastes the fewest impressions?

| Strategy | What it does |
|---|---|
| Random blast | each day, random IPs that haven't bought — baseline |
| Even rotation | one fixed loop through every IP, circle by circle |
| Follow hot circles | budget goes in chunks to the circles selling best so far; neighbours count as half-evidence |
| Hot circles + retry loop | the same, and IPs that just saw the ad rest until fatigue wears off, then rejoin |
| Perfect knowledge | knows every IP's rate — a ceiling |

Default run (20,000 IPs, 100 circles, 500 impressions a day for 30 days, 10 campaigns):

| Strategy | Sales | Failures per sale | vs blast |
|---|---|---|---|
| Hot circles + retry loop | ~264 | ~56 | +100% |
| Follow hot circles | ~194 | ~76 | +48% |
| Random blast | ~132 | ~113 | — |
| Even rotation | ~129 | ~115 | −2% |
| Perfect knowledge | ~392 | ~37 | +197% |

Following hot circles is the main lever, and it depends on circles really differing (`--spread`). The retry loop adds
most when fatigue bites; with no fatigue (`--fatigue 1`) resting good IPs costs sales. Rotation alone doesn't help
when the budget never reaches every IP. Note that in the EU/UK an IP address is personal data, and many IPs are
shared or change often, so circles are sturdier than single IPs.

```bash
python -m prisoners                   # defaults above
python -m prisoners --spread 0.2      # circles nearly alike
python -m prisoners --fatigue 1       # no fatigue
```

## Ad variants (`ads.html`, `python -m prisoners ads`)

Which ad should each person see next, when all you keep is campaign totals?

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
python -m prisoners ads                  # ad-variant comparison with defaults
python -m prisoners ads --overlap 0.8    # cohorts mostly like the same ads
python -m prisoners ads --audience 50000 --runs 3
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
