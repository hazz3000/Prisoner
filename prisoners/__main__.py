"""Command line.

    python -m prisoners [campaign options]   compare ad-sequencing strategies
    python -m prisoners puzzle [options]     the original 100 prisoners puzzle
"""

from __future__ import annotations

import argparse
import json
import sys

from . import campaign
from .engine import STRATEGIES, exact_p_all, min_successes, simulate


def campaign_main(argv: list[str]) -> None:
    d = campaign.Settings()
    p = argparse.ArgumentParser(prog="prisoners", description="Compare ad-sequencing strategies on a simulated campaign")
    p.add_argument("--audience", type=int, default=d.audience, help=f"people reached (default {d.audience})")
    p.add_argument("--cohorts", type=int, default=d.cohorts, help=f"audience cohorts (default {d.cohorts})")
    p.add_argument("--ads", type=int, default=d.ads, help=f"ad variants (default {d.ads})")
    p.add_argument("--per-person", type=int, default=d.per_person, help=f"max ads per person (default {d.per_person})")
    p.add_argument("--base", type=float, default=d.base, help=f"conversion rate of an ordinary ad (default {d.base})")
    p.add_argument("--best", type=float, default=d.best, help=f"conversion rate of a cohort's best ad (default {d.best})")
    p.add_argument("--winners", type=int, default=d.winners, help=f"strong ads per cohort (default {d.winners})")
    p.add_argument("--overlap", type=float, default=d.overlap,
                   help="chance a cohort's strong ad is one everyone likes, 0-1 (default 0)")
    p.add_argument("--test-share", type=float, default=d.test_share,
                   help=f"share of the audience used for the crossed-rotation test (default {d.test_share})")
    p.add_argument("--runs", type=int, default=d.runs, help=f"simulated campaigns to average (default {d.runs})")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)

    s = campaign.Settings(a.audience, a.cohorts, a.ads, a.per_person, a.base, a.best, a.winners,
                          a.overlap, a.test_share, a.runs, a.seed)
    res = campaign.compare(s)
    oracle = res["oracle"].mean or 1
    rotate = res["rotate"].mean or 1

    if a.json:
        print(json.dumps({k: {"label": campaign.LABELS[k], "mean": r.mean, "runs": r.conversions, "curve": r.curve}
                          for k, r in res.items()}, indent=2))
        return

    print(f"{s.audience} people, {s.cohorts} cohorts, {s.ads} ads, up to {s.per_person} each, "
          f"averaged over {s.runs} campaigns\n")
    print(f"{'strategy':<32} {'conversions':>11} {'rate':>7} {'vs random':>10} {'of perfect':>11}")
    for k, r in res.items():
        print(f"{campaign.LABELS[k]:<32} {r.mean:>11.1f} {r.mean / s.audience:>7.2%} "
              f"{r.mean / rotate - 1:>+10.0%} {r.mean / oracle:>11.0%}")


def puzzle_main(argv: list[str]) -> None:
    p = argparse.ArgumentParser(prog="prisoners puzzle", description="The classic 100 prisoners puzzle")
    p.add_argument("-n", type=int, default=100, help="prisoners / boxes (default 100)")
    p.add_argument("-k", type=int, default=50, help="boxes each may open (default 50)")
    p.add_argument("--trials", type=int, default=2000)
    p.add_argument("--noise", type=float, default=0.0, help="chance a box points to the wrong next box")
    p.add_argument("--threshold", type=float, default=1.0, help="share who must succeed (default 1.0)")
    p.add_argument("--seed", type=int, default=None)
    a = p.parse_args(argv)

    need = min_successes(a.n, a.threshold)
    print(f"{a.n} prisoners, {a.k} boxes each, {a.trials} trials, noise {a.noise:.0%}, success if >= {need} find theirs\n")
    print(f"{'strategy':<8}  {'avg hit':>8}  {'all succeed':>11}  {'>= threshold':>12}  {'exact (all)':>11}")
    for strat in STRATEGIES:
        r = simulate(a.n, a.k, strat, a.trials, a.noise, a.threshold, a.seed)
        exact = f"{exact_p_all(a.n, a.k, strat):.4%}" if a.noise == 0 else "n/a"
        print(f"{strat:<8}  {r.mean_fraction:>8.1%}  {r.p_all:>11.2%}  {r.p_threshold:>12.2%}  {exact:>11}")


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "puzzle":
        puzzle_main(argv[1:])
    else:
        campaign_main(argv)


if __name__ == "__main__":
    main()
