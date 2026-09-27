"""Command line.

    python -m prisoners [options]            single-product campaign over circles of IPs
    python -m prisoners ads [options]        compare ad-variant sequencing strategies
    python -m prisoners puzzle [options]     the original 100 prisoners puzzle
"""

from __future__ import annotations

import argparse
import json
import sys

from . import campaign, circles, diagnose
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


def circles_main(argv: list[str]) -> None:
    d = circles.Settings()
    p = argparse.ArgumentParser(prog="prisoners", description="Single-product campaign over circles of IPs")
    p.add_argument("--ips", type=int, default=d.ips, help=f"IP addresses in the audience (default {d.ips})")
    p.add_argument("--circles", type=int, default=d.circles, help=f"circles the IPs are grouped into (default {d.circles})")
    p.add_argument("--base", type=float, default=d.base, help=f"average conversion rate per impression (default {d.base})")
    p.add_argument("--spread", type=float, default=d.spread, help=f"how uneven circles are, 0 = all alike (default {d.spread})")
    p.add_argument("--similarity", type=float, default=d.similarity,
                   help=f"how alike neighbouring circles are, 0-0.95 (default {d.similarity})")
    p.add_argument("--days", type=int, default=d.days, help=f"campaign length (default {d.days})")
    p.add_argument("--per-day", type=int, default=d.per_day, help=f"impressions per day (default {d.per_day})")
    p.add_argument("--cooldown", type=int, default=d.cooldown, help=f"days an IP stays fatigued (default {d.cooldown})")
    p.add_argument("--fatigue", type=float, default=d.fatigue,
                   help=f"rate multiplier per recent showing, 1 = no fatigue (default {d.fatigue})")
    p.add_argument("--runs", type=int, default=d.runs, help=f"campaigns to average (default {d.runs})")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)

    s = circles.Settings(a.ips, a.circles, a.base, a.spread, a.similarity, a.days, a.per_day,
                         a.cooldown, a.fatigue, a.runs, a.seed)
    res = circles.compare(s)
    if a.json:
        print(json.dumps({k: {"label": circles.LABELS[k], "mean": r.mean, "failures": r.failures,
                              "failures_per_conversion": r.per_conversion - 1, "curve": r.curve}
                          for k, r in res.items()}, indent=2))
        return
    blast = res["blast"]
    print(f"{s.ips:,} IPs in {s.circles} circles, {s.per_day:,} impressions a day for {s.days} days, "
          f"averaged over {s.runs} campaigns\n")
    print(f"{'strategy':<28} {'conversions':>11} {'failures per sale':>18} {'vs blast':>9}")
    for k, r in res.items():
        print(f"{circles.LABELS[k]:<28} {r.mean:>11.1f} {r.per_conversion - 1:>18.1f} "
              f"{r.mean / (blast.mean or 1) - 1:>+9.0%}")


def diagnose_main(argv: list[str]) -> None:
    p = argparse.ArgumentParser(prog="prisoners diagnose",
                                description="Could hot circles + retry loop have worked on a finished campaign?")
    p.add_argument("csv", help="impression log (date, ip, converted) or circle totals (date, circle, impressions, conversions)")
    p.add_argument("--prefix", type=int, default=24, help="IPv4 subnet size that makes a circle (default 24)")
    p.add_argument("--cooldown", type=int, default=7, help="days a showing counts as recent (default 7)")
    p.add_argument("--no-sim", action="store_true", help="skip the what-if simulation")
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)

    d = diagnose.diagnose(diagnose.load_csv(a.csv, a.prefix), cooldown=a.cooldown, simulate=not a.no_sim)
    if a.json:
        print(json.dumps({"verdict": d.verdict, "checks": [vars(c) for c in d.checks], "whatif": d.whatif,
                          "fitted": vars(d.fitted) if d.fitted else None, "summary": d.summary}, indent=2, default=str))
        return
    mark = {"good": "[ok]  ", "mixed": "[~]   ", "poor": "[x]   ", "n/a": "[n/a] "}
    print(f"Verdict: {d.verdict}\n")
    for c in d.checks:
        print(f"{mark[c.status]}{c.title}\n      {c.finding}\n")
    print(f"The campaign put {d.summary['budget_in_top_fifth']:.0%} of its impressions in the best fifth of circles "
          f"(an even spread is 20%).")
    if d.whatif:
        f = d.fitted
        print(f"\nWhat-if, simulated with the fitted numbers (spread {f.spread}, neighbours {f.similarity}, "
              f"fatigue {f.fatigue}{' assumed' if d.whatif['fatigue_assumed'] else ''}):")
        print(f"  Follow hot circles        {d.whatif['hot']:+.0%} sales vs a random blast")
        print(f"  Hot circles + retry loop  {d.whatif['hot_retry']:+.0%} sales vs a random blast")
        if d.whatif["scaled"]:
            print("  (simulated on a scaled-down copy of the audience to keep it quick)")


def sample_main(argv: list[str]) -> None:
    p = argparse.ArgumentParser(prog="prisoners sample-data", description="Write a sample campaign log with known settings")
    p.add_argument("out", help="CSV file to write")
    p.add_argument("--spread", type=float, default=1.0)
    p.add_argument("--similarity", type=float, default=0.7)
    p.add_argument("--fatigue", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(argv)
    s = circles.Settings(ips=20000, circles=100, days=28, per_day=800, spread=a.spread,
                         similarity=a.similarity, fatigue=a.fatigue, seed=a.seed)
    diagnose.sample_log(a.out, s)
    print(f"Wrote {s.days * s.per_day:,} impressions to {a.out} "
          f"(spread {s.spread}, neighbours {s.similarity}, fatigue {s.fatigue})")


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "diagnose":
        diagnose_main(argv[1:])
    elif argv and argv[0] == "sample-data":
        sample_main(argv[1:])
    elif argv and argv[0] == "puzzle":
        puzzle_main(argv[1:])
    elif argv and argv[0] == "ads":
        campaign_main(argv[1:])
    else:
        circles_main(argv)


if __name__ == "__main__":
    main()
