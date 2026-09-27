"""Campaign simulator: which ad sequence should each person see?

The model:

    audience   -- people arriving one at a time, each in one of C cohorts
    ads        -- A creative variants
    budget     -- each person sees at most K ads (a frequency cap); no repeats
    conversion -- a person converts at most once; the sequence stops there

Every cohort converts at ``base`` on most ads and has ``winners`` strong ads
converting between 60% and 100% of ``best``. ``overlap`` is the chance a
cohort's strong ad is one everybody likes, which is what blind strategies
can find.

The strategies only ever store aggregate counts (impressions, clicks,
conversions per cohort x ad, or per ad path). Nothing is kept about a person
except where they are in their own sequence.

    rotate    -- random order, no repeats. The baseline.
    crossed   -- a test phase where each cohort cycles through the ads from its
                 own offset (a Latin-square rotation), so every cohort x ad
                 pair is tested evenly. After it, each cohort gets its best ads.
    bandit    -- cohort-level Thompson sampling: keeps shifting traffic to the
                 ads that are working for that cohort.
    blind     -- the cohort is unknown. Thompson sampling on overall results.
    lead      -- the cohort is unknown. "Follow the lead": the next ad is chosen
                 from what worked after the previous ad (and whether it was
                 clicked). This is the prisoners-puzzle idea applied to ads.
    oracle    -- knows every true rate. The upper limit.

Stdlib only.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

STRATEGIES = ("rotate", "crossed", "bandit", "blind", "lead", "oracle")
LABELS = {
    "rotate": "Rotate at random",
    "crossed": "Crossed rotation, then exploit",
    "bandit": "Cohort bandit",
    "blind": "Blind: learn overall best",
    "lead": "Blind: follow the lead",
    "oracle": "Perfect knowledge",
}
NEEDS_COHORT = {"crossed", "bandit", "oracle"}
CLICK_MULTIPLIER = 4.0  # clicks run at ~4x the conversion rate
LEAD_PRIOR = 20.0       # how many impressions of overall results a new path borrows


@dataclass
class Settings:
    audience: int = 5000
    cohorts: int = 10
    ads: int = 20
    per_person: int = 5
    base: float = 0.01
    best: float = 0.08
    winners: int = 2
    overlap: float = 0.0
    test_share: float = 0.2
    runs: int = 10
    seed: int | None = None

    def check(self) -> None:
        if not 1 <= self.per_person <= self.ads:
            raise ValueError("per_person must be between 1 and ads")
        if not 1 <= self.winners <= self.ads:
            raise ValueError("winners must be between 1 and ads")
        if not 0 <= self.base <= self.best <= 1:
            raise ValueError("need 0 <= base <= best <= 1")
        if not 0 <= self.overlap <= 1 or not 0 <= self.test_share <= 1:
            raise ValueError("overlap and test_share must be between 0 and 1")


@dataclass
class World:
    conv: list[list[float]]   # conv[cohort][ad]
    click: list[list[float]]  # click[cohort][ad], includes converting clicks


@dataclass
class Outcome:
    strategy: str
    conversions: list[int] = field(default_factory=list)  # one per run
    curve: list[float] = field(default_factory=list)      # mean cumulative conversions at checkpoints

    @property
    def mean(self) -> float:
        return sum(self.conversions) / len(self.conversions)


def make_world(s: Settings, rng: random.Random) -> World:
    favourites = rng.sample(range(s.ads), s.winners)
    conv = []
    for _ in range(s.cohorts):
        row = [s.base] * s.ads
        chosen: set[int] = set()
        for i in range(s.winners):
            if rng.random() < s.overlap and favourites[i] not in chosen:
                a = favourites[i]
            else:
                a = rng.choice([x for x in range(s.ads) if x not in chosen])
            chosen.add(a)
            row[a] = rng.uniform(0.6 * s.best, s.best)
        conv.append(row)
    click = [[min(0.5, max(p, CLICK_MULTIPLIER * p * rng.uniform(0.7, 1.3))) for p in row] for row in conv]
    return World(conv, click)


def _thompson_order(stats: list[list[float]], rng: random.Random) -> list[int]:
    draws = [(rng.betavariate(a, b), i) for i, (a, b) in enumerate(stats)]
    draws.sort(reverse=True)
    return [i for _, i in draws]


def run_once(strategy: str, s: Settings, world: World, people: list[int], rng: random.Random,
             checkpoints: list[int]) -> tuple[int, list[int]]:
    A, K = s.ads, s.per_person
    cohort_stats = [[[1.0, 1.0] for _ in range(A)] for _ in range(s.cohorts)]
    overall = [[1.0, 1.0] for _ in range(A)]
    paths: dict[tuple[int, bool], list[list[float]]] = {}
    seen_in_cohort = [0] * s.cohorts
    test_until = int(s.test_share * s.audience)
    stride = max(1, A // s.cohorts)
    oracle = [sorted(range(A), key=lambda a: -world.conv[c][a])[:K] for c in range(s.cohorts)]

    total, curve, cp = 0, [], 0
    for u, c in enumerate(people):
        order: list[int] | None = None
        if strategy == "rotate":
            order = rng.sample(range(A), K)
        elif strategy == "crossed":
            if u < test_until:
                start = (seen_in_cohort[c] * K + c * stride) % A
                order = [(start + i) % A for i in range(K)]
            else:
                st = cohort_stats[c]
                order = sorted(range(A), key=lambda a: -st[a][0] / (st[a][0] + st[a][1]))[:K]
        elif strategy == "bandit":
            order = _thompson_order(cohort_stats[c], rng)[:K]
        elif strategy == "blind":
            order = _thompson_order(overall, rng)[:K]
        elif strategy == "oracle":
            order = oracle[c]
        seen_in_cohort[c] += 1

        prev: tuple[int, bool] | None = None
        shown: set[int] = set()
        for slot in range(K):
            if strategy == "lead":
                if prev is None:
                    a = _thompson_order(overall, rng)[0]
                else:
                    st = paths.setdefault(prev, [[0.0, 0.0] for _ in range(A)])
                    best_draw, a = -1.0, 0
                    for b in range(A):
                        if b in shown:
                            continue
                        g = overall[b]
                        w = LEAD_PRIOR / (g[0] + g[1])
                        d = rng.betavariate(st[b][0] + g[0] * w, st[b][1] + g[1] * w)
                        if d > best_draw:
                            best_draw, a = d, b
            else:
                a = order[slot]
            shown.add(a)
            converted = rng.random() < world.conv[c][a]
            clicked = converted or rng.random() < world.click[c][a]
            hit = 0 if converted else 1
            cohort_stats[c][a][hit] += 1
            overall[a][hit] += 1
            if prev is not None and strategy == "lead":
                paths[prev][a][hit] += 1
            if converted:
                total += 1
                break
            prev = (a, clicked)

        while cp < len(checkpoints) and u + 1 >= checkpoints[cp]:
            curve.append(total)
            cp += 1
    return total, curve


def compare(s: Settings, strategies: tuple[str, ...] = STRATEGIES, points: int = 20) -> dict[str, Outcome]:
    """Run every strategy on the same worlds and the same stream of people."""
    s.check()
    rng = random.Random(s.seed)
    checkpoints = [max(1, round(s.audience * (i + 1) / points)) for i in range(points)]
    out = {k: Outcome(k, curve=[0.0] * points) for k in strategies}
    for _ in range(s.runs):
        world = make_world(s, rng)
        people = [rng.randrange(s.cohorts) for _ in range(s.audience)]
        run_seed = rng.random()
        for k in strategies:
            total, curve = run_once(k, s, world, people, random.Random(f"{run_seed}-{k}"), checkpoints)
            out[k].conversions.append(total)
            for i, v in enumerate(curve):
                out[k].curve[i] += v / s.runs
    return out
