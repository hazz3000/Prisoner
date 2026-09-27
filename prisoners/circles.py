"""Single-product campaign over circles of IP addresses.

The model:

    ips        -- the audience, grouped into ``circles`` (think subnet or area).
                  Circles sit in a ring; neighbouring circles have similar
                  appetite for the product (``similarity``).
    appetite   -- each circle has its own conversion rate around ``base``;
                  ``spread`` sets how uneven the circles are.
    budget     -- ``per_day`` impressions a day for ``days`` days, one per IP
                  per day at most.
    fatigue    -- an IP shown within the last ``cooldown`` days converts at
                  ``fatigue`` x its rate for each recent showing.
    converted  -- an IP that converts is removed from the campaign.

A failure is an impression that doesn't convert.

Strategies (each adds one idea to the one before, except the last):

    blast      -- each day, random IPs that haven't converted. The baseline.
    rotate     -- one fixed loop through every IP, circle by circle: nobody is
                  shown again until the whole audience has been reached.
    hot        -- follow hot circles: each day the budget goes to the circles
                  converting best so far (plus what their neighbours suggest),
                  and cold circles are dropped. IPs inside a circle are random.
    hot_retry  -- hot circles, plus a retry loop: inside a circle, IPs shown
                  recently wait out the cooldown and rejoin later.
    oracle     -- knows every IP's true rate; shows the best rested IPs first.
                  A practical upper limit.

Only circle-level counts are learned from. Stdlib only.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

STRATEGIES = ("blast", "rotate", "hot", "hot_retry", "oracle")
LABELS = {
    "blast": "Random blast",
    "rotate": "Even rotation",
    "hot": "Follow hot circles",
    "hot_retry": "Hot circles + retry loop",
    "oracle": "Perfect knowledge",
}


@dataclass
class Settings:
    ips: int = 20000
    circles: int = 100
    base: float = 0.01
    spread: float = 1.0
    similarity: float = 0.7
    days: int = 30
    per_day: int = 500
    cooldown: int = 7
    fatigue: float = 0.5
    runs: int = 10
    seed: int | None = None

    def check(self) -> None:
        if self.circles < 1 or self.ips < self.circles:
            raise ValueError("need at least one IP per circle")
        if not 1 <= self.per_day <= self.ips:
            raise ValueError("per_day must be between 1 and ips")
        if not 0 <= self.similarity < 1:
            raise ValueError("similarity must be in [0, 1)")
        if not 0 <= self.fatigue <= 1:
            raise ValueError("fatigue must be between 0 and 1")


@dataclass
class Outcome:
    strategy: str
    conversions: list[int] = field(default_factory=list)
    impressions: list[int] = field(default_factory=list)
    curve: list[float] = field(default_factory=list)  # mean cumulative conversions after each day

    @property
    def mean(self) -> float:
        return sum(self.conversions) / len(self.conversions)

    @property
    def failures(self) -> float:
        return (sum(self.impressions) - sum(self.conversions)) / len(self.conversions)

    @property
    def per_conversion(self) -> float:
        return sum(self.impressions) / max(1, sum(self.conversions))


@dataclass
class World:
    circle_of: list[int]        # circle index per IP
    rate: list[float]           # true conversion rate per IP
    members: list[list[int]]    # IPs per circle


def make_world(s: Settings, rng: random.Random) -> World:
    # smooth ring of circle appetites: AR(1) around the ring, then log-normal
    z = [rng.gauss(0, 1)]
    for _ in range(1, s.circles):
        z.append(s.similarity * z[-1] + math.sqrt(1 - s.similarity ** 2) * rng.gauss(0, 1))
    sig = s.spread
    circle_rate = [s.base * math.exp(sig * v - sig * sig / 2) for v in z]
    circle_of = [i * s.circles // s.ips for i in range(s.ips)]
    rate = [min(0.5, circle_rate[c] * math.exp(0.5 * rng.gauss(0, 1) - 0.125)) for c in circle_of]
    members = [[] for _ in range(s.circles)]
    for i, c in enumerate(circle_of):
        members[c].append(i)
    return World(circle_of, rate, members)


def run_once(strategy: str, s: Settings, w: World, rng: random.Random) -> tuple[int, int, list[int]]:
    n, G = s.ips, s.circles
    converted = [False] * n
    shows: list[list[int]] = [[] for _ in range(n)]  # days each IP was shown
    imps = [0] * G
    convs = [0] * G
    loop = [i for c in range(G) for i in rng.sample(w.members[c], len(w.members[c]))]
    pointer = 0
    total_conv = total_imps = 0
    curve = []

    def recent(i: int, day: int) -> int:
        return sum(1 for d in shows[i] if day - d <= s.cooldown)

    def effective(i: int, day: int) -> float:
        return w.rate[i] * s.fatigue ** recent(i, day)

    prior_a, prior_b = 1.0, 1.0 / max(s.base, 1e-4)

    def best_circle(open_circles: list[int]) -> int:
        # Thompson draw per circle, borrowing half the evidence of each neighbour
        best, pick = -1.0, open_circles[0]
        for c in open_circles:
            a = convs[c] + 0.5 * (convs[c - 1] + convs[(c + 1) % G])
            b = imps[c] - convs[c] + 0.5 * (imps[c - 1] - convs[c - 1] + imps[(c + 1) % G] - convs[(c + 1) % G])
            d = rng.betavariate(prior_a + a, prior_b + b)
            if d > best:
                best, pick = d, c
        return pick

    for day in range(s.days):
        live = n - total_conv
        want = min(s.per_day, live)
        chosen: list[int] = []
        if strategy == "blast":
            pool = [i for i in range(n) if not converted[i]]
            chosen = rng.sample(pool, want)
        elif strategy == "rotate":
            picked = set()
            steps = 0
            while len(chosen) < want and steps < n:
                i = loop[pointer]
                pointer = (pointer + 1) % n
                steps += 1
                if not converted[i] and i not in picked:
                    chosen.append(i)
                    picked.add(i)
        elif strategy in ("hot", "hot_retry"):
            # the day's budget goes out in small chunks, each to the circle with
            # the best Thompson draw, so several circles get tried every day
            avail = {}
            for c in range(G):
                pool = [i for i in w.members[c] if not converted[i]
                        and (strategy == "hot" or recent(i, day) == 0)]
                if pool:
                    rng.shuffle(pool)
                    avail[c] = pool
            chunk = max(1, want // 25)
            while len(chosen) < want and avail:
                c = best_circle(list(avail))
                take = avail[c][:min(chunk, want - len(chosen))]
                chosen.extend(take)
                del avail[c][:len(take)]
                if not avail[c]:
                    del avail[c]
            if strategy == "hot_retry" and len(chosen) < want:
                # everyone in every circle is resting: fall back to least-recently shown
                picked = set(chosen)
                rest = sorted((i for i in range(n) if not converted[i] and i not in picked),
                              key=lambda i: shows[i][-1] if shows[i] else -1)
                chosen.extend(rest[: want - len(chosen)])
        elif strategy == "oracle":
            pool = [i for i in range(n) if not converted[i]]
            pool.sort(key=lambda i: (recent(i, day) > 0, -effective(i, day)))
            chosen = pool[:want]

        for i in chosen:
            p = effective(i, day)
            shows[i].append(day)
            c = w.circle_of[i]
            imps[c] += 1
            total_imps += 1
            if rng.random() < p:
                converted[i] = True
                convs[c] += 1
                total_conv += 1
        curve.append(total_conv)
    return total_conv, total_imps, curve


def compare(s: Settings, strategies: tuple[str, ...] = STRATEGIES) -> dict[str, Outcome]:
    """Run every strategy on the same audiences."""
    s.check()
    rng = random.Random(s.seed)
    out = {k: Outcome(k, curve=[0.0] * s.days) for k in strategies}
    for _ in range(s.runs):
        w = make_world(s, rng)
        run_seed = rng.random()
        for k in strategies:
            conv, imps, curve = run_once(k, s, w, random.Random(f"{run_seed}-{k}"))
            out[k].conversions.append(conv)
            out[k].impressions.append(imps)
            for d, v in enumerate(curve):
                out[k].curve[d] += v / s.runs
    return out
