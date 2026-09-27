"""The 100 prisoners puzzle, framed as a marketing search problem.

The puzzle: N prisoners, N boxes, each box hides one prisoner's number in a
random arrangement. Each prisoner may open K boxes. Everyone goes free only if
*every* prisoner finds their own number.

The marketing framing used throughout this tool:

    prisoner  -> a campaign team (one per audience segment)
    box       -> a channel the team can test
    number    -> the segment that channel actually converts
    K         -> each team's test budget (channels it can afford to try)
    success   -> the launch works only if enough teams find their channel

Strategies:

    random  -- each team tests K channels at random, independently.
    chain   -- "follow the lead": start with the channel carrying your own
               label; each test reveals which segment it converts, so next test
               the channel labelled with *that* segment. This is the puzzle's
               loop-following strategy.

``noise`` is the chance a test result is misleading (the lead points to a
random channel instead of the true one) -- a knob for data quality.

The key insight the tool demonstrates: both strategies give each team the same
average hit rate (K/N), but "chain" correlates the outcomes. Teams win together
or lose together, which is what lifts the all-or-nothing success rate from
roughly zero to about 31% at N=100, K=50.

Stdlib only.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

STRATEGIES = ("random", "chain")


@dataclass
class Result:
    strategy: str
    n: int
    k: int
    trials: int
    noise: float
    threshold: float
    # histogram[s] = number of trials in which exactly s teams succeeded
    histogram: list[int] = field(default_factory=list)

    @property
    def p_all(self) -> float:
        """Share of trials in which every team succeeded."""
        return self.histogram[self.n] / self.trials

    @property
    def p_threshold(self) -> float:
        """Share of trials in which at least ``threshold`` of teams succeeded."""
        need = min_successes(self.n, self.threshold)
        return sum(self.histogram[need:]) / self.trials

    @property
    def mean_fraction(self) -> float:
        """Average share of teams that found their channel."""
        total = sum(s * c for s, c in enumerate(self.histogram))
        return total / (self.trials * self.n)


def min_successes(n: int, threshold: float) -> int:
    """Smallest team count that meets a fractional threshold (1.0 = everyone)."""
    need = int(threshold * n + 1e-9)
    if need < threshold * n - 1e-9:
        need += 1
    return max(0, min(n, need))


def _cycle_successes(perm: list[int], k: int) -> int:
    """Chain strategy with perfect data: a team succeeds iff its loop is <= k."""
    n = len(perm)
    seen = [False] * n
    wins = 0
    for start in range(n):
        if seen[start]:
            continue
        length = 0
        box = start
        while not seen[box]:
            seen[box] = True
            box = perm[box]
            length += 1
        if length <= k:
            wins += length
    return wins


def _noisy_chain_success(perm: list[int], team: int, k: int, noise: float, rng: random.Random) -> bool:
    """Chain strategy where each lead is wrong with probability ``noise``."""
    n = len(perm)
    opened = set()
    box = team
    for _ in range(k):
        opened.add(box)
        if perm[box] == team:
            return True
        lead = perm[box]
        if rng.random() < noise:
            lead = rng.randrange(n)
        if lead in opened:
            # A repeated lead would waste a test; try a fresh channel instead.
            remaining = [b for b in range(n) if b not in opened]
            if not remaining:
                return False
            lead = rng.choice(remaining)
        box = lead
    return False


def simulate(
    n: int = 100,
    k: int = 50,
    strategy: str = "chain",
    trials: int = 2000,
    noise: float = 0.0,
    threshold: float = 1.0,
    seed: int | None = None,
) -> Result:
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")
    if not 1 <= k <= n:
        raise ValueError("k must be between 1 and n")
    if not 0.0 <= noise <= 1.0:
        raise ValueError("noise must be between 0 and 1")

    rng = random.Random(seed)
    histogram = [0] * (n + 1)
    perm = list(range(n))
    hit = k / n

    for _ in range(trials):
        if strategy == "random":
            # Given any arrangement, a random K-subset contains your channel
            # with probability K/N, independently of every other team.
            wins = sum(1 for _ in range(n) if rng.random() < hit)
        else:
            rng.shuffle(perm)
            if noise == 0.0:
                wins = _cycle_successes(perm, k)
            else:
                wins = sum(_noisy_chain_success(perm, t, k, noise, rng) for t in range(n))
        histogram[wins] += 1

    return Result(strategy, n, k, trials, noise, threshold, histogram)


def exact_p_all(n: int, k: int, strategy: str) -> float:
    """Closed-form chance that every team succeeds (perfect data only)."""
    if strategy == "random":
        return (k / n) ** n
    # P(a random permutation of size m has no cycle longer than k):
    # a[m] = (1/m) * sum_{l=1..min(k,m)} a[m-l], a[0] = 1
    a = [1.0] + [0.0] * n
    for m in range(1, n + 1):
        a[m] = sum(a[m - l] for l in range(1, min(k, m) + 1)) / m
    return a[n]
