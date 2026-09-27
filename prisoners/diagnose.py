"""Post-campaign diagnosis: could "hot circles + retry loop" have worked here?

Input is a CSV from a finished campaign, in one of two shapes:

    impression log  -- one row per impression: date, ip, converted (0/1)
    circle totals   -- date, circle, impressions, conversions

Column names are matched loosely (see ``_COLUMNS``). IPs are grouped into
circles by subnet: /24 for IPv4, /48 for IPv6 (``prefix`` changes the IPv4
size). A ``circle`` column, if present, is used as given.

The checks, each tied to one condition the strategy needs:

    data        -- enough conversions to learn from at all
    differ      -- do circles convert differently beyond chance?
                   (dispersion test, and the spread in the simulator's units)
    persist     -- do hot circles stay hot? Rank the circles on the first half
                   of the campaign and see how the top fifth did in the second.
                   This is the most direct evidence.
    neighbours  -- do neighbouring subnets convert alike?
    fatigue     -- do repeat showings within the cooldown convert worse?
                   (impression log only)

Then the fitted numbers go into the circles simulator for a what-if estimate.

Stdlib only.
"""

from __future__ import annotations

import csv
import ipaddress
import math
import random
from dataclasses import dataclass, field
from datetime import date, datetime

from . import circles

_COLUMNS = {
    "date": ("date", "day", "timestamp", "time", "datetime", "event_date"),
    "ip": ("ip", "ip_address", "ipaddress", "client_ip", "remote_addr", "user_ip"),
    "circle": ("circle", "subnet", "segment", "group", "area", "region"),
    "impressions": ("impressions", "imps", "impression", "views", "served"),
    "conversions": ("conversions", "converted", "conversion", "sales", "sale", "purchases", "orders"),
}


# ---------- loading ----------

@dataclass
class Campaign:
    circle_keys: list[str]                    # ordered so neighbours sit side by side
    days: list[int]                           # sorted day ordinals present
    cells: dict[tuple[int, int], list[int]]   # (circle index, day) -> [impressions, conversions]
    log: list[tuple[int, str, int]] | None    # (day, ip, converted) when the input was a log
    unique_ips: int | None

    @property
    def impressions(self) -> int:
        return sum(v[0] for v in self.cells.values())

    @property
    def conversions(self) -> int:
        return sum(v[1] for v in self.cells.values())

    def totals(self, days: set[int] | None = None) -> tuple[list[int], list[int]]:
        n = [0] * len(self.circle_keys)
        x = [0] * len(self.circle_keys)
        for (c, d), (i, k) in self.cells.items():
            if days is None or d in days:
                n[c] += i
                x[c] += k
        return n, x


def _find(header: list[str], role: str) -> str | None:
    low = {h.strip().lower(): h for h in header}
    for name in _COLUMNS[role]:
        if name in low:
            return low[name]
    return None


def _day(value: str) -> int:
    v = value.strip()
    for parse in (lambda s: date.fromisoformat(s[:10]), lambda s: datetime.fromisoformat(s).date(),
                  lambda s: datetime.strptime(s[:10], "%d/%m/%Y").date(), lambda s: datetime.strptime(s[:10], "%m/%d/%Y").date()):
        try:
            return parse(v).toordinal()
        except ValueError:
            continue
    try:
        return int(float(v))  # already a day number
    except ValueError:
        raise ValueError(f"can't read date {value!r}; use YYYY-MM-DD") from None


def circle_of_ip(ip: str, prefix: int = 24) -> tuple[int, str]:
    """(sort key, label) for the subnet an IP belongs to."""
    addr = ipaddress.ip_address(ip.strip())
    bits = prefix if addr.version == 4 else 48
    net = ipaddress.ip_network(f"{addr}/{bits}", strict=False)
    return (addr.version << 128) | int(net.network_address), str(net)


def _truthy(v: str) -> int:
    v = v.strip().lower()
    if v in ("", "0", "false", "no", "n", "f"):
        return 0
    if v in ("1", "true", "yes", "y", "t"):
        return 1
    return int(float(v))


def load_rows(rows: list[dict[str, str]], prefix: int = 24) -> Campaign:
    if not rows:
        raise ValueError("the file has no data rows")
    header = list(rows[0].keys())
    col = {role: _find(header, role) for role in _COLUMNS}
    if not col["conversions"]:
        raise ValueError("need a conversions column (conversions, converted, sales, ...)")
    if not (col["ip"] or col["circle"]):
        raise ValueError("need an ip column or a circle column")

    is_log = col["impressions"] is None
    keyed: dict[str, tuple[object, str]] = {}
    raw: list[tuple[str, int, int, int, str | None]] = []  # circle id, day, imps, convs, ip
    for r in rows:
        day = _day(r[col["date"]]) if col["date"] else 0
        ip = r[col["ip"]].strip() if col["ip"] else None
        if col["circle"]:
            label = r[col["circle"]].strip()
            sort: object = (0, float(label)) if label.replace(".", "", 1).isdigit() else (1, label)
        else:
            k, label = circle_of_ip(ip, prefix)
            sort = (0, k)
        keyed.setdefault(label, (sort, label))
        imps = 1 if is_log else int(float(r[col["impressions"]] or 0))
        raw.append((label, day, imps, _truthy(r[col["conversions"]]), ip))

    ordered = [label for _, label in sorted(keyed.values(), key=lambda t: t[0])]
    index = {label: i for i, label in enumerate(ordered)}
    cells: dict[tuple[int, int], list[int]] = {}
    for label, day, imps, conv, _ in raw:
        cell = cells.setdefault((index[label], day), [0, 0])
        cell[0] += imps
        cell[1] += conv
    log = [(day, ip, conv) for _, day, _, conv, ip in raw] if is_log and col["ip"] else None
    ips = len({ip for *_, ip in raw if ip}) if col["ip"] else None
    return Campaign(ordered, sorted({d for _, d in cells}), cells, log, ips)


def load_csv(path: str, prefix: int = 24) -> Campaign:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return load_rows(list(csv.DictReader(f)), prefix)


# ---------- statistics helpers ----------

def _chi2_sf(x: float, k: int) -> float:
    """Upper tail of chi-square (Wilson-Hilferty approximation)."""
    if k <= 0:
        return 1.0
    z = ((x / k) ** (1 / 3) - (1 - 2 / (9 * k))) / math.sqrt(2 / (9 * k))
    return 0.5 * math.erfc(z / math.sqrt(2))


def _rank(v: list[float]) -> list[float]:
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2
        i = j + 1
    return r


def _corr(a: list[float], b: list[float]) -> float:
    n = len(a)
    if n < 3:
        return 0.0
    ma, mb = sum(a) / n, sum(b) / n
    sa = math.sqrt(sum((x - ma) ** 2 for x in a))
    sb = math.sqrt(sum((y - mb) ** 2 for y in b))
    if sa == 0 or sb == 0:
        return 0.0
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sa * sb)


@dataclass
class Spread:
    p: float           # overall rate
    tau2: float        # between-circle variance of true rates
    sigma: float       # log-normal spread, the simulator's `spread`
    chi2: float
    p_value: float
    strength: float    # empirical-Bayes prior strength (pseudo-impressions)


def spread_of(n: list[int], x: list[int]) -> Spread:
    idx = [i for i in range(len(n)) if n[i] > 0]
    N, X = sum(n[i] for i in idx), sum(x[i] for i in idx)
    G = len(idx)
    p = X / N if N else 0.0
    if G < 2 or p <= 0 or p >= 1:
        return Spread(p, 0.0, 0.0, 0.0, 1.0, float("inf"))
    between = sum(n[i] * (x[i] / n[i] - p) ** 2 for i in idx)
    denom = N - sum(n[i] ** 2 for i in idx) / N
    tau2 = max(0.0, (between - (G - 1) * p * (1 - p)) / denom) if denom > 0 else 0.0
    chi2 = sum((x[i] - n[i] * p) ** 2 / (n[i] * p * (1 - p)) for i in idx)
    cv2 = tau2 / (p * p)
    sigma = math.sqrt(math.log(1 + cv2))
    strength = (p * (1 - p) / tau2 - 1) if tau2 > 0 else float("inf")
    return Spread(p, tau2, sigma, chi2, _chi2_sf(chi2, G - 1), max(strength, 1.0))


def shrunk(n: list[int], x: list[int], sp: Spread) -> list[float]:
    if math.isinf(sp.strength):
        return [sp.p] * len(n)
    k = sp.strength
    return [(x[i] + k * sp.p) / (n[i] + k) for i in range(len(n))]


# ---------- the checks ----------

@dataclass
class Check:
    key: str
    title: str
    status: str        # "good", "mixed", "poor", "n/a"
    finding: str
    numbers: dict = field(default_factory=dict)


@dataclass
class Diagnosis:
    checks: list[Check]
    fitted: circles.Settings | None
    whatif: dict[str, float] | None
    verdict: str
    summary: dict


def _persistence(c: Campaign, rng: random.Random) -> Check:
    days = c.days
    if len(days) < 4:
        return Check("persist", "Hot circles stay hot", "n/a",
                     "Needs at least 4 days of dated data to compare the first half with the second.")
    mid = days[len(days) // 2]
    first = {d for d in days if d < mid}
    second = {d for d in days if d >= mid}
    n1, x1 = c.totals(first)
    n2, x2 = c.totals(second)
    both = [i for i in range(len(n1)) if n1[i] > 0 and n2[i] > 0]
    if len(both) < 10 or sum(x2) == 0:
        return Check("persist", "Hot circles stay hot", "n/a", "Too few circles with data in both halves.")
    s1 = shrunk(n1, x1, spread_of(n1, x1))
    ranked = sorted(both, key=lambda i: -s1[i])
    top = ranked[: max(1, len(both) // 5)]
    N2, X2 = sum(n2[i] for i in both), sum(x2[i] for i in both)
    avg2 = X2 / N2
    top_rate = sum(x2[i] for i in top) / max(1, sum(n2[i] for i in top))
    lift = top_rate / avg2 if avg2 else 0.0
    rho = _corr(_rank([s1[i] for i in both]), _rank([x2[i] / n2[i] for i in both]))
    # null: the same circles' second-half results, shuffled across circles
    null_lifts = []
    pairs = [(n2[i], x2[i]) for i in both]
    for _ in range(300):
        rng.shuffle(pairs)
        tn = sum(pairs[j][0] for j in range(len(top)))
        tx = sum(pairs[j][1] for j in range(len(top)))
        null_lifts.append((tx / tn) / avg2 if tn and avg2 else 0.0)
    p_value = (1 + sum(1 for v in null_lifts if v >= lift)) / (1 + len(null_lifts))
    status = "good" if lift >= 1.3 and p_value < 0.05 else "mixed" if lift >= 1.1 and p_value < 0.2 else "poor"
    finding = (f"The top fifth of circles from the first half converted at {lift:.2f}x the average in the second half "
               f"(chance of a lift this big by luck: {p_value:.0%}).")
    return Check("persist", "Hot circles stay hot", status, finding,
                 {"lift": lift, "p_value": p_value, "rank_corr": rho, "top_circles": len(top), "circles_compared": len(both)})


def _neighbours(c: Campaign, rng: random.Random, sp: Spread) -> Check:
    n, x = c.totals()
    idx = [i for i in range(len(n)) if n[i] > 0]
    if len(idx) < 10:
        return Check("neighbours", "Neighbours alike", "n/a", "Too few circles to compare neighbours.")
    s = shrunk(n, x, sp)
    z = [math.log(max(s[i], 1e-9)) for i in idx]
    obs = _corr(z[:-1], z[1:])
    null = []
    zz = z[:]
    for _ in range(300):
        rng.shuffle(zz)
        null.append(_corr(zz[:-1], zz[1:]))
    p_value = (1 + sum(1 for v in null if v >= obs)) / (1 + len(null))
    status = "good" if obs >= 0.3 and p_value < 0.05 else "mixed" if obs >= 0.1 and p_value < 0.2 else "poor"
    return Check("neighbours", "Neighbours alike", status,
                 f"Side-by-side subnets have a correlation of {obs:.2f} in their conversion rates "
                 f"(chance of this by luck: {p_value:.0%}).", {"corr": obs, "p_value": p_value})


def _fatigue(c: Campaign, cooldown: int) -> Check:
    if not c.log:
        return Check("fatigue", "Repeat showings convert worse", "n/a",
                     "Needs an impression log (date, ip, converted) to see repeat showings.")
    by_ip: dict[str, list[tuple[int, int]]] = {}
    for day, ip, conv in c.log:
        by_ip.setdefault(ip, []).append((day, conv))
    buckets = {0: [0, 0], 1: [0, 0], 2: [0, 0], 3: [0, 0]}  # recent showings -> [imps, convs]
    for events in by_ip.values():
        events.sort()
        seen: list[int] = []
        for day, conv in events:
            k = min(3, sum(1 for d in seen if 0 < day - d <= cooldown))
            buckets[k][0] += 1
            buckets[k][1] += conv
            seen.append(day)
    fresh = buckets[0][1] / buckets[0][0] if buckets[0][0] else 0.0
    rep_n = sum(buckets[k][0] for k in (1, 2, 3))
    rep_x = sum(buckets[k][1] for k in (1, 2, 3))
    if rep_n < 200 or fresh == 0:
        return Check("fatigue", "Repeat showings convert worse", "n/a",
                     f"Only {rep_n} repeat showings within {cooldown} days; not enough to measure fatigue.",
                     {"buckets": buckets})
    ratio1 = (buckets[1][1] / buckets[1][0]) / fresh if buckets[1][0] else 0.0
    # rough 95% interval on the ratio from Poisson counts
    se = math.sqrt(1 / max(1, buckets[1][1]) + 1 / max(1, buckets[0][1]))
    lo, hi = ratio1 * math.exp(-1.96 * se), ratio1 * math.exp(1.96 * se)
    status = "good" if hi < 0.85 else "mixed" if ratio1 < 0.9 else "poor"
    finding = (f"A second showing within {cooldown} days converted at {ratio1:.0%} of a first showing "
               f"(likely range {lo:.0%}-{hi:.0%}).")
    if status == "poor":
        finding += " Resting IPs wouldn't gain much."
    finding += (" IPs that got a repeat hadn't bought the first time, so they lean toward weaker prospects;"
                " the true fatigue is probably a little milder than this.")
    return Check("fatigue", "Repeat showings convert worse", status, finding,
                 {"ratio": ratio1, "low": lo, "high": hi, "buckets": buckets, "repeat_share": rep_n / len(c.log)})


def diagnose(c: Campaign, cooldown: int = 7, simulate: bool = True, runs: int = 4, seed: int = 1) -> Diagnosis:
    rng = random.Random(seed)
    n, x = c.totals()
    sp = spread_of(n, x)
    G = sum(1 for v in n if v > 0)
    checks = []

    X = c.conversions
    status = "good" if X >= 300 else "mixed" if X >= 80 else "poor"
    checks.append(Check("data", "Enough sales to learn from", status,
                        f"{X:,} conversions from {c.impressions:,} impressions across {G:,} circles "
                        f"over {len(c.days)} day{'s' if len(c.days) != 1 else ''}.",
                        {"conversions": X, "impressions": c.impressions, "circles": G, "days": len(c.days)}))

    status = "good" if sp.p_value < 0.01 and sp.sigma >= 0.5 else "mixed" if sp.p_value < 0.05 and sp.sigma >= 0.25 else "poor"
    s_all = shrunk(n, x, sp)
    order = sorted((i for i in range(len(n)) if n[i] > 0), key=lambda i: -s_all[i])
    top = order[: max(1, len(order) // 5)]
    top_rate = sum(s_all[i] for i in top) / len(top) if top else 0.0
    checks.append(Check("differ", "Circles convert differently", status,
                        f"After allowing for chance, the best fifth of circles convert about {top_rate / sp.p if sp.p else 0:.1f}x "
                        f"the average (spread {sp.sigma:.2f}; chance it's all noise: {min(1, sp.p_value):.1%}).",
                        {"spread": sp.sigma, "p_value": sp.p_value, "rate": sp.p, "top_fifth_multiple": top_rate / sp.p if sp.p else 0}))

    checks.append(_persistence(c, rng))
    checks.append(_neighbours(c, rng, sp))
    checks.append(_fatigue(c, cooldown))

    # how the real campaign spread its budget
    imps_top = sum(n[i] for i in top)
    summary = {"budget_in_top_fifth": imps_top / c.impressions if c.impressions else 0.0}

    fitted, whatif = None, None
    by = {ch.key: ch for ch in checks}
    if simulate and G >= 5 and sp.p > 0:
        fat = by["fatigue"].numbers.get("ratio")
        days = max(1, len(c.days))
        ips = c.unique_ips or max(G, int(c.impressions / 1.5))
        scale = min(1.0, 40000 / ips)  # keep the what-if quick: simulate a scaled-down copy
        fitted = circles.Settings(
            ips=max(G, int(ips * scale)), circles=G, base=sp.p, spread=round(sp.sigma, 2),
            similarity=round(min(0.95, max(0.0, by["neighbours"].numbers.get("corr", 0.0))), 2),
            days=days, per_day=max(1, int(c.impressions / days * scale)), cooldown=cooldown,
            fatigue=round(min(1.0, max(0.0, fat)), 2) if fat is not None else 0.5, runs=runs, seed=seed)
        fitted.per_day = min(fitted.per_day, fitted.ips)
        res = circles.compare(fitted, ("blast", "hot", "hot_retry"))
        b = res["blast"].mean or 1
        whatif = {"hot": res["hot"].mean / b - 1, "hot_retry": res["hot_retry"].mean / b - 1,
                  "scaled": scale < 1, "fatigue_assumed": fat is None}

    need = [by["differ"], by["persist"]]
    if any(ch.status == "poor" for ch in need):
        verdict = ("Unlikely. The strategy needs circles that differ and stay different, and this campaign "
                   "doesn't show that clearly.")
    elif all(ch.status == "good" for ch in need if ch.status != "n/a") and by["data"].status != "poor":
        verdict = "Likely. Circles differed and the hot ones stayed hot, so shifting budget toward them should have paid off."
    else:
        verdict = "Possibly. Some of the conditions are there, but the evidence is thin; a split test would settle it."
    if by["fatigue"].status == "poor":
        verdict += " Skip the retry loop: repeat showings didn't lose much."
    return Diagnosis(checks, fitted, whatif, verdict, summary)


# ---------- sample data ----------

def sample_log(path: str, s: circles.Settings | None = None, start: date = date(2026, 9, 1)) -> circles.Settings:
    """Write a random-blast campaign log with known settings, for trying the diagnosis."""
    s = s or circles.Settings(ips=20000, circles=100, days=28, per_day=800, seed=42)
    rng = random.Random(s.seed)
    w = circles.make_world(s, rng)
    converted = [False] * s.ips
    shows: list[list[int]] = [[] for _ in range(s.ips)]
    if max(len(m) for m in w.members) > 254:
        raise ValueError("sample data puts each circle in one /24, so use at most 254 IPs per circle")
    host = {i: k + 1 for m in w.members for k, i in enumerate(m)}
    with open(path, "w", newline="", encoding="utf-8") as f:
        out = csv.writer(f)
        out.writerow(["date", "ip", "converted"])
        for day in range(s.days):
            pool = [i for i in range(s.ips) if not converted[i]]
            for i in rng.sample(pool, min(s.per_day, len(pool))):
                recent = sum(1 for d in shows[i] if day - d <= s.cooldown)
                conv = rng.random() < w.rate[i] * s.fatigue ** recent
                shows[i].append(day)
                converted[i] = converted[i] or conv
                c = w.circle_of[i]
                ip = f"10.{c // 256}.{c % 256}.{host[i]}"
                out.writerow([date.fromordinal(start.toordinal() + day).isoformat(), ip, int(conv)])
    return s
