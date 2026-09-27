import os
import tempfile
import unittest

from prisoners import circles, diagnose


def _run(**kw):
    s = circles.Settings(ips=20000, circles=100, days=28, per_day=800, seed=kw.pop("seed", 42), **kw)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "log.csv")
        diagnose.sample_log(path, s)
        return diagnose.diagnose(diagnose.load_csv(path), simulate=False)


def _status(d, key):
    return next(c for c in d.checks if c.key == key).status


class DiagnoseTests(unittest.TestCase):
    def test_finds_structure_when_it_exists(self):
        d = _run(spread=1.2, similarity=0.8, fatigue=0.4)
        self.assertEqual(_status(d, "differ"), "good")
        self.assertEqual(_status(d, "persist"), "good")
        self.assertIn(_status(d, "neighbours"), ("good", "mixed"))
        self.assertTrue(d.verdict.startswith("Likely"))

    def test_finds_nothing_when_circles_are_alike(self):
        d = _run(spread=0.0, similarity=0.0, fatigue=1.0, seed=7)
        self.assertEqual(_status(d, "differ"), "poor")
        self.assertTrue(d.verdict.startswith("Unlikely"))
        self.assertNotEqual(_status(d, "fatigue"), "good")

    def test_circle_totals_format(self):
        rows = []
        for c in range(30):
            for day in range(1, 9):
                rate = 0.03 if c < 6 else 0.005
                rows.append({"Date": f"2026-09-0{day}", "Subnet": f"10.0.{c}.0/24",
                             "Impressions": "400", "Conversions": str(round(400 * rate))})
        camp = diagnose.load_rows(rows)
        self.assertIsNone(camp.log)
        self.assertEqual(len(camp.circle_keys), 30)
        d = diagnose.diagnose(camp, simulate=False)
        self.assertEqual(_status(d, "differ"), "good")
        self.assertEqual(_status(d, "fatigue"), "n/a")

    def test_ip_grouping(self):
        k1, l1 = diagnose.circle_of_ip("192.168.4.77")
        k2, l2 = diagnose.circle_of_ip("192.168.4.200")
        k3, _ = diagnose.circle_of_ip("192.168.5.1")
        self.assertEqual(l1, "192.168.4.0/24")
        self.assertEqual(k1, k2)
        self.assertLess(k1, k3)
        self.assertEqual(diagnose.circle_of_ip("2001:db8:1:2::5")[1], "2001:db8:1::/48")

    def test_spread_recovers_zero(self):
        n = [1000] * 50
        x = [10] * 50
        self.assertEqual(diagnose.spread_of(n, x).sigma, 0.0)

    def test_missing_columns(self):
        with self.assertRaises(ValueError):
            diagnose.load_rows([{"date": "2026-09-01", "ip": "10.0.0.1"}])


if __name__ == "__main__":
    unittest.main()
