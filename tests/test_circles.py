import random
import unittest

from prisoners.circles import STRATEGIES, Settings, compare, make_world


class CirclesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = Settings(ips=8000, circles=80, per_day=300, days=20, runs=6, seed=5)
        cls.res = compare(cls.s)

    def test_budget_is_spent(self):
        for k, r in self.res.items():
            self.assertTrue(all(i == 300 * 20 for i in r.impressions), k)

    def test_combined_idea_beats_baselines(self):
        best = self.res["hot_retry"].mean
        self.assertGreater(best, self.res["blast"].mean * 1.3)
        self.assertGreater(best, self.res["rotate"].mean * 1.3)

    def test_retry_loop_helps_when_fatigue_bites(self):
        self.assertGreater(self.res["hot_retry"].mean, self.res["hot"].mean)

    def test_no_fatigue_no_need_to_rest(self):
        r = compare(Settings(ips=8000, circles=80, per_day=300, days=20, runs=6, seed=5, fatigue=1.0),
                    ("hot", "hot_retry"))
        self.assertGreater(r["hot"].mean, r["hot_retry"].mean)

    def test_curve_ends_at_total(self):
        for r in self.res.values():
            self.assertEqual(len(r.curve), 20)
            self.assertAlmostEqual(r.curve[-1], r.mean)

    def test_world_shape(self):
        s = Settings(ips=1000, circles=10)
        w = make_world(s, random.Random(1))
        self.assertEqual(sum(len(m) for m in w.members), 1000)
        self.assertTrue(all(len(m) == 100 for m in w.members))
        self.assertTrue(all(0 <= p <= 0.5 for p in w.rate))

    def test_rejects_bad_settings(self):
        with self.assertRaises(ValueError):
            compare(Settings(ips=10, circles=20))
        with self.assertRaises(ValueError):
            compare(Settings(similarity=1.0))


if __name__ == "__main__":
    unittest.main()
