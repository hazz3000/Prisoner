import random
import unittest

from prisoners.campaign import STRATEGIES, Settings, compare, make_world


class CampaignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = Settings(audience=3000, runs=4, seed=7)
        cls.res = compare(cls.s)

    def test_oracle_is_the_ceiling(self):
        top = self.res["oracle"].mean
        for k in STRATEGIES:
            self.assertLessEqual(self.res[k].mean, top * 1.03, k)

    def test_learning_beats_random(self):
        rotate = self.res["rotate"].mean
        self.assertGreater(self.res["crossed"].mean, rotate * 1.3)
        self.assertGreater(self.res["bandit"].mean, rotate * 1.3)
        self.assertGreater(self.res["blind"].mean, rotate)

    def test_seeing_cohorts_beats_blind(self):
        self.assertGreater(self.res["crossed"].mean, self.res["blind"].mean)

    def test_curve_ends_at_total(self):
        for r in self.res.values():
            self.assertEqual(len(r.curve), 20)
            self.assertAlmostEqual(r.curve[-1], r.mean)

    def test_overlap_helps_blind(self):
        split = compare(Settings(audience=3000, runs=4, seed=3, overlap=0.0), ("rotate", "blind"))
        shared = compare(Settings(audience=3000, runs=4, seed=3, overlap=1.0), ("rotate", "blind"))
        lift = lambda r: r["blind"].mean / r["rotate"].mean
        self.assertGreater(lift(shared), lift(split))

    def test_world_shape(self):
        s = Settings(cohorts=4, ads=6, winners=2, base=0.01, best=0.1)
        w = make_world(s, random.Random(1))
        for row in w.conv:
            strong = [p for p in row if p > s.base]
            self.assertEqual(len(strong), 2)
            self.assertTrue(all(0.06 <= p <= 0.1 for p in strong))
        for crow, prow in zip(w.click, w.conv):
            self.assertTrue(all(c >= p for c, p in zip(crow, prow)))

    def test_rejects_bad_settings(self):
        with self.assertRaises(ValueError):
            compare(Settings(ads=3, per_person=5))


if __name__ == "__main__":
    unittest.main()
