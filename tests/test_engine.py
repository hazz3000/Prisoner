import math
import unittest

from prisoners import exact_p_all, min_successes, simulate


class ExactTests(unittest.TestCase):
    def test_classic_chain_answer(self):
        # Textbook result: 1 - (H_100 - H_50) ~= 0.3118
        expected = 1 - sum(1 / l for l in range(51, 101))
        self.assertAlmostEqual(exact_p_all(100, 50, "chain"), expected, places=12)

    def test_random_is_hopeless(self):
        self.assertAlmostEqual(exact_p_all(100, 50, "random"), 0.5 ** 100)

    def test_full_budget_always_wins(self):
        self.assertAlmostEqual(exact_p_all(10, 10, "chain"), 1.0)

    def test_small_case_by_enumeration(self):
        # n=3, k=1: only the identity permutation works -> 1/6
        self.assertAlmostEqual(exact_p_all(3, 1, "chain"), 1 / 6)


class SimulationTests(unittest.TestCase):
    def test_chain_matches_exact(self):
        r = simulate(100, 50, "chain", trials=4000, seed=1)
        p = exact_p_all(100, 50, "chain")
        se = math.sqrt(p * (1 - p) / r.trials)
        self.assertLess(abs(r.p_all - p), 4 * se)

    def test_same_average_hit_rate(self):
        # Coordination does not change a single team's odds, only correlation.
        for s in ("random", "chain"):
            r = simulate(100, 50, s, trials=2000, seed=2)
            self.assertAlmostEqual(r.mean_fraction, 0.5, delta=0.02)

    def test_noise_hurts_chain(self):
        clean = simulate(40, 20, "chain", trials=1500, seed=3)
        noisy = simulate(40, 20, "chain", trials=1500, noise=0.5, seed=3)
        self.assertLess(noisy.p_all, clean.p_all)

    def test_histogram_totals(self):
        r = simulate(20, 10, "chain", trials=300, noise=0.2, seed=4)
        self.assertEqual(sum(r.histogram), 300)

    def test_threshold_rounding(self):
        self.assertEqual(min_successes(100, 1.0), 100)
        self.assertEqual(min_successes(100, 0.5), 50)
        self.assertEqual(min_successes(3, 0.5), 2)
        self.assertEqual(min_successes(100, 0.0), 0)

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            simulate(10, 11)
        with self.assertRaises(ValueError):
            simulate(10, 5, "guess")


if __name__ == "__main__":
    unittest.main()
