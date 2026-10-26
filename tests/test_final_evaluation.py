                                                                    

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from si_audit.baselines import classification_summary
from si_audit.final_evaluation import rank_agreement, load_official_test, fit_and_validate


class FinalEvaluationTests(unittest.TestCase):
    def test_tied_ranks_have_known_rho_and_pair_counts(self):
        result = rank_agreement([1, 1, 2, 3], [1, 2, 2, 3])
        self.assertAlmostEqual(result["spearman_rho"], 5 / 6)
        self.assertEqual(result["pairs"], {"concordant": 4, "discordant": 0,
            "predictor_tie_only": 1, "outcome_tie_only": 1, "both_tied": 0})

    def test_constants_are_undefined(self):
        self.assertIsNone(rank_agreement([1, 1, 1], [1, 2, 3])["spearman_rho"])
        self.assertIsNone(rank_agreement([1, 2, 3], [1, 1, 1])["spearman_rho"])
        self.assertAlmostEqual(rank_agreement([1, 2, 3], [3, 2, 1])["spearman_rho"], -1)

    def test_nonfinite_and_mismatched_scores_rejected(self):
        for x, y in [([1, 2], [1]), ([1, np.nan], [1, 2]), ([1], [1])]:
            with self.assertRaises(ValueError):
                rank_agreement(x, y)

    def test_corrupt_test_rejected_before_parser(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong.csv"
            path.write_bytes(b"not the official test")
            with patch("si_audit.final_evaluation.parse_optdigits") as parser:
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    load_official_test(path)
                parser.assert_not_called()

    def test_fit_sees_only_training_and_validation_drift_stops(self):
        class Spy:
            def fit(self, x, y):
                self.seen_x, self.seen_y = x.copy(), y.copy()
            def predict(self, x):
                return np.zeros(len(x), dtype=int)
            def get_params(self, deep=False):
                return {}
        train_x, train_y = np.array([[1.], [2.], [3.]]), np.array([0, 1, 0])
        val_x, val_y = np.array([[101.], [102.]]), np.array([0, 0])
        spy = Spy()
        expected = classification_summary(val_y, np.zeros(2, dtype=int))
        fit_and_validate(spy, train_x, train_y, val_x, val_y, expected)
        np.testing.assert_array_equal(spy.seen_x, train_x)
        np.testing.assert_array_equal(spy.seen_y, train_y)
        with self.assertRaisesRegex(ValueError, "test evaluation prohibited"):
            fit_and_validate(spy, train_x, train_y, val_x, val_y, {**expected, "accuracy": 0.5})


if __name__ == "__main__":
    unittest.main()
