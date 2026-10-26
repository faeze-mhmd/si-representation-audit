                                                                         

from collections import Counter
from pathlib import Path
import sys
import unittest

import numpy as np

                                                                               
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from si_audit import separation_index


def reference_neighbors(points, ids):
                                                                                 
    result = []
    for i, point in enumerate(points):
        candidates = []
        for j, other in enumerate(points):
            if i != j:
                squared = sum((float(a) - float(b)) ** 2 for a, b in zip(point, other))
                candidates.append((squared, int(ids[j]), j))
        result.append(min(candidates)[2])
    return np.array(result)


class SeparationIndexTests(unittest.TestCase):
    def test_separated_groups_score_one(self):
        result = separation_index([[0], [1], [10], [11]], [0, 0, 1, 1])
        self.assertEqual(result.score, 1.0)
        np.testing.assert_array_equal(result.nearest_neighbor_indices, [1, 0, 3, 2])
        self.assertEqual(result.tie_fraction, 0.0)
        self.assertEqual(result.duplicate_vector_fraction, 0.0)

    def test_alternating_labels_score_zero(self):
        self.assertEqual(separation_index([[0], [1], [3], [7]], [0, 1, 0, 1]).score, 0.0)

    def test_hand_computed_partial_agreement(self):
        result = separation_index([[0], [1], [4], [10]], ["a", "a", "b", "b"])
        self.assertEqual(result.score, 0.75)
        np.testing.assert_array_equal(result.same_label, [True, True, False, True])
        np.testing.assert_allclose(result.nearest_distances, [1, 1, 3, 6])

    def test_single_class_score_one(self):
        self.assertEqual(separation_index([[0, 1], [2, 3], [8, 9]], [7, 7, 7]).score, 1.0)

    def test_self_is_excluded(self):
        result = separation_index([[0], [100]], [0, 1], batch_size=1)
        np.testing.assert_array_equal(result.nearest_neighbor_indices, [1, 0])
        self.assertEqual(result.score, 0.0)

    def test_duplicate_rows_are_other_samples_not_self(self):
        result = separation_index([[0], [0]], [0, 1], sample_ids=[8, 2])
        np.testing.assert_array_equal(result.nearest_neighbor_ids, [2, 8])
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.duplicate_vector_fraction, 1.0)
        self.assertEqual(result.tie_fraction, 0.0)

    def test_same_label_duplicates(self):
        self.assertEqual(separation_index([[0], [0]], [1, 1]).score, 1.0)

    def test_tie_uses_lowest_persistent_id(self):
        result = separation_index([[0], [2], [1]], [0, 1, 0], sample_ids=[30, 10, 20])
        self.assertEqual(result.nearest_neighbor_indices[2], 1)
        self.assertEqual(result.nearest_neighbor_ids[2], 10)
        self.assertAlmostEqual(result.tie_fraction, 1 / 3)

    def test_duplicate_and_tie_diagnostics(self):
        result = separation_index([[0], [0], [0], [4]], [0, 1, 0, 1], sample_ids=[30, 10, 20, 40])
        self.assertEqual(result.score, 0.25)
        self.assertEqual(result.duplicate_vector_fraction, 0.75)
        self.assertEqual(result.tie_fraction, 1.0)

    def test_preserved_ids_make_ties_stable_under_reordering(self):
        points = np.array([[0], [2], [1], [0]], dtype=float)
        labels = np.array([0, 1, 0, 1])
        ids = np.array([30, 10, 20, 40])
        order = np.array([3, 1, 0, 2])
        original = separation_index(points, labels, sample_ids=ids)
        shuffled = separation_index(points[order], labels[order], sample_ids=ids[order])
        self.assertEqual(original.score, shuffled.score)
        np.testing.assert_array_equal(original.nearest_neighbor_ids[order], shuffled.nearest_neighbor_ids)

    def test_no_tie_score_is_row_order_invariant(self):
        points = np.array([[0], [1], [4], [10]])
        labels = np.array([0, 0, 1, 1])
        order = [2, 0, 3, 1]
        self.assertEqual(separation_index(points, labels).score, separation_index(points[order], labels[order]).score)

    def test_matches_independent_oracle_across_batches(self):
        for seed in (0, 7, 29):
            generator = np.random.default_rng(seed)
            for n, d in ((2, 1), (7, 3), (19, 8)):
                points = generator.normal(size=(n, d))
                labels = generator.integers(0, 3, size=n)
                ids = generator.permutation(n) + 100
                expected = reference_neighbors(points, ids)
                for batch_size in (1, 2, 7, 64):
                    with self.subTest(seed=seed, shape=(n, d), batch_size=batch_size):
                        result = separation_index(points, labels, sample_ids=ids, batch_size=batch_size)
                        np.testing.assert_array_equal(result.nearest_neighbor_indices, expected)
                        self.assertEqual(result.score, float(np.mean(labels == labels[expected])))

    def test_ties_match_oracle_across_batch_boundaries(self):
        points = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [0, 0]])
        labels = np.array([0, 0, 1, 1, 0, 1])
        ids = np.array([60, 50, 40, 30, 20, 10])
        expected = reference_neighbors(points, ids)
        counts = Counter(map(tuple, points))
        for batch_size in (1, 2, 4, 10):
            with self.subTest(batch_size=batch_size):
                result = separation_index(points, labels, sample_ids=ids, batch_size=batch_size)
                np.testing.assert_array_equal(result.nearest_neighbor_indices, expected)
                self.assertEqual(result.duplicate_vector_fraction, sum(counts[tuple(x)] > 1 for x in points) / len(points))

    def test_translation_and_uniform_scale_preserve_neighbors(self):
        points = np.array([[0, 2], [1, 3], [4, 7], [10, 11]], dtype=float)
        original = separation_index(points, [0, 0, 1, 1])
        transformed = separation_index(points * 3 + 1e12, [0, 0, 1, 1])
        np.testing.assert_array_equal(original.nearest_neighbor_indices, transformed.nearest_neighbor_indices)
        self.assertEqual(original.score, transformed.score)

    def test_integer_and_float_representations_agree(self):
        points = np.array([[0, 1], [2, 3], [5, 7]], dtype=np.int16)
        self.assertEqual(separation_index(points, [0, 1, 1]).score, separation_index(points.astype(float), [0, 1, 1]).score)

    def test_boolean_labels_supported(self):
        self.assertEqual(separation_index([[0], [1]], [True, False]).score, 0.0)

    def test_float_class_labels_supported(self):
        self.assertEqual(separation_index([[0], [1]], [1.5, 1.5]).score, 1.0)

    def test_metadata_and_per_example_output(self):
        result = separation_index([[0, 1], [1, 2]], [0, 1])
        self.assertEqual((result.n_samples, result.n_features), (2, 2))
        self.assertEqual(result.same_label.dtype, np.bool_)
        np.testing.assert_allclose(result.nearest_distances, [np.sqrt(2), np.sqrt(2)])

    def test_invalid_shapes_rejected(self):
        for points in ([], [[0]], np.empty((0, 2)), np.empty((2, 0)), [0, 1], np.zeros((2, 1, 1))):
            with self.subTest(points=repr(points)), self.assertRaises(ValueError):
                separation_index(points, [0, 1])

    def test_invalid_representation_types_rejected(self):
        for points in ([["0"], ["1"]], [[True], [False]], [[1j], [2j]], np.array([[0], [1]], dtype=object)):
            with self.subTest(points=repr(points)), self.assertRaises(ValueError):
                separation_index(points, [0, 1])

    def test_nonfinite_representations_rejected(self):
        for bad in (np.nan, np.inf, -np.inf):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                separation_index([[0], [bad]], [0, 1])

    def test_invalid_labels_rejected(self):
        for labels in ([], [0], [[0], [1]], [np.nan, 1], [np.inf, 1], [None, 1], [1j, 2j]):
            with self.subTest(labels=repr(labels)), self.assertRaises(ValueError):
                separation_index([[0], [1]], labels)

    def test_mixed_labels_are_not_silently_coerced_to_strings(self):
        for labels in ([1, "1"], [b"a", "a"]):
            with self.subTest(labels=repr(labels)), self.assertRaises(ValueError):
                separation_index([[0], [1]], labels)

    def test_invalid_ids_rejected(self):
        for ids in ([1, 1], [1], [[1], [2]], [1.0, 2.0], [True, False], ["a", "b"]):
            with self.subTest(ids=repr(ids)), self.assertRaises(ValueError):
                separation_index([[0], [1]], [0, 1], sample_ids=ids)

    def test_invalid_batch_size_rejected(self):
        for batch in (0, -1, 1.5, True, np.bool_(True), "2"):
            with self.subTest(batch=batch), self.assertRaises(ValueError):
                separation_index([[0], [1]], [0, 1], batch_size=batch)

    def test_numpy_integer_batch_size_supported(self):
        self.assertEqual(separation_index([[0], [1]], [0, 0], batch_size=np.int64(1)).score, 1.0)

    def test_overflowing_distances_rejected(self):
        for points in ([[1e308], [-1e308]], [[0], [1e200]]):
            with self.subTest(points=points), self.assertRaises(ValueError):
                separation_index(points, [0, 1])

    def test_inputs_are_not_mutated(self):
        points = np.array([[0], [1], [4]], dtype=float)
        labels = np.array([0, 1, 1])
        ids = np.array([30, 20, 10])
        before = [x.copy() for x in (points, labels, ids)]
        separation_index(points, labels, sample_ids=ids)
        for original, current in zip(before, (points, labels, ids)):
            np.testing.assert_array_equal(original, current)


if __name__ == "__main__":
    unittest.main()
