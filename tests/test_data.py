                                                                                     

import copy
import hashlib
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from si_audit.data import duplicate_audit, load_training_pool, make_split_manifest, parse_optdigits, validate_manifest


def payload(rows):
    return ("\n".join(",".join(map(str, row)) for row in rows) + "\n").encode("ascii")


def synthetic_pool():
    rows = []
    for i in range(800):
                                                                    
        row = [i % 17, (i // 17) % 17, i // (17 * 17)] + [0] * 61
        rows.append(row + [i % 10])
    return parse_optdigits(payload(rows))


class DataValidationTests(unittest.TestCase):
    def test_training_loader_does_not_open_official_test(self):
        data = payload([[i % 17] + [0] * 63 + [i % 10] for i in range(3823)])
        def read_training_only(path):
            self.assertEqual(path.name, "optdigits.tra")
            return data
        with patch.object(Path, "read_bytes", autospec=True, side_effect=read_training_only) as read:
            self.assertEqual(len(load_training_pool(Path("synthetic")).labels), 3823)
            self.assertEqual(read.call_count, 1)

    def test_training_loader_rejects_changed_release_shape(self):
        with patch.object(Path, "read_bytes", return_value=payload([[0] * 64 + [0]])):
            with self.assertRaises(ValueError):
                load_training_pool(Path("synthetic"))

    def test_schema_hash_and_persistent_ids(self):
        data = payload([[0] * 64 + [3], [16] * 64 + [9]])
        pool = parse_optdigits(data, sample_id_start=100)
        self.assertEqual(pool.features.shape, (2, 64))
        np.testing.assert_array_equal(pool.labels, [3, 9])
        np.testing.assert_array_equal(pool.sample_ids, [100, 101])
        self.assertEqual(pool.sha256, hashlib.sha256(data).hexdigest())

    def test_empty_and_bad_column_count_rejected(self):
        for data in (b"", b"\n", b"0,1,2\n", payload([[0] * 66])):
            with self.subTest(data=data[:20]), self.assertRaises(ValueError):
                parse_optdigits(data)

    def test_invalid_features_and_targets_rejected(self):
        for row in ([17] + [0] * 63 + [1], [-1] + [0] * 63 + [1], [0] * 64 + [10],
                    [0] * 64 + [-1], ["nan"] + [0] * 63 + [1], ["1.5"] + [0] * 63 + [1]):
            with self.subTest(row=row[:2]), self.assertRaises(ValueError):
                parse_optdigits(payload([row]))

    def test_split_is_disjoint_complete_and_repeatable(self):
        pool = synthetic_pool()
        a, b = make_split_manifest(pool, per_class=10), make_split_manifest(pool, per_class=10)
        self.assertEqual(a, b)
        self.assertFalse(set(a["train_rows"]) & set(a["validation_rows"]))
        self.assertEqual(set(a["train_rows"]) | set(a["validation_rows"]), set(range(800)))
        validate_manifest(pool, a)

    def test_si_subsets_stay_in_their_split_and_are_balanced(self):
        pool = synthetic_pool()
        manifest = make_split_manifest(pool, per_class=10)
        for name, parent in (("si_train_rows", "train_rows"), ("si_validation_rows", "validation_rows")):
            self.assertTrue(set(manifest[name]) <= set(manifest[parent]))
            np.testing.assert_array_equal(np.bincount(pool.labels[manifest[name]], minlength=10), [10] * 10)

    def test_insufficient_si_examples_rejected(self):
        with self.assertRaises(ValueError):
            make_split_manifest(synthetic_pool(), per_class=60)

    def test_changed_source_and_cross_split_manifest_rejected(self):
        pool = synthetic_pool()
        good = make_split_manifest(pool, per_class=10)
        for edit in ("hash", "overlap", "subset", "negative", "duplicate", "floating"):
            bad = copy.deepcopy(good)
            if edit == "hash": bad["training_file_sha256"] = "wrong"
            elif edit == "overlap": bad["validation_rows"][0] = bad["train_rows"][0]
            elif edit == "subset": bad["si_train_rows"][0] = bad["validation_rows"][0]
            elif edit == "negative": bad["train_rows"][0] = -1
            elif edit == "duplicate": bad["train_rows"][1] = bad["train_rows"][0]
            elif edit == "floating": bad["train_rows"][0] = 0.5
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                validate_manifest(pool, bad)

    def test_conflicting_duplicate_labels_are_detected(self):
        pool = parse_optdigits(payload([[0] * 64 + [0], [0] * 64 + [1], [1] * 64 + [1]]))
        audit = duplicate_audit(pool, range(3))
        self.assertEqual(audit, {"duplicate_groups": 1, "rows_in_duplicate_groups": 2, "conflicting_label_groups": 1})

    def test_cross_split_duplicate_is_detected(self):
        pool = parse_optdigits(payload([[0] * 64 + [0], [0] * 64 + [0], [1] * 64 + [1]]))
        audit = duplicate_audit(pool, [0, 2], pool, [1])
        self.assertEqual(audit["shared_vectors"], 1)
        self.assertEqual(audit["left_rows_in_overlap"], 1)
        self.assertEqual(audit["right_rows_in_overlap"], 1)


if __name__ == "__main__":
    unittest.main()
