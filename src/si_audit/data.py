                                                                            

from collections import defaultdict
from dataclasses import dataclass
import csv
import hashlib
import io
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DigitData:
    features: np.ndarray
    labels: np.ndarray
    sample_ids: np.ndarray
    sha256: str


def parse_optdigits(payload: bytes, *, sample_id_start: int = 0) -> DigitData:
                                                                            

                                                                             
                                                                     
       
    try:
        text = payload.decode("ascii")
        rows = []
        for number, row in enumerate(csv.reader(io.StringIO(text)), start=1):
            if len(row) != 65:
                raise ValueError(f"row {number}: expected exactly 65 columns")
            rows.append([int(value) for value in row])
        if not rows:
            raise ValueError("empty optdigits file")
        values = np.array(rows, dtype=np.int64)
    except (UnicodeDecodeError, OverflowError, csv.Error) as error:
        raise ValueError("invalid canonical optdigits CSV") from error
    if np.any((values[:, :64] < 0) | (values[:, :64] > 16)):
        raise ValueError("features must be integers in 0..16")
    if np.any((values[:, 64] < 0) | (values[:, 64] > 9)):
        raise ValueError("labels must be integers in 0..9")
    if isinstance(sample_id_start, bool) or not isinstance(sample_id_start, int) or sample_id_start < 0:
        raise ValueError("sample_id_start must be a nonnegative integer")
    return DigitData(
        features=values[:, :64].astype(np.uint8),
        labels=values[:, 64].copy(),
        sample_ids=np.arange(sample_id_start, sample_id_start + len(values), dtype=np.int64),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def load_training_pool(data_dir: Path) -> DigitData:
                                                                                
    pool = parse_optdigits((Path(data_dir) / "optdigits.tra").read_bytes())
    if len(pool.labels) != 3823 or set(pool.labels) != set(range(10)):
        raise ValueError("unexpected official training release shape/classes")
    return pool


def class_counts(labels) -> dict:
    values, counts = np.unique(labels, return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(values, counts)}


def make_split_manifest(pool: DigitData, *, split_seed=2027, si_seed=2028, per_class=60) -> dict:
                                                                                   
    if not isinstance(per_class, int) or isinstance(per_class, bool) or per_class < 1:
        raise ValueError("per_class must be a positive integer")
    if set(pool.labels) != set(range(10)):
        raise ValueError("all ten digit classes are required")
    train, validation = train_test_split(
        np.arange(len(pool.labels)), test_size=0.2,
        random_state=split_seed, stratify=pool.labels,
    )
                                                                             
    train, validation = np.sort(train), np.sort(validation)
    rng = np.random.default_rng(si_seed)

    def sample(rows):
        chosen = []
        for label in range(10):
            candidates = rows[pool.labels[rows] == label]
            if len(candidates) < per_class:
                raise ValueError(f"class {label}: insufficient split examples for SI subset")
            chosen.extend(rng.choice(candidates, size=per_class, replace=False).tolist())
        return np.array(sorted(chosen), dtype=np.int64)

    si_train, si_validation = sample(train), sample(validation)
    return {
        "schema_version": 1,
        "training_file_sha256": pool.sha256,
        "split_seed": split_seed, "si_seed": si_seed,
        "si_sampling_order": "single NumPy Generator: training classes 0..9, then validation classes 0..9",
        "validation_fraction": 0.2, "si_per_class": per_class,
        "row_index_convention": "zero-based within original optdigits.tra; sorted ascending",
        "train_rows": train.tolist(), "validation_rows": validation.tolist(),
        "si_train_rows": si_train.tolist(), "si_validation_rows": si_validation.tolist(),
        "class_counts": {
            "train": class_counts(pool.labels[train]),
            "validation": class_counts(pool.labels[validation]),
            "si_train": class_counts(pool.labels[si_train]),
            "si_validation": class_counts(pool.labels[si_validation]),
        },
    }


def validate_manifest(pool: DigitData, manifest: dict) -> None:
                                                                                    
    if manifest["training_file_sha256"] != pool.sha256:
        raise ValueError("training file hash differs from split manifest")
    groups = {}
    for name in ("train_rows", "validation_rows", "si_train_rows", "si_validation_rows"):
        original = manifest[name]
        if any(isinstance(x, bool) or not isinstance(x, int) for x in original):
            raise ValueError("row indices must be integers")
        rows = np.asarray(original, dtype=np.int64)
        if rows.ndim != 1 or not len(rows) or len(np.unique(rows)) != len(rows):
            raise ValueError("row indices must be nonempty, unique and one-dimensional")
        if np.any(rows < 0) or np.any(rows >= len(pool.labels)):
            raise ValueError("row index outside original training pool")
        groups[name] = set(rows.tolist())
    train, validation = groups["train_rows"], groups["validation_rows"]
    if train & validation or train | validation != set(range(len(pool.labels))):
        raise ValueError("training/validation must be disjoint and cover original training")
    for subset, parent in (("si_train_rows", "train_rows"), ("si_validation_rows", "validation_rows")):
        if not groups[subset] <= groups[parent]:
            raise ValueError("SI subset crosses its parent split")
        expected = {str(i): manifest["si_per_class"] for i in range(10)}
        if class_counts(pool.labels[manifest[subset]]) != expected:
            raise ValueError("SI subset is not class balanced at the declared size")


def _groups(pool, rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[pool.features[row].tobytes()].append((int(pool.sample_ids[row]), int(pool.labels[row])))
    return grouped


def duplicate_audit(left: DigitData, left_rows, right: DigitData | None = None, right_rows=None) -> dict:
                                                                                 
    a = _groups(left, left_rows)
    if right is None:
        duplicated = [members for members in a.values() if len(members) > 1]
        conflicts = [members for members in a.values() if len({label for _, label in members}) > 1]
        return {"duplicate_groups": len(duplicated), "rows_in_duplicate_groups": sum(map(len, duplicated)),
                "conflicting_label_groups": len(conflicts)}
    b = _groups(right, right_rows)
    shared = a.keys() & b.keys()
    conflicts = sum(len({label for _, label in a[key] + b[key]}) > 1 for key in shared)
    return {"shared_vectors": len(shared), "left_rows_in_overlap": sum(len(a[k]) for k in shared),
            "right_rows_in_overlap": sum(len(b[k]) for k in shared), "conflicting_label_groups": conflicts}


def audit_original_data(data_dir: Path, manifest: dict) -> dict:
                                                                               
    data_dir = Path(data_dir)
    training = load_training_pool(data_dir)
    test = parse_optdigits((data_dir / "optdigits.tes").read_bytes(), sample_id_start=1_000_000)
    if len(test.labels) != 1797 or set(test.labels) != set(range(10)):
        raise ValueError("unexpected official test release shape/classes")
    validate_manifest(training, manifest)
    all_train, all_test = np.arange(len(training.labels)), np.arange(len(test.labels))
    checks = {
        "within_official_training": duplicate_audit(training, all_train),
        "within_official_test": duplicate_audit(test, all_test),
        "official_training_vs_test": duplicate_audit(training, all_train, test, all_test),
        "inner_training_vs_validation": duplicate_audit(training, manifest["train_rows"], training, manifest["validation_rows"]),
    }
    clear = all(item.get("conflicting_label_groups", 0) == 0 for item in checks.values())
    clear = clear and all(checks[name]["shared_vectors"] == 0 for name in ("official_training_vs_test", "inner_training_vs_validation"))
    return {
        "schema": "64 integer features in 0..16; integer target in 0..9; no missing/nonfinite fields",
        "official_training": {"n": len(training.labels), "sha256": training.sha256, "class_counts": class_counts(training.labels)},
        "official_test": {"n": len(test.labels), "sha256": test.sha256, "class_counts": class_counts(test.labels)},
        "inner_split": {"train_n": len(manifest["train_rows"]), "validation_n": len(manifest["validation_rows"]),
                        "si_train_n": len(manifest["si_train_rows"]), "si_validation_n": len(manifest["si_validation_rows"])},
        "exact_duplicate_checks": checks, "audit_gate_passed": bool(clear),
        "test_access": "schema, label-domain/count and duplicate QA only; no model prediction or performance evaluation",
        "independence_limit": "official writer-disjoint split per UCI; inner validation writer independence cannot be verified from these processed files; near-duplicates not assessed",
    }
