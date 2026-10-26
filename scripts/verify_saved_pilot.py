                                                                                    

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from si_audit.data import load_training_pool, validate_manifest
from si_audit.metrics import separation_index
from si_audit.training import canonical_hash, evaluate, model_from_bundle, model_state_hash, permutation_control


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = ROOT / "results/pilot"
    index = json.loads((root / "pilot_index.json").read_text(encoding="utf-8"))
    provenance, config = index["provenance"], index["config"]
    expected_ids = {f"{setting}_seed{seed}" for setting in ("A", "B") for seed in (11, 22, 33)}
    require({r["run_id"] for r in index["runs"]} == expected_ids and len(index["runs"]) == 6, "incorrect run set")
    require(not index["test_evaluated"], "unexpected test evaluation")
    for relative, expected in provenance["source_sha256"].items():
        require(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, "source changed: " + relative)
    require(hashlib.sha256((ROOT / "configs/pilot_v1.json").read_bytes()).hexdigest() == provenance["config_file_sha256"], "config changed")
    audit_bytes = (ROOT / "results/data_audit.json").read_bytes()
    require(hashlib.sha256(audit_bytes).hexdigest() == provenance["data_artifact_sha256"], "split artifact changed")
    pool = load_training_pool(args.data_dir)                                     
    manifest = json.loads(audit_bytes)["split_manifest"]
    validate_manifest(pool, manifest)
    data = {}
    for split, key, subset_key in (("train", "train_rows", "si_train_rows"), ("validation", "validation_rows", "si_validation_rows")):
        rows = np.array(manifest[key])
        data[split] = {"rows": rows, "x": torch.from_numpy(pool.features[rows].astype(np.float32) / 16.0),
                       "y": torch.from_numpy(pool.labels[rows].copy()), "selected": np.searchsorted(rows, manifest[subset_key])}
    records = []
    for entry in index["runs"]:
        record = json.loads((root / entry["record"]).read_text(encoding="utf-8"))
        require(record["completed"] and not record["test_evaluated"], "wrong run stage")
        require(record["provenance"] == provenance and record["config"] == config, "provenance mismatch")
        require(record["fingerprint"] == canonical_hash({"config": config, "provenance": provenance,
                "setting": record["setting"], "seed": record["seed"]}), "fingerprint mismatch")
        bundle = json.loads((root / entry["bundle"]).read_text(encoding="utf-8"))
        model = model_from_bundle(bundle)
        require(model_state_hash(model) == record["final_model_state_sha256"], "wrong final weights")
        require(bundle["metadata"]["fingerprint"] == record["fingerprint"], "bundle provenance mismatch")
        require([h["epoch"] for h in record["history"]] == list(range(51)), "incomplete history")
        for split, item in data.items():
            metrics, parts, _ = evaluate(model, item["x"], item["y"], detailed=True)
            require(metrics == record["final_prediction"][split], "prediction metrics changed")
            selected = item["selected"]
            labels = item["y"].numpy()[selected]
            measured = separation_index(parts["penultimate"][selected], labels, sample_ids=pool.sample_ids[item["rows"]][selected])
            expected = record["snapshots"]["50"][split]["representations"]["penultimate"]
            require(measured.score == expected["si"] and measured.tie_fraction == expected["tie_fraction"]
                    and measured.duplicate_vector_fraction == expected["duplicate_vector_fraction"], "primary SI changed")
            control = permutation_control(labels, measured.nearest_neighbor_indices,
                seed=config["permutation_seeds"][split], count=config["permutation_count"])
            require(control == expected["label_permutation"], "primary shuffle control changed")
        records.append(record)
    for seed in config["seeds"]:
        pair = [r for r in records if r["seed"] == seed]
        require(pair[0]["initial_model_state_sha256"] == pair[1]["initial_model_state_sha256"], "unpaired initialization")
        require(len(pair[0]["epoch_order_sha256"]) == 50 and pair[0]["epoch_order_sha256"] == pair[1]["epoch_order_sha256"], "unpaired order")
    print("PASS: six delivered models; all source/config/split/tensor hashes; all final prediction metrics; 12 primary SI scores and shuffle controls; paired initialization and 50 epoch orders. No test access or file writes.")


if __name__ == "__main__":
    main()
