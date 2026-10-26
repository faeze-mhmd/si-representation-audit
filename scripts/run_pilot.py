                                                                                    

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import numpy as np
import scipy
import sklearn
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from si_audit.data import duplicate_audit, load_training_pool, make_split_manifest, validate_manifest
from si_audit.training import run_one, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--data-artifact", type=Path, default=ROOT / "results/data_audit.json")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/pilot_v1.json")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    expected = {"architecture": [64, 64, 32, 10], "epochs": 50, "batch_size": 64, "learning_rate": 0.001,
                "settings": {"A": 0.0, "B": 0.01}, "seeds": [11, 22, 33], "checkpoint_epochs": [0, 5, 20, 50],
                "primary_layer": "penultimate", "primary_epoch": 50, "threads": 1}
    if any(config[k] != value for k, value in expected.items()):
        raise ValueError("config differs from frozen pilot; create an explicit protocol amendment first")
    if config["permutation_count"] < 1:
        raise ValueError("at least one label permutation is required")
    artifact_bytes = args.data_artifact.read_bytes()
    artifact = json.loads(artifact_bytes)
    if not artifact["audit"]["audit_gate_passed"]:
        raise ValueError("data audit gate failed")
    pool = load_training_pool(args.data_dir)                                
    manifest = artifact["split_manifest"]
    validate_manifest(pool, manifest)
    if make_split_manifest(pool) != manifest:
        raise ValueError("split or SI subset differs from the predeclared protocol")
    if duplicate_audit(pool, manifest["train_rows"], pool, manifest["validation_rows"])["shared_vectors"]:
        raise ValueError("training/validation overlap")

    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    data = {}
    for split, rows_key, subset_key in (("train", "train_rows", "si_train_rows"),
                                        ("validation", "validation_rows", "si_validation_rows")):
        rows = np.array(manifest[rows_key], dtype=np.int64)
        subset = np.array(manifest[subset_key], dtype=np.int64)
        positions = np.searchsorted(rows, subset)
        if not np.array_equal(rows[positions], subset):
            raise ValueError("SI subset misaligned with prediction rows")
        data[split] = {"x": torch.from_numpy(pool.features[rows].astype(np.float32) / 16.0),
                       "y": torch.from_numpy(pool.labels[rows].copy()), "ids": pool.sample_ids[rows],
                       "si_positions": positions}

    paths = ["src/si_audit/training.py", "src/si_audit/data.py", "src/si_audit/metrics.py",
             "src/si_audit/cpu_benchmark.py", "src/si_audit/baselines.py", "scripts/run_pilot.py", "EXPERIMENT_PROTOCOL.md"]
    provenance = {
        "training_file_sha256": pool.sha256,
        "data_artifact_sha256": hashlib.sha256(artifact_bytes).hexdigest(),
        "config_file_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                        "sklearn": sklearn.__version__, "torch": str(torch.__version__), "platform": platform.platform(),
                        "device": "cpu", "torch_threads": torch.get_num_threads(), "loky_max_cpu_count": 1,
                        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
    }
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    input_cache, index_runs, completed = {}, [], []
    total_start = perf_counter()
    for seed in config["seeds"]:
        pair = []
        for setting in config["settings"]:
            run_id = f"{setting}_seed{seed}"
            start = perf_counter()
            record, disposition = run_one(data, config, provenance, output / run_id,
                                         setting=setting, seed=seed, input_cache=input_cache)
            wall = perf_counter() - start
            pair.append(record)
            completed.append(record)
            index_runs.append({"run_id": run_id, "record": f"{run_id}/metrics.json", "bundle": f"{run_id}/model_bundle.json",
                               "disposition": disposition, "invocation_wall_seconds": wall,
                               "final_model_state_sha256": record["final_model_state_sha256"]})
            print(json.dumps({"event": "run_completed", "run_id": run_id, "disposition": disposition,
                              "validation_accuracy": record["final_prediction"]["validation"]["accuracy"],
                              "train_si": record["snapshots"]["50"]["train"]["representations"]["penultimate"]["si"],
                              "validation_si": record["snapshots"]["50"]["validation"]["representations"]["penultimate"]["si"],
                              "wall_seconds": wall}), flush=True)
        if pair[0]["initial_model_state_sha256"] != pair[1]["initial_model_state_sha256"]:
            raise ValueError("paired initialization differs")
        if pair[0]["epoch_order_sha256"] != pair[1]["epoch_order_sha256"]:
            raise ValueError("paired minibatch order differs")
    if sum(r["training_seconds"] for r in completed) > config["training_budget_seconds"]:
        raise ValueError("total training budget exceeded")
    index = {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "provenance": provenance, "config": config,
             "runs": index_runs, "pilot_runs_completed": len(completed), "test_evaluated": False,
             "paired_initialization_and_orders_verified": True,
             "invocation_wall_seconds": perf_counter() - total_start,
             "training_seconds": sum(r["training_seconds"] for r in completed),
             "measurement_seconds": sum(r["measurement_seconds"] for r in completed)}
    save_json(output / "pilot_index.json", index)
    print(json.dumps({"event": "pilot_completed", "index": str(output / "pilot_index.json"),
                      "runs": len(completed), "paired_checks": "passed", "test_evaluated": False,
                      "training_seconds": index["training_seconds"], "wall_seconds": index["invocation_wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
