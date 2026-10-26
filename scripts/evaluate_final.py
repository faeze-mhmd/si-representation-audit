                                                                                   

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter
import warnings

os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import scipy
import sklearn
from threadpoolctl import threadpool_limits
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from si_audit.baselines import classification_summary
from si_audit.data import load_training_pool, validate_manifest
from si_audit.final_evaluation import (TEST_SHA256, require, load_official_test, rank_agreement,
                                      fixed_estimators, fit_and_validate, mean_sample_sd)
from si_audit.training import canonical_hash, evaluate, model_from_bundle, model_state_hash, save_json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
                                                                                             
    with (args.output_dir / "evaluation_started.json").open("x", encoding="utf-8") as handle:
        json.dump({"started_at_utc": datetime.now(timezone.utc).isoformat(),
                   "plan_sha256": sha(ROOT / "results/pilot/FINAL_EVALUATION_PLAN.md"),
                   "evaluator_sha256": sha(__file__),
                   "module_sha256": sha(ROOT / "src/si_audit/final_evaluation.py")}, handle, indent=2)
    start = perf_counter()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    pilot = ROOT / "results/pilot"
    index = json.loads((pilot / "pilot_index.json").read_text(encoding="utf-8"))
    config, provenance = index["config"], index["provenance"]
    plan = (pilot / "FINAL_EVALUATION_PLAN.md").read_text(encoding="utf-8")
    require(TEST_SHA256 in plan, "test hash absent from frozen plan")
    for relative, expected in provenance["source_sha256"].items():
        require(sha(ROOT / relative) == expected, "training source changed: " + relative)
    require(sha(ROOT / "configs/pilot_v1.json") == provenance["config_file_sha256"], "config changed")
    require(sha(ROOT / "results/data_audit.json") == provenance["data_artifact_sha256"], "data audit changed")
    pool = load_training_pool(args.data_dir)
    audit = json.loads((ROOT / "results/data_audit.json").read_text(encoding="utf-8"))
    manifest = audit["split_manifest"]
    validate_manifest(pool, manifest)
    train_rows, val_rows = np.array(manifest["train_rows"]), np.array(manifest["validation_rows"])
    require(len(train_rows) == 3058 and len(val_rows) == 765, "wrong split size")
    train_y, val_y = pool.labels[train_rows], pool.labels[val_rows]
    tx = torch.from_numpy(pool.features[train_rows].astype(np.float32) / 16)
    vx = torch.from_numpy(pool.features[val_rows].astype(np.float32) / 16)
    ty, vy = torch.from_numpy(train_y), torch.from_numpy(val_y)
    baseline_history = json.loads((ROOT / "results/baseline_results.json").read_text(encoding="utf-8"))
    require(baseline_history["data_artifact_sha256"] == provenance["data_artifact_sha256"], "baseline split changed")
    baselines = {r["model"]: r for r in baseline_history["baseline_results"]["models"]}
    expected_runs = {f"{s}_seed{seed}" for s in ("A", "B") for seed in (11, 22, 33)}
    require(len(index["runs"]) == 6 and {r["run_id"] for r in index["runs"]} == expected_runs, "wrong run set")
    ready, models, records, tensor_hashes = [], {}, {}, {}
                                                                                         
    with threadpool_limits(limits=1):
        for entry in index["runs"]:
            run_id = entry["run_id"]
            record = json.loads((pilot / entry["record"]).read_text(encoding="utf-8"))
            require(record["completed"] and not record["test_evaluated"], "wrong historical stage")
            require(record["config"] == config and record["provenance"] == provenance, "run provenance drift")
            require(record["fingerprint"] == canonical_hash({"config": config, "provenance": provenance,
                    "setting": record["setting"], "seed": record["seed"]}), "run fingerprint drift")
            model = model_from_bundle(json.loads((pilot / entry["bundle"]).read_text(encoding="utf-8")))
            digest = model_state_hash(model)
            require(digest == entry["final_model_state_sha256"] == record["final_model_state_sha256"]
                    and digest in plan, "unplanned model weights")
            train_metrics, train_parts, _ = evaluate(model, tx, ty, detailed=True)
            val_metrics, val_parts, _ = evaluate(model, vx, vy, detailed=True)
            require(train_metrics == record["final_prediction"]["train"] and
                    val_metrics == record["final_prediction"]["validation"], "head validation drift")
            models[run_id], records[run_id], tensor_hashes[run_id] = model, record, digest
            previous = {r["model"]: r for r in record["probes"]}
            for name, estimator in fixed_estimators(config, probes=True):
                meta = fit_and_validate(estimator, train_parts["penultimate"], train_y,
                    val_parts["penultimate"], val_y, previous[name]["validation"])
                require(meta["parameters"] == previous[name]["parameters"], "probe parameters changed")
                ready.append((run_id + "__" + name, "probe", run_id, estimator, meta))
        for name, estimator in fixed_estimators(config):
            meta = fit_and_validate(estimator, pool.features[train_rows].astype(np.float64) / 16,
                train_y, pool.features[val_rows].astype(np.float64) / 16, val_y, baselines[name]["validation"])
            require(meta["parameters"] == baselines[name]["parameters"], "baseline parameters changed")
            ready.append(("input__" + name, "input_baseline", None, estimator, meta))
        gate_seconds = perf_counter() - start
        print("PASS: all six heads and 15 fitted readouts reproduced validation; opening official test now.", flush=True)
        test = load_official_test(args.data_dir / "optdigits.tes")
        x = torch.from_numpy(test.features.astype(np.float32) / 16)
        y = torch.from_numpy(test.labels)
        predictions, outcomes, test_parts = {}, [], {}
        for run_id, model in models.items():
            tick = perf_counter()
            summary, parts, predicted = evaluate(model, x, y, detailed=True)
            test_parts[run_id] = parts["penultimate"]
            require(model_state_hash(model) == tensor_hashes[run_id], "weights changed during evaluation")
            record = records[run_id]
            scores = {"train_si": record["snapshots"]["50"]["train"]["representations"]["penultimate"]["si"],
                      "validation_si": record["snapshots"]["50"]["validation"]["representations"]["penultimate"]["si"],
                      "validation_accuracy": record["final_prediction"]["validation"]["accuracy"],
                      "negative_validation_cross_entropy": -record["final_prediction"]["validation"]["cross_entropy"]}
            outcomes.append({"readout_id": run_id, "role": "primary_head", "setting": record["setting"],
                "seed": record["seed"], "test": summary, "fixed_predictor_scores": scores,
                "test_predict_seconds": perf_counter() - tick, "warnings": []})
            predictions[run_id] = predicted
        for readout_id, role, run_id, estimator, meta in ready:
            tick = perf_counter()
            features = test_parts[run_id] if role == "probe" else test.features.astype(np.float64) / 16
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                predicted = estimator.predict(features)
            result = {"readout_id": readout_id, "role": role, "parent_run": run_id,
                      "test": classification_summary(test.labels, predicted),
                      "test_predict_seconds": perf_counter() - tick, **meta}
            result["warnings"].extend(str(w.message) for w in caught)
            outcomes.append(result)
            predictions[readout_id] = predicted
    require(len(predictions) == 21, "missing readouts")
    prediction_path = args.output_dir / "predictions.csv"
    with prediction_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["test_row", "sample_id", "truth", *predictions])
        for i in range(len(test.labels)):
            writer.writerow([i, int(test.sample_ids[i]), int(test.labels[i]), *[int(p[i]) for p in predictions.values()]])
    heads = [r for r in outcomes if r["role"] == "primary_head"]
    group_summary = {s: {metric: mean_sample_sd([r["test"][metric] for r in heads if r["setting"] == s])
                        for metric in ("accuracy", "macro_f1")} for s in ("A", "B")}
    paired = []
    for seed in (11, 22, 33):
        a, b = [next(r for r in heads if r["setting"] == s and r["seed"] == seed) for s in ("A", "B")]
        paired.append({"seed": seed, **{m + "_B_minus_A": b["test"][m] - a["test"][m] for m in ("accuracy", "macro_f1")}})
    agreements = {score: rank_agreement([r["fixed_predictor_scores"][score] for r in heads],
                                      [r["test"]["accuracy"] for r in heads]) for score in heads[0]["fixed_predictor_scores"]}
    within = {s: rank_agreement([r["fixed_predictor_scores"]["validation_si"] for r in heads if r["setting"] == s],
                               [r["test"]["accuracy"] for r in heads if r["setting"] == s]) for s in ("A", "B")}
    gallery_pred = predictions[config["error_gallery_run"]]
    error_rows = np.flatnonzero(gallery_pred != test.labels)[:config["error_gallery_count"]]
    save_json(args.output_dir / "error_gallery.json", {"run_id": config["error_gallery_run"],
        "policy": "first 12 errors in ascending official test row; run fixed before test",
        "examples": [{"test_row": int(i), "sample_id": int(test.sample_ids[i]), "truth": int(test.labels[i]),
                      "predicted": int(gallery_pred[i]), "pixels": test.features[i].tolist()} for i in error_rows]})
    sources = set(provenance["source_sha256"]) | {"src/si_audit/final_evaluation.py", "scripts/evaluate_final.py",
        "tests/test_final_evaluation.py", "results/pilot/FINAL_EVALUATION_PLAN.md", "results/pilot/pilot_index.json",
        "results/baseline_results.json", "results/data_audit.json", "configs/pilot_v1.json"}
    sources.update(str((pilot / e["record"]).relative_to(ROOT)).replace("\\", "/") for e in index["runs"])
    results = {"executed_at_utc": datetime.now(timezone.utc).isoformat(), "test_evaluated": True,
        "plan": "results/pilot/FINAL_EVALUATION_PLAN.md", "test_sha256": test.sha256,
        "training_sha256": pool.sha256, "training_n": 3058, "validation_n": 765, "test_n": 1797,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                        "sklearn": sklearn.__version__, "torch": torch.__version__, "platform": platform.platform(), "threads": 1},
        "source_sha256": {r: sha(ROOT / r) for r in sorted(sources)}, "model_state_sha256": tensor_hashes,
        "original_training_config": config, "normalization": "fixed /16; float32 neural/probes, float64 input baselines",
        "verification": {"all_validation_metrics_reproduced_before_test": True, "no_neural_refit": True,
                         "no_test_SI": True, "all_21_readouts_reported": True},
        "outcomes": outcomes, "primary_group_summary": group_summary, "paired_B_minus_A": paired,
        "paired_accuracy_difference_summary": mean_sample_sd([p["accuracy_B_minus_A"] for p in paired]),
        "six_model_rank_agreement": agreements, "within_condition_validation_si_agreement": within,
        "rank_interpretation": "Descriptive six related models on one shared test set. No p-values or general predictive validity claims.",
        "predictions_sha256": sha(prediction_path), "error_gallery_sha256": sha(args.output_dir / "error_gallery.json"),
        "runtime_seconds": {"pretest_validation_gate": gate_seconds, "total_after_imports": perf_counter() - start}}
    save_json(args.output_dir / "final_results.json", results)
    print(json.dumps({"groups": group_summary, "rank_agreement": agreements,
                      "readouts": [{"id": r["readout_id"], "accuracy": r["test"]["accuracy"], "f1": r["test"]["macro_f1"]} for r in outcomes],
                      "runtime": results["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
