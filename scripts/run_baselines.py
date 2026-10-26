                                                                                   

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

                                                                            
                                                                          
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import numpy as np
import scipy
import sklearn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from si_audit.baselines import fit_input_baselines
from si_audit.data import duplicate_audit, load_training_pool, make_split_manifest, validate_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--data-artifact", type=Path, required=True)
    parser.add_argument("--benchmark-cpu", action="store_true")
    parser.add_argument("--output", default="-")
    args = parser.parse_args()
    artifact_bytes = args.data_artifact.read_bytes()
    data_artifact = json.loads(artifact_bytes)
    if not data_artifact["audit"]["audit_gate_passed"]:
        raise ValueError("dataset audit gate did not pass")
    pool = load_training_pool(args.data_dir)                                     
    manifest = data_artifact["split_manifest"]
    validate_manifest(pool, manifest)
    if make_split_manifest(pool) != manifest:
        raise ValueError("manifest differs from predeclared seeds/sampling protocol")
    cross = duplicate_audit(pool, manifest["train_rows"], pool, manifest["validation_rows"])
    if cross["shared_vectors"]:
        raise ValueError("cross-split duplicate vectors: amend protocol before fitting")
    paths = ["src/si_audit/data.py", "src/si_audit/baselines.py", "scripts/run_baselines.py", "EXPERIMENT_PROTOCOL.md"]
    timing = None
    if args.benchmark_cpu:
        from si_audit.cpu_benchmark import benchmark_cpu_training
        timing = benchmark_cpu_training(pool, manifest)
        paths.append("src/si_audit/cpu_benchmark.py")
    result = {
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                        "sklearn": sklearn.__version__, "platform": platform.platform(), "loky_max_cpu_count": 1},
        "data_artifact_sha256": hashlib.sha256(artifact_bytes).hexdigest(),
        "training_file_sha256": pool.sha256,
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths},
        "baseline_results": fit_input_baselines(pool, manifest),
        "cpu_training_timing_check": timing,
        "pilot_runs_completed": 0,
    }
    payload = json.dumps(result, indent=2) + "\n"
    if args.output == "-":
        print(payload, end="")
    else:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload, encoding="utf-8")
        print(json.dumps({"saved": str(destination.resolve()), "validation": [
            {"model": m["model"], "accuracy": m["validation"]["accuracy"], "macro_f1": m["validation"]["macro_f1"]}
            for m in result["baseline_results"]["models"]], "cpu_timing": timing}, indent=2))


if __name__ == "__main__":
    main()
