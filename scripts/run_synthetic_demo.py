                                                                                 

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from si_audit import separation_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "results" / "synthetic_demo.json",
        help="JSON destination, or '-' to print the complete artifact to stdout",
    )
    args = parser.parse_args()
    examples = [
        ("separated", [[0], [1], [10], [11]], [0, 0, 1, 1], 1.0),
        ("alternating", [[0], [1], [3], [7]], [0, 1, 0, 1], 0.0),
        ("partial_agreement", [[0], [1], [4], [10]], [0, 0, 1, 1], 0.75),
    ]
    records = []
    for name, points, labels, expected in examples:
        result = separation_index(points, labels)
        if result.score != expected:
            raise AssertionError(f"{name}: expected {expected}, observed {result.score}")
        records.append({
            "name": name,
            "points": points,
            "labels": labels,
            "expected_si": expected,
            "observed_si": result.score,
            "neighbor_indices": result.nearest_neighbor_indices.tolist(),
            "neighbor_ids": result.nearest_neighbor_ids.tolist(),
            "same_label": result.same_label.tolist(),
            "nearest_distances": result.nearest_distances.tolist(),
            "tie_fraction": result.tie_fraction,
            "duplicate_vector_fraction": result.duplicate_vector_fraction,
        })

    sources = [
        "src/si_audit/__init__.py",
        "src/si_audit/metrics.py",
        "tests/test_metrics.py",
        "scripts/run_synthetic_demo.py",
        "EXPERIMENT_PROTOCOL.md",
    ]
    artifact = {
        "evidence_kind": "synthetic metric verification; not model or benchmark results",
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_reference": "https://arxiv.org/abs/1906.05156v1; Section 2.2, equation (1)",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "source_sha256": {
            path: hashlib.sha256((PROJECT_ROOT / path).read_bytes()).hexdigest()
            for path in sources
        },
        "examples": records,
    }
    if args.output == Path("-"):
        print(json.dumps(artifact, indent=2, ensure_ascii=False))
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "synthetic_si": {r["name"]: r["observed_si"] for r in records}}, indent=2))


if __name__ == "__main__":
    main()
