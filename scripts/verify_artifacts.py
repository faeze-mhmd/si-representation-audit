                                                                                    

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    audit_bytes = (ROOT / "results/data_audit.json").read_bytes()
    audit = json.loads(audit_bytes)
    result = json.loads((ROOT / "results/baseline_results.json").read_text(encoding="utf-8"))
    require(hashlib.sha256(audit_bytes).hexdigest() == result["data_artifact_sha256"], "data artifact hash mismatch")
    for artifact in (audit, result):
        for relative, expected in artifact["source_sha256"].items():
            require(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, "source changed: " + relative)
    require(audit["audit"]["audit_gate_passed"], "data audit did not pass")
    for model in result["baseline_results"]["models"]:
        for split in ("training", "validation"):
            metrics = model[split]
            matrix = metrics["confusion_matrix"]
            total = sum(map(sum, matrix))
            correct = sum(matrix[i][i] for i in range(10))
            f1 = []
            for i in range(10):
                denominator = sum(matrix[i]) + sum(row[i] for row in matrix)
                f1.append(2 * matrix[i][i] / denominator if denominator else 0.0)
            require(total == metrics["n"], "confusion-matrix sample count mismatch")
            require(abs(correct / total - metrics["accuracy"]) < 1e-12, "accuracy mismatch")
            require(abs(sum(f1) / 10 - metrics["macro_f1"]) < 1e-12, "macro-F1 mismatch")
        require(model["validation"]["n"] == 765, "unexpected validation size")
        errors = 765 - sum(model["validation"]["confusion_matrix"][i][i] for i in range(10))
        require(errors == model["validation_error_count"], "error count mismatch")
    require(result["pilot_runs_completed"] == 0, "this verifier describes the baseline milestone only")
    print("PASS: baseline snapshot source/data hashes, audit gate, training/validation accuracy and macro-F1 values, error counts. This historical baseline artifact records zero pilot runs.")


if __name__ == "__main__":
    main()
