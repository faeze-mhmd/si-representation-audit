                                                                                         

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from si_audit.baselines import classification_summary
from si_audit.final_evaluation import require, rank_agreement, mean_sample_sd

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(directory):
    packaged = {}
    if (directory / "packaging_manifest.json").exists():
        packaged = json.loads((directory / "packaging_manifest.json").read_text(encoding="utf-8"))["files"]
        for name, metadata in packaged.items():
            require(name in {"error_gallery.json", "final_results.json", "evaluation_started.json", "predictions.csv"}, "unknown packaging entry")
            content = (directory / name).read_bytes()
            require(hashlib.sha256(content).hexdigest() == metadata["delivered_sha256"], "packaged file changed")
            require(b"\r" not in content, "mixed packaged newlines")
            restored = content if metadata["original_final_newline"] else content.rstrip(b"\n")
            if metadata["original_newlines"] == "CRLF":
                restored = restored.replace(b"\n", b"\r\n")
            require(hashlib.sha256(restored).hexdigest() == metadata["original_sha256"], "packaging altered original content")
    result = json.loads((directory / "final_results.json").read_text(encoding="utf-8"))
    require(sha(directory / "predictions.csv") == result["predictions_sha256"], "prediction file changed")
    require((packaged.get("error_gallery.json", {}).get("original_sha256") or sha(directory / "error_gallery.json"))
            == result["error_gallery_sha256"], "gallery changed")
    for relative, expected in result["source_sha256"].items():
        require(sha(ROOT / relative) == expected, "source changed since final evaluation: " + relative)
    rows = list(csv.DictReader((directory / "predictions.csv").open(encoding="utf-8")))
    require(len(rows) == result["test_n"] == 1797, "test count changed")
    require([int(r["test_row"]) for r in rows] == list(range(1797)), "test row order changed")
    require([int(r["sample_id"]) for r in rows] == list(range(3823, 5620)), "test sample ids changed")
    labels = np.array([int(r["truth"]) for r in rows])
    require(len(result["outcomes"]) == 21, "wrong readout count")
    for item in result["outcomes"]:
        pred = np.array([int(r[item["readout_id"]]) for r in rows])
        summary = classification_summary(labels, pred)
        require(all(item["test"][key] == value for key, value in summary.items()), "saved metrics mismatch")
    heads = [r for r in result["outcomes"] if r["role"] == "primary_head"]
    for key, expected in result["six_model_rank_agreement"].items():
        require(rank_agreement([r["fixed_predictor_scores"][key] for r in heads],
            [r["test"]["accuracy"] for r in heads]) == expected, "rank agreement mismatch")
    for setting in ("A", "B"):
        for metric in ("accuracy", "macro_f1"):
            require(mean_sample_sd([r["test"][metric] for r in heads if r["setting"] == setting]) ==
                    result["primary_group_summary"][setting][metric], "group summary mismatch")
    gallery = json.loads((directory / "error_gallery.json").read_text(encoding="utf-8"))
    first_errors = [i for i, row in enumerate(rows) if int(row[gallery["run_id"]]) != int(row["truth"])][:12]
    require([e["test_row"] for e in gallery["examples"]] == first_errors, "error gallery selection changed")
    for example in gallery["examples"]:
        row = rows[example["test_row"]]
        require(example["truth"] == int(row["truth"]) and example["predicted"] == int(row[gallery["run_id"]]), "gallery labels changed")
    return result, gallery


def render(result, gallery, directory):
    os.environ["MPLCONFIGDIR"] = str(directory / "mpl-cache")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "svg.fonttype": "none", "svg.hashsalt": "si-audit-final-v1",
        "axes.spines.top": False, "axes.spines.right": False})
    output = directory / "figures"
    output.mkdir(exist_ok=True)

    def save(fig, name):
        fig.savefig(output / (name + ".svg"), metadata={"Date": None}, facecolor="white")
        fig.savefig(output / (name + ".png"), dpi=160, facecolor="white")
        plt.close(fig)

    outcomes = result["outcomes"]
    heads = [r for r in outcomes if r["role"] == "primary_head"]
    colors = {"A": "#1864ab", "B": "#d9480f"}
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    for j, (title, suffix) in enumerate([("Trained head", None), ("Linear probe", "linear_probe"),
                                        ("1-NN probe", "cross_split_1nn_probe")]):
        for k, setting in enumerate(("A", "B")):
            values = [r["test"]["accuracy"] for r in outcomes if
                      (r["role"] == "primary_head" and r["setting"] == setting if suffix is None else
                       r["role"] == "probe" and r["parent_run"].startswith(setting) and r["readout_id"].endswith("__" + suffix))]
            center = j + (k - .5) * .24
            ax.scatter(center + np.array([-.035, 0, .035]), values, color=colors[setting],
                       label=setting + " (three seeds)" if j == 0 else None, zorder=3)
            ax.plot([center-.07, center+.07], [np.mean(values)]*2, color=colors[setting], linewidth=2)
    for name, color, style in [("euclidean_1nn", "#2b8a3e", "--"), ("logistic_regression", "#7048e8", ":")]:
        value = next(r["test"]["accuracy"] for r in outcomes if r["readout_id"] == "input__" + name)
        ax.axhline(value, color=color, linestyle=style, label="Input " + ("1-NN" if name == "euclidean_1nn" else "logistic"))
    ax.set(xticks=range(3), xticklabels=["Trained head (primary)", "Linear probe", "1-NN probe"],
           ylabel="Official test accuracy", ylim=(.943, .98), title="Frozen models on 1,797 official test examples")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="y", alpha=.15)
    ax.legend(loc="upper left", fontsize=9)
    fig.supxlabel("Dots: seeds 11, 22, 33 from left to right; short lines: means. Majority baseline: 10.13% (outside axis).", fontsize=9)
    save(fig, "test_performance")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), layout="constrained")
    titles = ["Train SI", "Validation SI", "Validation accuracy", "Negative validation cross-entropy"]
    offsets = [(-30,-14), (7,4), (-33,-15), (7,5), (-32,6), (7,-13)]
    for ax, key, title in zip(axes.flat, result["six_model_rank_agreement"], titles):
        for i, r in enumerate(heads):
            x, y = r["fixed_predictor_scores"][key], r["test"]["accuracy"]
            ax.scatter(x, y, color=colors[r["setting"]], marker="o" if r["setting"] == "A" else "^", s=42)
            ax.annotate(r["readout_id"].replace("_seed", ""), (x,y), xytext=offsets[i], textcoords="offset points", fontsize=8)
        rho = result["six_model_rank_agreement"][key]["spearman_rho"]
        ax.set(xlabel=title, ylabel="Test head accuracy", title=f"Descriptive Spearman rho = {rho:.3f}", ylim=(.947, .957))
        if key != "negative_validation_cross_entropy":
            ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=2 if key == "validation_si" else 1))
        if key == "validation_si":
            ax.set_xticks(sorted({r["fixed_predictor_scores"][key] for r in heads}))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
        ax.margins(x=.25)
        ax.grid(alpha=.15)
    fig.suptitle("Predeclared predictors versus test accuracy: six related models, one dataset")
    fig.supxlabel("A: blue circles; B: orange triangles. No fitted trend, p-value or independent-model claim. Within-condition validation SI is constant.", fontsize=9)
    save(fig, "si_vs_test")

    selected = next(r for r in heads if r["readout_id"] == gallery["run_id"])
    matrix = np.array(selected["test"]["confusion_matrix"])
    recall_matrix = matrix / matrix.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    image = ax.imshow(recall_matrix, vmin=0, vmax=1, cmap="Blues")
    for i in range(10):
        for j in range(10):
            if matrix[i,j]:
                ax.text(j,i,str(matrix[i,j]),ha="center",va="center",fontsize=8,
                        color="white" if recall_matrix[i,j]>.5 else "black")
    ax.set(xticks=range(10),yticks=range(10),xlabel="Predicted digit",ylabel="True digit",
           title="A, seed 11: test confusion (run chosen before test)\nColor: row recall; text: counts")
    fig.colorbar(image,ax=ax,label="Fraction within true class")
    save(fig,"test_confusion")
    fig, axes = plt.subplots(3,4,figsize=(8,6),layout="constrained")
    for ax, e in zip(axes.flat, gallery["examples"]):
        ax.imshow(np.array(e["pixels"]).reshape(8,8), cmap="gray_r", vmin=0, vmax=16, interpolation="nearest")
        ax.set_title(f"Row {e['test_row']}: {e['truth']} -> {e['predicted']}",fontsize=10)
        ax.axis("off")
    fig.suptitle("A, seed 11: first 12 test errors in original row order\nFixed run and selection policy; diagnostic examples, not a random error sample",fontsize=11)
    save(fig,"test_errors")


def report(result, directory):
    lines = ["# Final test evaluation — SI representation audit", "", "Executed October 21, 2026, under the [frozen plan](../pilot/FINAL_EVALUATION_PLAN.md). No neural retraining, checkpoint selection, hyperparameter changes or test SI. All six heads and 15 readouts reproduced their original validation metrics before test access.", "",
        "## Primary results", "", "3,058 training / 765 validation / 1,797 official test examples. All six epoch-50 models are reported. The official test writers differ from the original training pool; the inner validation split has no verified writer separation.", "",
        "| Setting | Seed | Test accuracy | Macro-F1 | Errors | Train SI | Validation SI |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in result["outcomes"]:
        if r["role"] != "primary_head":
            continue
        t, s = r["test"], r["fixed_predictor_scores"]
        errors = t["n"] - int(np.trace(t["confusion_matrix"]))
        lines.append(f"| {r['setting']} | {r['seed']} | {t['accuracy']:.2%} | {t['macro_f1']:.4f} | {errors} | {s['train_si']:.2%} | {s['validation_si']:.2%} |")
    lines += ["", "| Setting | Mean test accuracy | Sample SD (percentage points) | Mean macro-F1 |", "|---|---:|---:|---:|"]
    for s, group in result["primary_group_summary"].items():
        lines.append(f"| {s} | {group['accuracy']['mean']:.2%} | {100*group['accuracy']['sample_sd']:.3f} | {group['macro_f1']['mean']:.4f} |")
    lines += ["", "Sample SD describes three training seeds on one fixed test set; it is not a confidence interval. Paired B minus A accuracy differences:", "",
              "| Seed | Difference (percentage points) |", "|---|---:|"]
    for p in result["paired_B_minus_A"]:
        lines.append(f"| {p['seed']} | {100*p['accuracy_B_minus_A']:+.3f} |")
    lines += ["", "B changes the correct count by +1, +2 and -1 examples in the paired seeds. The average change is too narrow a basis for a general regularization advantage.", "", "## Predeclared model-ranking analysis", "",
              "| Fixed predictor | Six-model Spearman rho |", "|---|---:|"]
    for name, value in result["six_model_rank_agreement"].items():
        lines.append(f"| {name.replace('_',' ')} | {value['spearman_rho']:.3f} |")
    lines += ["", "Training SI agrees more closely with test accuracy than validation SI in this particular six-model comparison. That contrasts with its near-zero agreement with validation accuracy in the preview. Validation SI remains constant across seeds within each condition, so its within-condition correlation is undefined. These findings do not validate training SI as a general predictor or invalidate validation accuracy/SI. The model family, dataset and accuracy range are narrow; all models share data, architecture and paired seeds. No p-values, statistical superiority, universal metric ranking or novelty claim is made.", "",
        "## Secondary readouts — all reported", "", "Every probe was fitted only on the same 3,058 training representations. Input baselines used those same training rows, without refitting on validation. Logistic C=1, lbfgs, max_iter=1000, random_state=2027; 1-NN uses brute-force Euclidean distance. These are separate classifier readouts, not additional independent replications of the primary neural-head comparison.", "",
        "| Readout | Test accuracy | Macro-F1 | Errors |", "|---|---:|---:|---:|"]
    for r in result["outcomes"]:
        if r["role"] == "primary_head":
            continue
        t = r["test"]
        lines.append(f"| {r['readout_id']} | {t['accuracy']:.2%} | {t['macro_f1']:.4f} | {t['n']-int(np.trace(t['confusion_matrix']))} |")
    lines += ["", "The input 1-NN baseline (97.27%) outperforms every trained head and every learned-feature probe here. Changing the readout to a probe does not remove that observed disadvantage. There was no post-test attempt to improve it.", "",
        "## Errors and reproducibility", "", "Full confusion matrices and per-class precision/recall/F1 for all 21 readouts are in [final_results.json](final_results.json). [predictions.csv](predictions.csv) stores every prediction: test_row is zero-based in optdigits.tes; sample_id = 3823 + test_row. Every error is recoverable from truth and readout columns. The gallery uses the prechosen A_seed11 model and its first 12 errors in ascending test row order; it is not a representative sample of all mistakes.", "",
        "Figures: [test comparison](figures/test_performance.svg), [predictors versus test](figures/si_vs_test.svg), [fixed-model confusion](figures/test_confusion.svg), [fixed-model errors](figures/test_errors.svg).", "",
        "The saved prediction CSV is independently checked against all 21 confusion matrices, accuracies and macro-F1 scores. Source/config/split/plan/tensor hashes and software versions are recorded; no clean-environment or cross-platform reproduction claim is made. The historical pilot index and plan keep their pre-test status text; this report and final_results.json record completion.", "",
        "Packaging note: the workspace editor converted result JSON from CRLF to LF and added a terminal newline to the start sentinel. The numeric text is unchanged. [packaging_manifest.json](packaging_manifest.json) retains both original and delivered hashes. The verifier reconstructs original line endings and requires the original byte hashes as well as delivered hashes. The prediction CSV is byte-identical to the executed output. Initial strict gallery verification correctly rejected the newline difference; the packaging check was added afterward, without rerunning models or modifying numerical results.", "",
        "Measured runtime (excluding Python imports/startup):", ""]
    for key, value in result["runtime_seconds"].items():
        lines.append(f"- {key}: {value:.6f} seconds.")
    fixed_head = next(r for r in result["outcomes"] if r["readout_id"] == "A_seed11")
    lines += ["", "Per-class audit of the prechosen A_seed11 head:", "",
              "| True digit | Test support | Recall | Errors |", "|---|---:|---:|---:|"]
    matrix = np.array(fixed_head["test"]["confusion_matrix"])
    for digit in range(10):
        item = fixed_head["test"]["per_class"][str(digit)]
        lines.append(f"| {digit} | {item['support']} | {item['recall']:.2%} | {int(matrix[digit].sum()-matrix[digit,digit])} |")
    lines += ["", "The largest directional confusions in this fixed model are 7 -> 5 (10 examples), 8 -> 1 (9), and 3 -> 2 (8). These are descriptive observations after evaluation, not evidence of a specific cause or a basis for tuning on this test.", "",
        "No estimator warnings were emitted." if not any(r["warnings"] for r in result["outcomes"]) else "Estimator warnings are preserved per readout in JSON.", "",
        "```powershell", "python -B -m unittest discover -s tests -v", "python -B scripts/verify_saved_pilot.py --data-dir (Join-Path $digitCache 'raw')",
        "# Final evaluator requires a fresh output folder; existing sentinel prevents overwrite/repeated runs.",
        "python -B scripts/evaluate_final.py --data-dir (Join-Path $digitCache 'raw') --output-dir (Join-Path $digitCache 'final_v1')",
        "# Saved-results verification/plotting does not refit or run model predictions:",
        "python -B scripts/analyze_final.py --results-dir results/final --output-dir (Join-Path $digitCache 'final_figures')", "```", "",
        "Set $digitCache using the README acquisition workflow. The completed run used the C: temporary data cache; text/SVG deliverables were then preserved in this project. Drive C had about 94 GiB available before this stage. No unrelated file cleanup, package installation, Git initialization, staging, commit, push or publication was performed.", "",
        "Next milestone: review the scientific interpretation and the user's own contribution before publication. Any later model changes are exploratory relative to this now-inspected test; a confirmatory extension needs a fresh evaluation protocol and dataset.", ""]
    (directory / "FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def ledger(result, directory):
    path = ROOT / "experiments/experiment_log.csv"
    existing = path.read_text(encoding="utf-8")
    fields = next(csv.reader(io.StringIO(existing)))
    require("final_test_" not in existing, "final evaluation already logged")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    for r in result["outcomes"]:
        for metric in ("accuracy", "macro_f1"):
            writer.writerow({"run_id": "final_test_" + r["readout_id"] + "_" + metric,
                "run_date_utc": result["executed_at_utc"], "git_sha": "",
                "dataset_version": "UCI80 test " + result["test_sha256"],
                "split_protocol": "frozen 3058 train / 765 validation; official 1797 test; no refit on validation",
                "seed": r.get("seed", int(r["parent_run"].split("seed")[-1]) if r.get("parent_run") else 2027), "model_name": r["readout_id"],
                "config_path": "results/final/final_results.json", "hardware": "Windows CPU; threads=1",
                "metric_name": "test_" + metric, "metric_value": r["test"][metric],
                "runtime_seconds": r["test_predict_seconds"] + r.get("fit_and_validation_seconds",0),
                "artifact_path": "results/final/final_results.json", "notes": r["role"] + "; same test/shared models; not independent replications; runtime excludes imports and neural training"})
    (directory / "experiment_log_updated.csv").write_text(existing.rstrip("\n") + "\n" + buffer.getvalue(), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build-ledger", action="store_true", help="produce ledger update for first integration only")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result, gallery = verify(args.results_dir)
    render(result, gallery, args.output_dir)
    report(result, args.output_dir)
    if args.build_ledger:
        ledger(result, args.output_dir)
    checks = {"status": "PASS", "source_sha256": sha(__file__), "prediction_rows": 1797,
        "readouts_verified": 21, "primary_rankings_verified": 4, "group_summaries_verified": True,
        "error_gallery_policy_verified": True, "no_model_fitting_or_inference": True}
    (args.output_dir / "saved_results_verification.json").write_text(json.dumps(checks,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(checks,indent=2))


if __name__ == "__main__":
    main()
