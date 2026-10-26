                                                                                  

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import numpy as np
from scipy.stats import rankdata
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from si_audit.data import load_training_pool, validate_manifest
from si_audit.metrics import separation_index
from si_audit.training import canonical_hash, evaluate, model_from_bundle, model_state_hash, save_json


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rank_agreement(scores, outcome):
    scores, outcome = np.asarray(scores), np.asarray(outcome)
    ranks_x, ranks_y = rankdata(scores), rankdata(outcome)
    rho = None if np.ptp(ranks_x) == 0 or np.ptp(ranks_y) == 0 else float(np.corrcoef(ranks_x, ranks_y)[0, 1])
    counts = {"concordant_pairs": 0, "discordant_pairs": 0, "tied_pairs": 0}
    for i in range(len(scores)):
        for j in range(i + 1, len(scores)):
            product = (scores[i] - scores[j]) * (outcome[i] - outcome[j])
            key = "tied_pairs" if product == 0 else "concordant_pairs" if product > 0 else "discordant_pairs"
            counts[key] += 1
    return {"spearman_rho": rho, "n_models": len(scores), **counts,
            "interpretation": "descriptive; shared data and paired seeds; no p-value or independence claim; validation SI uses overlapping labelled validation examples"}


def check_confusion(summary):
    matrix = np.array(summary["confusion_matrix"])
    require(int(matrix.sum()) == summary["n"], "confusion sample count mismatch")
    require(abs(float(np.trace(matrix) / matrix.sum()) - summary["accuracy"]) < 1e-12, "confusion accuracy mismatch")


def save_figure(figure, directory, name):
    figure.savefig(directory / (name + ".svg"), metadata={"Date": None}, facecolor="white")
    figure.savefig(directory / (name + ".png"), dpi=180, facecolor="white")


def figures(records, summary, gallery, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "font.family": "DejaVu Sans"})
    colors = {"A": "#2563a6", "B": "#b34b25"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for setting in ("A", "B"):
        group = [r for r in records if r["setting"] == setting]
        for split, style in (("train", "--"), ("validation", "-")):
            values = np.array([[h[split]["accuracy"] for h in r["history"]] for r in group])
            epochs = [h["epoch"] for h in group[0]["history"]]
            axes[0].plot(epochs, values.mean(axis=0), style, color=colors[setting], label=f"{setting}: {split}")
            axes[0].fill_between(epochs, values.min(axis=0), values.max(axis=0), color=colors[setting], alpha=.09)
            checkpoints = [0, 5, 20, 50]
            values = np.array([[r["snapshots"][str(e)][split]["representations"]["penultimate"]["si"] for e in checkpoints] for r in group])
            axes[1].plot(checkpoints, values.mean(axis=0), style, marker="o", color=colors[setting], label=f"{setting}: {split}")
            axes[1].fill_between(checkpoints, values.min(axis=0), values.max(axis=0), color=colors[setting], alpha=.09)
    axes[0].set(title="Prediction accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1.02))
    axes[1].set(title="Penultimate representation SI", xlabel="Observed checkpoint epoch", ylabel="SI")
    for ax in axes:
        ax.grid(alpha=.18)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("UCI digits pilot: mean and min–max range across 3 paired seeds")
    fig.text(.5, .02, "A: weight decay 0; B: 0.01. Shading is seed range, not a confidence interval. No final test results.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .94))
    save_figure(fig, output, "learning_curves")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    layers = ["input", "hidden1", "penultimate"]
    for setting in ("A", "B"):
        group = [r for r in records if r["setting"] == setting]
        for split, style in (("train", "--"), ("validation", "-")):
            values = np.array([[r["snapshots"]["50"][split]["representations"][layer]["si"] for layer in layers] for r in group])
            ax.plot(range(3), values.mean(axis=0), style, marker="o", color=colors[setting], label=f"{setting}: {split}")
            ax.fill_between(range(3), values.min(axis=0), values.max(axis=0), color=colors[setting], alpha=.1)
    ax.set(xticks=range(3), xticklabels=["Input (64)", "Hidden ReLU (64)", "Penultimate ReLU (32)"],
           ylabel="SI on fixed 600-example subsets", title="Final-epoch layer geometry: mean and seed range")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(alpha=.2)
    ax.legend(fontsize=9)
    fig.text(.5, .02, "Raw Euclidean geometry; no learned feature scaling. SI is a local label-agreement measure.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1))
    save_figure(fig, output, "layer_si")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    offsets = [(5, 8), (5, -12), (-34, 9), (6, 10), (-34, -14), (6, -15)]
    for axis, split in zip(axes, ("train", "validation")):
        for i, r in enumerate(records):
            x = r["snapshots"]["50"][split]["representations"]["penultimate"]["si"]
            y = r["final_prediction"]["validation"]["accuracy"]
            axis.scatter(x, y, color=colors[r["setting"]], marker="o" if r["setting"] == "A" else "s", s=65,
                         facecolors="none" if r["setting"] == "A" else colors[r["setting"]])
            axis.annotate(r["run_id"].replace("_seed", ""), (x, y), xytext=offsets[i], textcoords="offset points", fontsize=8)
        axis.set(xlabel=f"{split.capitalize()} SI at epoch 50", title=f"{split.capitalize()} SI vs validation accuracy")
        axis.xaxis.set_major_formatter(PercentFormatter(1))
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.margins(x=.35, y=.4)
        axis.grid(alpha=.2)
    axes[0].set_ylabel("Full validation accuracy (765 examples)")
    fig.text(.5, .02, "Six related models; descriptive comparison only. Validation SI uses labels from an overlapping validation subset.", ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, .06, 1, 1))
    save_figure(fig, output, "si_vs_validation")
    plt.close(fig)

    fig, axes = plt.subplots(3, 4, figsize=(8, 6.5))
    for axis in axes.ravel(): axis.axis("off")
    for axis, item in zip(axes.ravel(), gallery):
        axis.imshow(np.array(item["features"]).reshape(8, 8), cmap="gray_r", vmin=0, vmax=16, interpolation="nearest")
        axis.set_title(f"ID {item['sample_id']}: {item['true']} → {item['predicted']}", fontsize=9)
    fig.suptitle("Validation mistakes: A, seed 11 — first 12 by original row ID")
    if not gallery:
        axes[0, 0].text(.1, .5, "No validation errors in this run.")
    fig.text(.5, .02, "Original UCI optdigits features; CC BY 4.0. Preselected run; no test images.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .94))
    save_figure(fig, output, "validation_errors")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--data-artifact", type=Path, default=ROOT / "results/data_audit.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(args.output_dir / "mplconfig")
    torch.set_num_threads(1)
    index = json.loads((args.pilot_root / "pilot_index.json").read_text(encoding="utf-8"))
    config, provenance = index["config"], index["provenance"]
    require(index["pilot_runs_completed"] == 6 and not index["test_evaluated"], "wrong pilot stage")
    for relative, expected in provenance["source_sha256"].items():
        require(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, "source changed: " + relative)
    require(hashlib.sha256((ROOT / "configs/pilot_v1.json").read_bytes()).hexdigest() == provenance["config_file_sha256"], "config changed")
    data_bytes = args.data_artifact.read_bytes()
    require(hashlib.sha256(data_bytes).hexdigest() == provenance["data_artifact_sha256"], "split artifact changed")
    data_artifact = json.loads(data_bytes)
    manifest = data_artifact["split_manifest"]
    pool = load_training_pool(args.data_dir)
    validate_manifest(pool, manifest)
    data = {}
    for split, rows_key, si_key in (("train", "train_rows", "si_train_rows"), ("validation", "validation_rows", "si_validation_rows")):
        rows = np.array(manifest[rows_key])
        data[split] = {"rows": rows, "x": torch.from_numpy(pool.features[rows].astype(np.float32) / 16.0),
                       "y": torch.from_numpy(pool.labels[rows].copy()), "si_positions": np.searchsorted(rows, manifest[si_key])}
    records, table, gallery = [], [], []
    for entry in index["runs"]:
        record = json.loads((args.pilot_root / entry["record"]).read_text(encoding="utf-8"))
        require(record["completed"] and not record["test_evaluated"], "incomplete run or test was evaluated")
        require(record["provenance"] == provenance and record["config"] == config, "run provenance mismatch")
        require(record["fingerprint"] == canonical_hash({"config": config, "provenance": provenance,
                "setting": record["setting"], "seed": record["seed"]}), "fingerprint mismatch")
        require([h["epoch"] for h in record["history"]] == list(range(51)), "missing/duplicated training epoch")
        require(set(record["snapshots"]) == {"0", "5", "20", "50"}, "checkpoint schedule changed")
        require(len(record["epoch_order_sha256"]) == 50, "missing minibatch-order evidence")
        bundle = json.loads((args.pilot_root / entry["bundle"]).read_text(encoding="utf-8"))
        model = model_from_bundle(bundle)
        require(model_state_hash(model) == record["final_model_state_sha256"], "model state mismatch")
        for split, item in data.items():
            fresh, parts, predicted = evaluate(model, item["x"], item["y"], detailed=True)
            require(fresh == record["final_prediction"][split], "model reload changed prediction metrics")
            check_confusion(fresh)
            selected = item["si_positions"]
            si = separation_index(parts["penultimate"][selected], item["y"].numpy()[selected],
                                  sample_ids=pool.sample_ids[item["rows"]][selected])
            stored = record["snapshots"]["50"][split]["representations"]["penultimate"]
            require(si.score == stored["si"] and si.tie_fraction == stored["tie_fraction"]
                    and si.duplicate_vector_fraction == stored["duplicate_vector_fraction"], "reloaded primary SI differs")
            if record["run_id"] == config["error_gallery_run"] and split == "validation":
                errors = np.flatnonzero(item["y"].numpy() != predicted)[:config["error_gallery_count"]]
                gallery = [{"sample_id": int(pool.sample_ids[item["rows"][i]]),
                            "true": int(item["y"][i]), "predicted": int(predicted[i]),
                            "features": pool.features[item["rows"][i]].tolist()} for i in errors]
        records.append(record)
        row = {"run_id": record["run_id"], "setting": record["setting"], "seed": record["seed"], "weight_decay": record["weight_decay"],
               "train_accuracy": record["final_prediction"]["train"]["accuracy"],
               "validation_accuracy": record["final_prediction"]["validation"]["accuracy"],
               "validation_macro_f1": record["final_prediction"]["validation"]["macro_f1"],
               "validation_cross_entropy": record["final_prediction"]["validation"]["cross_entropy"],
               "train_si": record["snapshots"]["50"]["train"]["representations"]["penultimate"]["si"],
               "validation_si": record["snapshots"]["50"]["validation"]["representations"]["penultimate"]["si"],
               "training_seconds": record["training_seconds"], "measurement_seconds": record["measurement_seconds"],
               "probe_accuracy": {p["model"]: p["validation"]["accuracy"] for p in record["probes"]}}
        table.append(row)
    require({r["run_id"] for r in records} == {f"{s}_seed{seed}" for s in ("A", "B") for seed in (11, 22, 33)}, "wrong scientific conditions")
    for seed in config["seeds"]:
        pair = [r for r in records if r["seed"] == seed]
        require(pair[0]["initial_model_state_sha256"] == pair[1]["initial_model_state_sha256"], "unpaired initialization")
        require(pair[0]["epoch_order_sha256"] == pair[1]["epoch_order_sha256"], "unpaired minibatch order")
    outcome = [r["validation_accuracy"] for r in table]
    groups = {}
    for setting in ("A", "B"):
        selected = [r for r in table if r["setting"] == setting]
        groups[setting] = {metric: {"mean": float(np.mean([r[metric] for r in selected])),
                                   "sample_std": float(np.std([r[metric] for r in selected], ddof=1))}
                           for metric in ("validation_accuracy", "validation_macro_f1", "train_si", "validation_si")}
    pairs = []
    for seed in config["seeds"]:
        a = next(r for r in table if r["seed"] == seed and r["setting"] == "A")
        b = next(r for r in table if r["seed"] == seed and r["setting"] == "B")
        pairs.append({"seed": seed, **{metric + "_B_minus_A": b[metric] - a[metric]
                      for metric in ("validation_accuracy", "validation_cross_entropy", "train_si", "validation_si")}})
    within_setting = {setting: rank_agreement([r["validation_si"] for r in table if r["setting"] == setting],
                                              [r["validation_accuracy"] for r in table if r["setting"] == setting]) for setting in ("A", "B")}
    summary = {"analyzed_at_utc": datetime.now(timezone.utc).isoformat(), "scope": "validation preview only; no final test results",
               "n_unique_conditions": 6, "table": table, "groups": groups, "paired_differences": pairs,
               "validation_rank_agreement": {"train_si": rank_agreement([r["train_si"] for r in table], outcome),
                                             "validation_si": rank_agreement([r["validation_si"] for r in table], outcome),
                                             "negative_validation_loss": rank_agreement([-r["validation_cross_entropy"] for r in table], outcome)},
               "within_setting_validation_si_rank_agreement": within_setting,
               "verification": {"all_source_and_split_hashes_match": True, "all_model_bundles_reloaded": True,
                                "all_final_prediction_metrics_reproduced": True, "all_12_primary_SI_scores_reproduced": True,
                                "paired_initialization_and_minibatch_orders_match": True, "test_evaluated": False},
               "source_sha256": {"scripts/analyze_pilot.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
               "notes": ["Three seeds and a shared dataset are not independent datasets.",
                         "Validation SI uses 600 labelled examples overlapping the 765-example validation target.",
                         "Per-setting SI ties make within-setting Spearman undefined; do not replace with zero.",
                         "Label shuffling is a fixed-graph sanity control, not a significance test of trained representations."]}
    save_json(args.output_dir / "pilot_analysis.json", summary)
    save_json(args.output_dir / "validation_error_gallery.json", {"run_id": config["error_gallery_run"], "examples": gallery})
    figures(records, summary, gallery, args.output_dir)
    print(json.dumps({"saved": str(args.output_dir), "table": table, "groups": groups,
                      "validation_rank_agreement": summary["validation_rank_agreement"], "verification": summary["verification"]}, indent=2))


if __name__ == "__main__":
    main()
