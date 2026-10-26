                                                                                 

import argparse
import csv
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter

os.environ["LOKY_MAX_CPU_COUNT"]="1"
os.environ["OMP_NUM_THREADS"]="1"
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"

import numpy as np
from threadpoolctl import threadpool_limits
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from si_audit.cnn import build_fashion_cnn
from si_audit.baselines import classification_summary
from si_audit.fashion_data import load_training_pool,validate_manifest,digest
from si_audit.fashion_training import cnn_from_bundle,prepare_data,evaluate_cnn,snapshot_cnn,extract_cnn,save_json
from si_audit.fashion_readouts import fixed_readouts,fit_readout
from si_audit.training import model_state_hash


def require(condition,message):
    if not condition: raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mean_sd(values):
    return {"n":len(values),"mean":float(np.mean(values)),"sample_sd":float(np.std(values,ddof=1))}


def verify(root,data_dir):
    index=json.loads((root/"experiment_index.json").read_text(encoding="utf-8"))
    config,provenance=index["config"],index["provenance"]
    expected={f"fashion_{s}_seed{seed}" for s in config["budgets"] for seed in config["seeds"]}
    require(index["completed"] and not index["test_evaluated"] and len(index["runs"])==6 and
            {e["run_id"] for e in index["runs"]}==expected,"incomplete/wrong CNN condition set")
    for relative,value in provenance["source_sha256"].items():
        require(sha(ROOT/relative)==value,"source drift: "+relative)
    freeze_path=ROOT/"results/fashion/TRAINING_FREEZE.json"
    require(sha(freeze_path)==provenance["training_freeze_sha256"],"training freeze changed")
    freeze=json.loads(freeze_path.read_text(encoding="utf-8"))
    base_provenance={k:v for k,v in provenance.items() if k!="training_freeze_sha256"}
    require(freeze["provenance"]==base_provenance,"freeze provenance mismatch")
    require(sha(ROOT/"results/fashion/preflight/measurement_gate.json")==freeze["measurement_gate_sha256"],"calibration changed")
    require(sha(ROOT/"results/fashion/data/split_manifest.json")==provenance["manifest_file_sha256"],"manifest bytes changed")
    require(sha(ROOT/"results/fashion/baselines/baseline_results.json")==provenance["baseline_results_sha256"],"baseline artifact changed")
    manifest=json.loads((ROOT/"results/fashion/data/split_manifest.json").read_text(encoding="utf-8"))
    pool=load_training_pool(data_dir)                          
    validate_manifest(pool,manifest,config)
    data={s:prepare_data(pool,manifest,s) for s in config["budgets"]}
    predictions={}
    baseline=json.loads((ROOT/"results/fashion/baselines/baseline_results.json").read_text(encoding="utf-8"))
    baseline_csv=ROOT/"results/fashion/baselines/validation_predictions.csv"
    require(sha(baseline_csv)==baseline["predictions_sha256"],"baseline predictions changed")
    with baseline_csv.open(encoding="utf-8") as handle: baseline_rows=list(csv.DictReader(handle))
    val_rows=data["A"]["validation"]["rows"]
    val_y=data["A"]["validation"]["y"].numpy()
    require([int(r["sample_id"]) for r in baseline_rows]==val_rows.tolist(),"baseline query IDs changed")
    require([int(r["truth"]) for r in baseline_rows]==val_y.tolist(),"baseline query labels changed")
    for item in baseline["models"]:
        predicted=np.array([int(r[item["readout_id"]]) for r in baseline_rows])
        require(classification_summary(val_y,predicted)==item["validation"],"baseline metric mismatch")
        predictions[item["readout_id"]]=predicted
    rows=[]
    records=[]
    gallery=[]
    input_cache={}
    for entry in index["runs"]:
        record=json.loads((root/entry["record"]).read_text(encoding="utf-8"))
        require(record["completed"] and not record["test_evaluated"],"wrong CNN stage")
        require(record["config"]==config and record["provenance"]==provenance,"CNN metadata changed")
        fingerprint=digest({"config":config,"provenance":provenance,"setting":record["setting"],"seed":record["seed"]})
        require(record["fingerprint"]==fingerprint,"CNN source/split fingerprint changed")
        require([h["epoch"] for h in record["history"]]==list(range(config["epochs"]+1)) and
                record["last_epoch"]==config["epochs"] and len(record["epoch_order_sha256"])==config["epochs"],"incomplete CNN epochs")
        require(model_state_hash(build_fashion_cnn(record["seed"]))==record["initial_model_state_sha256"],"initialization changed")
        bundle=json.loads((root/entry["bundle"]).read_text(encoding="utf-8"))
        require(bundle["metadata"]["fingerprint"]==fingerprint and bundle["metadata"]["epoch"]==config["epochs"],"bundle metadata changed")
        require(bundle["payload_sha256"]==record["model_bundle_payload_sha256"],"bundle payload changed")
        model=cnn_from_bundle(bundle)
        require(model_state_hash(model)==record["final_model_state_sha256"]==entry["final_model_state_sha256"],"CNN weights changed")
        item=data[record["setting"]]
        for split,values in item.items():
            summary,predicted=evaluate_cnn(model,values["x"],values["y"],detailed=True)
            require(summary==record["final_prediction"][split],"CNN prediction metrics changed")
            require(all(summary[k]==record["history"][-1][split][k] for k in ("accuracy","cross_entropy")),"final curve/metric mismatch")
            if split=="validation":
                predictions[record["run_id"]]=predicted
                if record["run_id"]==config["error_gallery_run"]:
                    selected=np.flatnonzero(predicted!=val_y)[:config["error_gallery_count"]]
                    gallery=[{"sample_id":int(val_rows[i]),"truth":int(val_y[i]),"predicted":int(predicted[i]),
                              "pixels":pool.images[val_rows[i]].tolist()} for i in selected]
        measured=snapshot_cnn(model,item,config,config["primary_epoch"],input_cache)
        require(measured==record["snapshots"][str(config["primary_epoch"])],"final CNN SI/control/sensitivity changed")
        parts={split:extract_cnn(model,values["x"],layers=("penultimate",))["penultimate"] for split,values in item.items()}
        previous={p["model"]:p for p in record["probes"]}
        for name,estimator in fixed_readouts(config,probes=True):
            replay,predicted=fit_readout(estimator,parts["train"],item["train"]["y"].numpy(),parts["validation"],val_y)
            require(all(replay[k]==previous[name][k] for k in ("validation","parameters","warnings","iterations")),"probe replay mismatch")
            predictions[record["run_id"]+"__"+name]=predicted
        snapshot=record["snapshots"][str(config["primary_epoch"])]
        rows.append({"run_id":record["run_id"],"setting":record["setting"],"seed":record["seed"],"training_n":record["training_n"],
            "train_accuracy":record["final_prediction"]["train"]["accuracy"],
            "validation":record["final_prediction"]["validation"],
            "train_si600":snapshot["train"]["representations"]["penultimate"]["si"],
            "validation_si600":snapshot["validation"]["representations"]["penultimate"]["si"],
            "validation_si2000":snapshot["validation"]["sensitivity_penultimate"]["si"],
            "probes":{p["model"]:{"accuracy":p["validation"]["accuracy"],"macro_f1":p["validation"]["macro_f1"],"warnings":p["warnings"]} for p in record["probes"]},
            "training_seconds":record["training_seconds"],"measurement_seconds":record["measurement_seconds"],
            "checkpoint_seconds":record["checkpoint_seconds"]})
        records.append(record)
        print("VERIFIED "+record["run_id"],flush=True)
    require(len(predictions)==24,"missing validation readouts")
    for seed in config["seeds"]:
        pair=[r for r in records if r["seed"]==seed]
        require(len({r["initial_model_state_sha256"] for r in pair})==1,"unpaired initial weights")
    return index,baseline,rows,records,gallery,predictions,val_rows,val_y


def render_figures(records,rows,gallery,output):
    os.environ["MPLCONFIGDIR"]=str(output/"mpl-cache")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"svg.fonttype":"none",
        "svg.hashsalt":"fashion-si-preview-v1","axes.spines.top":False,"axes.spines.right":False})
    folder=output/"figures"
    folder.mkdir(exist_ok=True)
    colors={"A":"#1864ab","B":"#d9480f"}
    def save(fig,name):
        fig.savefig(folder/(name+".svg"),metadata={"Date":None},facecolor="white")
        fig.savefig(folder/(name+".png"),dpi=150,facecolor="white")
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout="constrained")
    for setting in ("A","B"):
        selected=[r for r in records if r["setting"]==setting]
        epochs=[h["epoch"] for h in selected[0]["history"]]
        for split,style in (("train","--"),("validation","-")):
            for ax,metric in zip(axes,("accuracy","cross_entropy")):
                values=np.array([[h[split][metric] for h in r["history"]] for r in selected])
                ax.plot(epochs,values.mean(0),color=colors[setting],linestyle=style,label=f"{setting} {split}")
                ax.fill_between(epochs,values.min(0),values.max(0),color=colors[setting],alpha=.08)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set(ylabel="Accuracy",xlabel="Epoch",title="Observed learning curves")
    axes[1].set(ylabel="Cross-entropy",xlabel="Epoch",title="Observed learning curves")
    axes[0].legend(fontsize=8)
    fig.supxlabel("A: 3,000 training images; B: 12,000. Lines: seed means; shading: seed range, not confidence intervals.",fontsize=9)
    save(fig,"learning_curves")
    fig,axes=plt.subplots(1,3,figsize=(11,4),layout="constrained")
    for ax,key,title in zip(axes,("train_si600","validation_si600","validation_si2000"),
                            ("Train SI (600)","Validation SI (600, primary)","Validation SI (2,000, sensitivity)")):
        for r in rows:
            ax.scatter(r[key],r["validation"]["accuracy"],color=colors[r["setting"]],marker="o" if r["setting"]=="A" else "^")
            offsets={"A11":(-30,-16),"A22":(7,-15),"A33":(-30,8),
                     "B11":(7,6),"B22":(-30,8),"B33":(7,-16)}
            offset=offsets[r["setting"]+str(r["seed"])]
            ax.annotate(r["setting"]+str(r["seed"]),(r[key],r["validation"]["accuracy"]),xytext=offset,textcoords="offset points",fontsize=8)
        ax.set(xlabel=title,ylabel="Validation head accuracy")
        ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=1))
        ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=1))
        ax.margins(x=.3,y=.25)
        ax.grid(alpha=.15)
    fig.suptitle("Validation preview: six fixed CNNs; official test remains uninspected")
    fig.supxlabel("Validation SI uses labels overlapping the accuracy target. This is not independent performance-prediction evidence.",fontsize=9)
    save(fig,"si_vs_validation")
    fig,axes=plt.subplots(1,2,figsize=(9,4),layout="constrained")
    layers=("input","conv_features","penultimate")
    for ax,split in zip(axes,("train","validation")):
        for setting in ("A","B"):
            values=np.array([[r["snapshots"]["30"][split]["representations"][layer]["si"] for layer in layers]
                             for r in records if r["setting"]==setting])
            ax.plot(range(3),values.mean(0),"o-",color=colors[setting],label=setting)
            ax.fill_between(range(3),values.min(0),values.max(0),color=colors[setting],alpha=.12)
        ax.set(xticks=range(3),xticklabels=["Input\n784D","Pooled conv\n256D","Penultimate\n32D"],ylabel="SI on fixed 600 examples",title=split.capitalize())
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.legend()
    fig.suptitle("Epoch-30 layer geometry; seed ranges are descriptive")
    save(fig,"layer_si")
    fig,axes=plt.subplots(3,4,figsize=(8,6),layout="constrained")
    for ax in axes.flat: ax.axis("off")
    for ax,e in zip(axes.flat,gallery):
        ax.imshow(e["pixels"],cmap="gray_r",vmin=0,vmax=255,interpolation="nearest")
        ax.set_title(f"ID {e['sample_id']}: {e['truth']} -> {e['predicted']}",fontsize=9)
    fig.suptitle("A, seed 11: first 12 validation errors in original ID order\nDiagnostic gallery; no model selection or test outcomes",fontsize=11)
    save(fig,"validation_errors")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root",type=Path,required=True)
    parser.add_argument("--data-dir",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started=perf_counter()
    with threadpool_limits(limits=1):
        index,baseline,rows,records,gallery,predictions,ids,labels=verify(args.experiment_root,args.data_dir)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    groups={s:{metric:mean_sd([r["validation"][metric] for r in rows if r["setting"]==s]) for metric in ("accuracy","macro_f1")} for s in ("A","B")}
    pairs=[]
    for seed in index["config"]["seeds"]:
        a,b=[next(r for r in rows if r["setting"]==s and r["seed"]==seed) for s in ("A","B")]
        pairs.append({"seed":seed,"validation_accuracy_B_minus_A":b["validation"]["accuracy"]-a["validation"]["accuracy"],
                      "validation_si600_B_minus_A":b["validation_si600"]-a["validation_si600"]})
    path=args.output_dir/"validation_predictions.csv"
    with path.open("w",encoding="utf-8",newline="") as handle:
        writer=csv.writer(handle,lineterminator="\n")
        writer.writerow(["sample_id","truth",*predictions])
        for i in range(len(ids)):
            writer.writerow([int(ids[i]),int(labels[i]),*[int(p[i]) for p in predictions.values()]])
    analysis={"analyzed_at_utc":datetime.now(timezone.utc).isoformat(),"scope":"train/validation preview; no official test access",
        "source_sha256":sha(__file__),"table":rows,"groups":groups,"paired_B_minus_A":pairs,
        "training_seconds":sum(r["training_seconds"] for r in rows),"measurement_seconds":sum(r["measurement_seconds"] for r in rows),
        "checkpoint_seconds":sum(r["checkpoint_seconds"] for r in rows),"training_runner_wall_seconds":index["invocation_wall_seconds"],
        "baseline_seconds":baseline["runtime_after_setup_seconds"],"probes_with_warnings":sum(bool(p["warnings"]) for r in rows for p in r["probes"].values()),
        "validation_predictions_sha256":sha(path),"verification":{"source_freeze_and_manifest_checked":True,
            "models_reloaded":6,"primary_SI600_reproduced":12,"SI2000_reproduced":6,"all_final_layer_SI_and_controls_reproduced":True,
            "head_metrics_reproduced":12,"probe_metrics_reproduced":12,"baseline_metrics_verified":6,"validation_readouts_saved":24,
            "initialization_pairs_verified":3,"epoch_histories_verified":6,"test_evaluated":False},
        "verification_runtime_after_imports_seconds":perf_counter()-started}
    save_json(args.output_dir/"validation_analysis.json",analysis)
    save_json(args.output_dir/"validation_error_gallery.json",{"run_id":index["config"]["error_gallery_run"],"examples":gallery})
    render_figures(records,rows,gallery,args.output_dir)
    print(json.dumps({"groups":groups,"training_seconds":analysis["training_seconds"],"measurement_seconds":analysis["measurement_seconds"],
        "probes_with_warnings":analysis["probes_with_warnings"],"verification":analysis["verification"]},indent=2))


if __name__=="__main__":
    main()
