                                                                                       

import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

os.environ["LOKY_MAX_CPU_COUNT"]="1"
os.environ["OMP_NUM_THREADS"]="1"
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"

import numpy as np
import sklearn
from threadpoolctl import threadpool_limits
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from si_audit.cnn import build_fashion_cnn
from si_audit.fashion_data import load_training_pool,validate_manifest,digest
from si_audit.fashion_training import (prepare_data,evaluate_cnn,snapshot_cnn,fit_cnn_probes,
    cnn_bundle,cnn_from_bundle,save_json,save_checkpoint,run_cnn)
from si_audit.training import make_loader,train_epoch,model_state_hash


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance_now(manifest,baseline):
    files=["src/si_audit/cnn.py","src/si_audit/fashion_data.py","src/si_audit/fashion_readouts.py",
        "src/si_audit/fashion_training.py","src/si_audit/metrics.py","src/si_audit/baselines.py",
        "src/si_audit/training.py","scripts/run_fashion_experiment.py","tests/test_fashion_training.py",
        "tests/test_fashion_readouts.py","configs/fashion_cnn_v1.json","FASHION_CNN_PROTOCOL.md"]
    return {"phase":"Fashion-MNIST CNN fixed-budget train/validation v1",
        "manifest_sha256":manifest["manifest_sha256"],
        "manifest_file_sha256":sha(ROOT/"results/fashion/data/split_manifest.json"),
        "baseline_results_sha256":sha(ROOT/"results/fashion/baselines/baseline_results.json"),
        "source_sha256":{p:sha(ROOT/p) for p in files},
        "environment":{"python":platform.python_version(),"numpy":str(np.__version__),
                       "sklearn":str(sklearn.__version__),"torch":str(torch.__version__),"device":"cpu","threads":1}}


def calibrate(data,config,provenance,baseline,output):
    output.mkdir(parents=True,exist_ok=True)
    path=output/"measurement_gate.json"
    if path.exists():
        old=json.loads(path.read_text(encoding="utf-8"))
        if old["provenance"]!=provenance:
            raise ValueError("calibration source changed; preserve output and use a fresh directory")
        print(json.dumps(old,indent=2))
        return
    item=data["B"]
    model=build_fashion_cnn(11)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config["learning_rate"],weight_decay=config["weight_decay"])
    loader,generator=make_loader(item["train"]["x"],item["train"]["y"],item["train"]["ids"],seed=11,batch_size=config["batch_size"])
    start=perf_counter()
    with threadpool_limits(limits=1):
        training_times=[train_epoch(model,loader,optimizer)[0] for _ in range(2)]
        tick=perf_counter()
        for split in ("train","validation"):
            evaluate_cnn(model,item[split]["x"],item[split]["y"],detailed=True)
        eval_seconds=perf_counter()-tick
        tick=perf_counter()
                                                                             
                                                                                  
        snapshot_cnn(model,item,config,config["primary_epoch"],{})
        snapshot_seconds=perf_counter()-tick
        tick=perf_counter()
        probes=fit_cnn_probes(model,item,config)
        probe_seconds=perf_counter()-tick
        tick=perf_counter()
        save_checkpoint(output/"calibration_checkpoint.pt",{"model":model.state_dict(),"optimizer":optimizer.state_dict(),"loader_rng":generator.get_state(),
            "torch_rng":torch.get_rng_state(),"provenance":provenance})
        save_json(output/"calibration_metadata.json",{"stage":"throwaway cost calibration","provenance":provenance})
        checkpoint_seconds=perf_counter()-tick
        tick=perf_counter()
        bundle=cnn_bundle(model,{"stage":"throwaway cost calibration"})
        save_json(output/"calibration_bundle.json",bundle)
        restored=cnn_from_bundle(bundle)
        if model_state_hash(restored)!=model_state_hash(model):
            raise ValueError("calibration export mismatch")
        export_seconds=perf_counter()-tick
    models=len(config["seeds"])*len(config["budgets"])
    measurement_estimate=(eval_seconds*models*(config["epochs"]+2)+snapshot_seconds*models*len(config["checkpoint_epochs"])
        +probe_seconds*models+checkpoint_seconds*models*(config["epochs"]+1)+export_seconds*models)
    margin=config["projection_safety_factor"]
    real_preflight=json.loads((ROOT/"results/fashion/preflight/real_preflight.json").read_text(encoding="utf-8"))
    training_with_margin=real_preflight["twofold_margin_training_seconds"]
    measurement_with_margin=margin*measurement_estimate+baseline["runtime_after_setup_seconds"]
    result={"executed_at_utc":datetime.now(timezone.utc).isoformat(),"provenance":provenance,
        "role":"throwaway B seed11 two-epoch cost calibration; no selected main model or reported accuracy/SI",
        "calibration_training_epoch_seconds":training_times,
        "measured_seconds":{"full_B_train_validation_eval":eval_seconds,"cold_full_snapshot_with_SI2000_and_controls":snapshot_seconds,
            "both_B_probes_with_extraction":probe_seconds,"checkpoint_and_metadata":checkpoint_seconds,"portable_export_restore":export_seconds},
        "probe_warnings":{p["model"]:p["warnings"] for p in probes},
        "projection_method":"All six models charged worst-budget eval/probe cost; all snapshots charged cold final SI/control/sensitivity cost, then 2x margin. Input cache will reduce actual work.",
        "projected_training_with_margin_seconds":training_with_margin,
        "projected_measurement_with_margin_and_baselines_seconds":measurement_with_margin,
        "training_cap_seconds":config["training_budget_seconds"],"measurement_cap_seconds":config["measurement_budget_seconds"],
        "gate_passed":training_with_margin<=config["training_budget_seconds"] and measurement_with_margin<=config["measurement_budget_seconds"],
        "runtime_after_setup_seconds":perf_counter()-start,"test_evaluated":False,
        "calibration_weights":"temporary timing payloads only; never reused by main conditions"}
    save_json(path,result)
    print(json.dumps(result,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",type=Path,required=True)
    parser.add_argument("--output-root",type=Path,required=True)
    parser.add_argument("--calibrate",action="store_true")
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config=json.loads((ROOT/"configs/fashion_cnn_v1.json").read_text(encoding="utf-8"))
    manifest=json.loads((ROOT/"results/fashion/data/split_manifest.json").read_text(encoding="utf-8"))
    baseline=json.loads((ROOT/"results/fashion/baselines/baseline_results.json").read_text(encoding="utf-8"))
    if baseline["test_evaluated"] or baseline["provenance"]["manifest_sha256"]!=manifest["manifest_sha256"] or len(baseline["models"])!=6:
        raise ValueError("baseline prerequisite mismatch")
    for name,expected in baseline["provenance"]["source_sha256"].items():
        if sha(ROOT/name)!=expected:
            raise ValueError("baseline source drift: "+name)
    pool=load_training_pool(args.data_dir)                             
    validate_manifest(pool,manifest,config)
    data={setting:prepare_data(pool,manifest,setting) for setting in config["budgets"]}
    provenance=provenance_now(manifest,baseline)
    if args.calibrate:
        calibrate(data,config,provenance,baseline,args.output_root)
        return
    freeze_path=ROOT/"results/fashion/TRAINING_FREEZE.json"
    freeze=json.loads(freeze_path.read_text(encoding="utf-8"))
    gate_path=ROOT/"results/fashion/preflight/measurement_gate.json"
    gate=json.loads(gate_path.read_text(encoding="utf-8"))
    if not gate["gate_passed"] or gate["provenance"]!=provenance or freeze["provenance"]!=provenance:
        raise ValueError("training freeze or calibrated source/config does not match")
    if freeze["measurement_gate_sha256"]!=sha(gate_path):
        raise ValueError("measurement gate changed since freeze")
    provenance={**provenance,"training_freeze_sha256":sha(freeze_path)}
    args.output_root.mkdir(parents=True,exist_ok=True)
    index={"config":config,"provenance":provenance,"runs":[],"test_evaluated":False,
           "baseline_runtime_seconds":baseline["runtime_after_setup_seconds"],"completed":False}
    started=perf_counter()
    cache={}
    for seed in config["seeds"]:
        pair=[]
        for setting in config["budgets"]:
            run_id=f"fashion_{setting}_seed{seed}"
            tick=perf_counter()
            record,disposition=run_cnn(data[setting],config,provenance,args.output_root/run_id,setting=setting,seed=seed,input_cache=cache)
            pair.append(record)
            index["runs"].append({"run_id":run_id,"record":run_id+"/metrics.json","bundle":run_id+"/model_bundle.json",
                "disposition":disposition,"invocation_seconds":perf_counter()-tick,
                "training_seconds":record["training_seconds"],"measurement_seconds":record["measurement_seconds"],
                "checkpoint_seconds":record["checkpoint_seconds"],"final_model_state_sha256":record["final_model_state_sha256"]})
            index["invocation_wall_seconds"]=perf_counter()-started
            save_json(args.output_root/"experiment_index.json",index)
            if sum(r["training_seconds"] for r in index["runs"])>config["training_budget_seconds"]:
                raise RuntimeError("aggregate training cap exceeded")
            if baseline["runtime_after_setup_seconds"]+sum(r["measurement_seconds"]+r["checkpoint_seconds"] for r in index["runs"])>config["measurement_budget_seconds"]:
                raise RuntimeError("aggregate measurement cap exceeded")
            score=record["final_prediction"]["validation"]["accuracy"]
            print(f"DONE {run_id}: validation={score:.4f}, train={record['training_seconds']:.2f}s, measure={record['measurement_seconds']:.2f}s",flush=True)
        if pair[0]["initial_model_state_sha256"]!=pair[1]["initial_model_state_sha256"]:
            raise ValueError("paired CNN initial tensors differ")
    index["completed"]=True
    index["completed_at_utc"]=datetime.now(timezone.utc).isoformat()
    index["invocation_wall_seconds"]=perf_counter()-started
    save_json(args.output_root/"experiment_index.json",index)
    print("PASS: all six fixed CNN conditions completed; no test access.")


if __name__=="__main__":
    main()
