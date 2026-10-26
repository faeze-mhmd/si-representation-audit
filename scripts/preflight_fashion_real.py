                                                                                   

import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter

os.environ["OMP_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
os.environ["LOKY_MAX_CPU_COUNT"]="1"

import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from si_audit.cnn import build_fashion_cnn
from si_audit.fashion_data import load_training_pool,validate_manifest
from si_audit.metrics import separation_index
from si_audit.training import make_loader,train_epoch,model_state_hash


def tensor_images(pool,rows):
    return torch.from_numpy(pool.images[rows].astype(np.float32)[:,None,:,:]/255.0)


def parts_batched(model,images,batch=256):
    collected={}
    model.eval()
    with torch.no_grad():
        for start in range(0,len(images),batch):
            for name,value in model.forward_parts(images[start:start+batch]).items():
                collected.setdefault(name,[]).append(value.numpy())
    return {name:np.concatenate(values) for name,values in collected.items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",type=Path,required=True)
    parser.add_argument("--audit-dir",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    start=perf_counter()
    config=json.loads((ROOT/"configs/fashion_cnn_v1.json").read_text(encoding="utf-8"))
    manifest=json.loads((args.audit_dir/"split_manifest.json").read_text(encoding="utf-8"))
    audit=json.loads((args.audit_dir/"data_audit.json").read_text(encoding="utf-8"))
                                                                              
    if audit["manifest_sha256"]!=manifest["manifest_sha256"] or audit["exact_duplicate_audit"]["duplicate_groups"]:
        raise ValueError("review duplicate/provenance findings before real-data preflight")
    pool=load_training_pool(args.data_dir)                               
    validate_manifest(pool,manifest,config)
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    tiny=[]
    core=np.array(manifest["budget_rows"]["A"])
    for label in range(10):
        tiny.extend(core[pool.labels[core]==label][:5].tolist())
    tiny=np.array(sorted(tiny))
    tx,ty=tensor_images(pool,tiny),torch.from_numpy(pool.labels[tiny].copy())
    small=build_fashion_cnn(11)
    optimizer=torch.optim.AdamW(small.parameters(),lr=config["learning_rate"],weight_decay=config["weight_decay"])
    checks=[]
    tick=perf_counter()
    with threadpool_limits(limits=1):
        for step in range(401):
            if step%20==0:
                with torch.no_grad():
                    logits=small(tx)
                    accuracy=float((logits.argmax(1)==ty).float().mean())
                    ce=float(torch.nn.functional.cross_entropy(logits,ty))
                checks.append({"step":step,"tiny_training_accuracy":accuracy,"cross_entropy":ce})
                if accuracy>=.95:
                    break
            if step==400:
                break
            optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.cross_entropy(small(tx),ty)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite tiny-batch loss")
            loss.backward()
            optimizer.step()
        tiny_result={"selection":"first five original row IDs per class within A; fixed before outcomes",
            "training_rows":tiny.tolist(),"n":50,"max_steps":400,"check_every":20,"required_accuracy":.95,
            "checks":checks,"passed":checks[-1]["tiny_training_accuracy"]>=.95,
            "runtime_seconds":perf_counter()-tick,
            "interpretation":"memorization sanity only; in-sample, throwaway weights, not validation/generalization evidence"}
        runs=[]
        if tiny_result["passed"]:
            for setting,rows in manifest["budget_rows"].items():
                rows=np.array(rows)
                images=tensor_images(pool,rows)
                labels=torch.from_numpy(pool.labels[rows].copy())
                model=build_fashion_cnn(11)
                initial_hash=model_state_hash(model)
                optimizer=torch.optim.AdamW(model.parameters(),lr=config["learning_rate"],weight_decay=config["weight_decay"])
                loader,_=make_loader(images,labels,pool.sample_ids[rows],seed=11,batch_size=config["batch_size"])
                times=[]
                orders=[]
                for _ in range(2):
                    seconds,order=train_epoch(model,loader,optimizer)
                    times.append(seconds)
                    orders.append(order)
                runs.append({"setting":setting,"seed":11,"n":len(rows),"epochs":2,
                    "steps_per_epoch":len(loader),"epoch_seconds":times,"epoch_order_sha256":orders,
                    "initial_model_state_sha256":initial_hash})
            if runs[0]["initial_model_state_sha256"]!=runs[1]["initial_model_state_sha256"]:
                raise ValueError("unpaired timing-model initialization")
                                                                                
            selected=np.array(manifest["si_train_rows"])
            measure_start=perf_counter()
            parts=parts_batched(model,tensor_images(pool,selected))
            extraction_seconds=perf_counter()-measure_start
            si_checks={}
            for layer in ("input","conv_features","penultimate"):
                tick=perf_counter()
                measured=separation_index(parts[layer],pool.labels[selected],sample_ids=pool.sample_ids[selected],batch_size=config["si_distance_batch_size"])
                si_checks[layer]={"n":600,"features":parts[layer].shape[1],"seconds":perf_counter()-tick,
                    "si":measured.score,"tie_fraction":measured.tie_fraction,
                    "duplicate_vector_fraction":measured.duplicate_vector_fraction,
                    "role":"training-only timing sanity of a discarded two-epoch model; not a study result"}
            projection=sum(np.mean(r["epoch_seconds"]) for r in runs)*config["epochs"]*len(config["seeds"])
        else:
            extraction_seconds,si_checks,projection=None,{},None
    files=["scripts/preflight_fashion_real.py","src/si_audit/fashion_data.py","src/si_audit/cnn.py",
           "src/si_audit/metrics.py","src/si_audit/training.py","configs/fashion_cnn_v1.json","FASHION_CNN_PROTOCOL.md"]
    result={"executed_at_utc":datetime.now(timezone.utc).isoformat(),"stage":"real-training-only preflight",
        "environment":{"torch":torch.__version__,"numpy":np.__version__,"device":"cpu","threads":torch.get_num_threads()},
        "manifest_sha256":manifest["manifest_sha256"],"data_audit_sha256":hashlib.sha256((args.audit_dir/"data_audit.json").read_bytes()).hexdigest(),
        "source_sha256":{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files},
        "tiny_batch_overfit":tiny_result,"two_epoch_timing_runs":runs,"training_si_timing":si_checks,
        "si600_feature_extraction_seconds":extraction_seconds,
        "projected_six_run_training_seconds":float(projection) if projection is not None else None,
        "twofold_margin_training_seconds":float(2*projection) if projection is not None else None,
        "training_budget_seconds":config["training_budget_seconds"],
        "training_projection_passed":bool(projection is not None and 2*projection<=config["training_budget_seconds"]),
        "cost_limits":"Two real epochs/budget estimate only. Excludes main validation/probes, full SI schedule, checkpoints and interruptions. Baseline/probe cost not yet certified.",
        "runtime_after_imports_seconds":perf_counter()-start,"test_files_opened":False,
        "validation_or_test_performance_evaluated":False,"weights_saved":False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"tiny_batch":tiny_result,"timing_runs":runs,
        "training_projection_with_margin":result["twofold_margin_training_seconds"],"training_si_timing":si_checks,
        "training_gate_passed":result["training_projection_passed"]},indent=2))
    if not tiny_result["passed"] or not result["training_projection_passed"]:
        raise SystemExit("Preflight gate failed; preserve artifact and investigate before main training.")


if __name__=="__main__":
    main()
