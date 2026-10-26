                                                                                         

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from si_audit.fashion_data import load_training_pool,validate_manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",type=Path,required=True)
    args=parser.parse_args()
    folder=ROOT/"results/fashion"
    manifest=json.loads((folder/"data/split_manifest.json").read_text(encoding="utf-8"))
    audit_path=folder/"data/data_audit.json"
    audit=json.loads(audit_path.read_text(encoding="utf-8"))
    preflight=json.loads((folder/"preflight/real_preflight.json").read_text(encoding="utf-8"))
    config=json.loads((ROOT/"configs/fashion_cnn_v1.json").read_text(encoding="utf-8"))
    for document in (audit,preflight):
        for relative,expected in document["source_sha256"].items():
            if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()!=expected:
                raise ValueError("source/config changed: "+relative)
    payload=audit_path.read_bytes()
    original_hash=preflight["data_audit_sha256"]
    mode="byte-identical"
    if hashlib.sha256(payload).hexdigest()!=original_hash:
        if b"\r" in payload or hashlib.sha256(payload.replace(b"\n",b"\r\n")).hexdigest()!=original_hash:
            raise ValueError("data audit changed beyond declared newline normalization")
        mode="exact original CRLF reconstructed from delivered LF"
    pool=load_training_pool(args.data_dir)                                     
    validate_manifest(pool,manifest,config)
    a,b=[set(manifest["budget_rows"][k]) for k in ("A","B")]
    val=set(manifest["validation_rows"])
    if not (len(a)==3000 and len(b)==12000 and len(val)==6000 and a<b and not b&val):
        raise ValueError("invalid actual budget/validation identities")
    for key,expected in (("si_train_rows",60),("si_validation_rows",60),("si_validation_sensitivity_rows",200)):
        if not np.all(np.bincount(pool.labels[manifest[key]],minlength=10)==expected):
            raise ValueError("unbalanced SI subset")
    if not set(manifest["si_train_rows"])<=a or not set(manifest["si_validation_rows"])<=set(manifest["si_validation_sensitivity_rows"])<=val:
        raise ValueError("SI subset leakage")
    if audit["exact_duplicate_audit"]["duplicate_groups"]:
        raise ValueError("duplicate audit requires review")
    if not (audit["manifest_sha256"]==manifest["manifest_sha256"]==preflight["manifest_sha256"]):
        raise ValueError("manifest provenance mismatch")
    tiny=preflight["tiny_batch_overfit"]
    tiny_rows=tiny["training_rows"]
    if not (tiny["passed"] and len(tiny_rows)==50 and set(tiny_rows)<=a and
            np.all(np.bincount(pool.labels[tiny_rows],minlength=10)==5) and tiny["checks"][-1]["tiny_training_accuracy"]>=.95):
        raise ValueError("tiny-batch gate mismatch")
    runs=preflight["two_epoch_timing_runs"]
    if {r["setting"] for r in runs}!={"A","B"} or len({r["initial_model_state_sha256"] for r in runs})!=1:
        raise ValueError("unpaired timing models")
    projection=float(sum(np.mean(r["epoch_seconds"]) for r in runs)*config["epochs"]*len(config["seeds"]))
    if projection!=preflight["projected_six_run_training_seconds"] or 2*projection!=preflight["twofold_margin_training_seconds"]:
        raise ValueError("projection arithmetic mismatch")
    if preflight["test_files_opened"] or preflight["validation_or_test_performance_evaluated"] or preflight["weights_saved"]:
        raise ValueError("unexpected preflight scope")
    result={"status":"PASS","source_hashes_verified":True,"manifest_sha256":manifest["manifest_sha256"],
        "nested_budget_ids_verified":True,"si_subset_balance_verified":True,
        "tiny_batch_gate_and_training_cost_verified":True,"audit_packaging":mode,
        "no_model_fitting_or_inference":True,"test_files_opened":False,
        "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
