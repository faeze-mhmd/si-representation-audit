                                                                                 

import argparse
import csv
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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from si_audit.fashion_data import load_training_pool,validate_manifest,digest
from si_audit.fashion_readouts import fixed_readouts,fit_readout


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    config=json.loads((ROOT/"configs/fashion_cnn_v1.json").read_text(encoding="utf-8"))
    manifest=json.loads((ROOT/"results/fashion/data/split_manifest.json").read_text(encoding="utf-8"))
    pool=load_training_pool(args.data_dir)
    validate_manifest(pool,manifest,config)
    source_files=["src/si_audit/fashion_readouts.py","src/si_audit/fashion_data.py","src/si_audit/baselines.py",
                  "scripts/run_fashion_baselines.py","configs/fashion_cnn_v1.json","FASHION_CNN_PROTOCOL.md"]
    provenance={"manifest_sha256":manifest["manifest_sha256"],"source_sha256":{
        p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_files}}
    fingerprint=digest(provenance)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if (args.output_dir/"baseline_results.json").exists():
        raise ValueError("baseline results already exist; use saved-results verification")
    val=np.array(manifest["validation_rows"])
    queries=pool.images[val].reshape(len(val),784).astype(np.float32)/255
    predictions={}
    records=[]
    start=perf_counter()
    for setting,rows in manifest["budget_rows"].items():
        rows=np.array(rows)
        features=pool.images[rows].reshape(len(rows),784).astype(np.float32)/255
        for name,estimator in fixed_readouts(config):
            readout_id="fashion_"+setting+"__input_"+name
            measured,predicted=fit_readout(estimator,features,pool.labels[rows],queries,pool.labels[val])
            errors=np.flatnonzero(predicted!=pool.labels[val])
            records.append({"readout_id":readout_id,"setting":setting,"training_n":len(rows),"model":name,
                **measured,"error_count":len(errors),"errors_first30":[{"sample_id":int(val[i]),"truth":int(pool.labels[val[i]]),
                    "predicted":int(predicted[i])} for i in errors[:30]]})
            predictions[readout_id]=predicted
            print(f"{readout_id}: validation={measured['validation']['accuracy']:.4f}, warnings={len(measured['warnings'])}",flush=True)
    csv_path=args.output_dir/"validation_predictions.csv"
    with csv_path.open("w",encoding="utf-8",newline="") as handle:
        writer=csv.writer(handle,lineterminator="\n")
        writer.writerow(["sample_id","truth",*predictions])
        for i,row in enumerate(val):
            writer.writerow([int(row),int(pool.labels[row]),*[int(p[i]) for p in predictions.values()]])
    result={"executed_at_utc":datetime.now(timezone.utc).isoformat(),"fingerprint":fingerprint,"provenance":provenance,
        "environment":{"python":platform.python_version(),"numpy":np.__version__,"sklearn":sklearn.__version__,"threads":1},
        "scope":"training-only fitting; fixed 6000-example validation; no test access",
        "normalization":"float32 /255.0, flattened 784; no fitted scaling/PCA",
        "knn_ties":"scikit-learn brute on ascending original training IDs; no cross-library tie equivalence claim",
        "models":records,"runtime_after_setup_seconds":perf_counter()-start,"test_evaluated":False,
        "predictions_sha256":hashlib.sha256(csv_path.read_bytes()).hexdigest()}
    (args.output_dir/"baseline_results.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8",newline="\n")
    print("PASS: six baseline records and all validation predictions saved.")


if __name__=="__main__":
    main()
