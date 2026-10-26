                                                                                       

import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import sklearn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from si_audit.fashion_data import load_training_pool,load_official_test_for_qa,make_manifest,validate_manifest,exact_duplicate_audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    start=perf_counter()
    config=json.loads((ROOT/"configs/fashion_cnn_v1.json").read_text(encoding="utf-8"))
    training=load_training_pool(args.data_dir)
    test=load_official_test_for_qa(args.data_dir)
    manifest=make_manifest(training,config)
    validate_manifest(training,manifest,config)
    duplicates=exact_duplicate_audit(training,test,manifest)
    def counts(labels):
        return {str(i):int(np.sum(labels==i)) for i in range(10)}
    source_files=["src/si_audit/fashion_data.py","scripts/prepare_fashion_data.py","configs/fashion_cnn_v1.json","FASHION_CNN_PROTOCOL.md"]
    result={"executed_at_utc":datetime.now(timezone.utc).isoformat(),
        "purpose":"schema/split/duplicate QA; no model predictions or test performance",
        "environment":{"python":platform.python_version(),"numpy":np.__version__,"sklearn":sklearn.__version__},
        "acquisition":json.loads((args.data_dir/"acquisition.json").read_text(encoding="utf-8-sig")),
        "official_training":{"n":len(training.labels),"class_counts":counts(training.labels),"provenance":training.provenance},
        "official_test":{"n":len(test.labels),"class_counts":counts(test.labels),"provenance":test.provenance},
        "split_counts":{"pool":manifest["pool_n"],"validation":len(manifest["validation_rows"]),
            "budgets":{k:len(v) for k,v in manifest["budget_rows"].items()},
            "train_si":len(manifest["si_train_rows"]),"validation_si":len(manifest["si_validation_rows"]),
            "validation_si_sensitivity":len(manifest["si_validation_sensitivity_rows"])},
        "manifest_sha256":manifest["manifest_sha256"],"exact_duplicate_audit":duplicates,
        "source_sha256":{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_files},
        "runtime_seconds":perf_counter()-start,"test_performance_evaluated":False}
    args.output_dir.mkdir(parents=True,exist_ok=True)
                                                                                 
    (args.output_dir/"split_manifest.json").write_text(json.dumps(manifest,separators=(",", ":"))+"\n",encoding="utf-8")
    (args.output_dir/"data_audit.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"split_counts":result["split_counts"],"duplicate_summary":{k:v for k,v in duplicates.items() if k!="groups"},
                      "runtime_seconds":result["runtime_seconds"]},indent=2))


if __name__=="__main__":
    main()
