                                                                                      

import base64
import copy
from datetime import datetime,timezone
import hashlib
import io
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits
import torch

from .baselines import classification_summary
from .cnn import build_fashion_cnn
from .fashion_data import digest
from .fashion_readouts import fixed_readouts,fit_readout
from .metrics import separation_index
from .training import make_loader,train_epoch,model_state_hash,permutation_control


def save_json(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_suffix(path.suffix+".pending")
    pending.write_text(json.dumps(value,indent=2)+"\n",encoding="utf-8",newline="\n")
    pending.replace(path)


def save_checkpoint(path,value):
                                                                                  
    path=Path(path)
    pending=path.with_suffix(path.suffix+".pending")
    with pending.open("wb") as handle:
        torch.save(value,handle)
    pending.replace(path)


def cnn_bundle(model,metadata):
    stream=io.BytesIO()
    np.savez_compressed(stream,**{k:t.detach().cpu().numpy() for k,t in model.state_dict().items()})
    payload=stream.getvalue()
    return {"format":"fashion_cnn_numpy_npz_base64_v1","architecture":"SmallFashionCNN_v1",
        "metadata":metadata,"model_state_sha256":model_state_hash(model),
        "payload_sha256":hashlib.sha256(payload).hexdigest(),"payload_base64":base64.b64encode(payload).decode("ascii")}


def cnn_from_bundle(bundle):
    if bundle["format"]!="fashion_cnn_numpy_npz_base64_v1" or bundle["architecture"]!="SmallFashionCNN_v1":
        raise ValueError("wrong CNN bundle architecture/format")
    payload=base64.b64decode(bundle["payload_base64"],validate=True)
    if hashlib.sha256(payload).hexdigest()!=bundle["payload_sha256"]:
        raise ValueError("CNN bundle payload mismatch")
    model=build_fashion_cnn(0)
    expected=model.state_dict()
    with np.load(io.BytesIO(payload),allow_pickle=False) as arrays:
        if set(arrays.files)!=set(expected):
            raise ValueError("CNN tensor names differ")
        state={}
        for name,tensor in expected.items():
            value=arrays[name]
            if value.shape!=tuple(tensor.shape) or value.dtype!=np.float32 or not np.isfinite(value).all():
                raise ValueError("invalid CNN tensor: "+name)
            state[name]=torch.from_numpy(value.copy())
    model.load_state_dict(state,strict=True)
    if model_state_hash(model)!=bundle["model_state_sha256"]:
        raise ValueError("CNN tensor digest mismatch")
    return model.eval()


def prepare_data(pool,manifest,setting):
    result={}
    for split,rows,selected in (("train",manifest["budget_rows"][setting],manifest["si_train_rows"]),
                                ("validation",manifest["validation_rows"],manifest["si_validation_rows"])):
        rows=np.array(rows)
        positions=np.searchsorted(rows,selected)
        if not np.array_equal(rows[positions],selected):
            raise ValueError("SI selection outside declared split")
        result[split]={"rows":rows,"ids":pool.sample_ids[rows],
            "x":torch.from_numpy(pool.images[rows].astype(np.float32)[:,None,:,:]/255),
            "y":torch.from_numpy(pool.labels[rows].copy()),"si_positions":positions}
    rows=result["validation"]["rows"]
    sensitivity=np.searchsorted(rows,manifest["si_validation_sensitivity_rows"])
    if not np.array_equal(rows[sensitivity],manifest["si_validation_sensitivity_rows"]):
        raise ValueError("sensitivity subset outside validation")
    result["validation"]["sensitivity_positions"]=sensitivity
    return result


def evaluate_cnn(model,features,labels, *, detailed=False,batch_size=256):
    if len(features)!=len(labels) or len(labels)<1:
        raise ValueError("nonempty aligned CNN evaluation required")
    model.eval()
    predictions=[]
    total_loss=0.0
    with torch.no_grad():
        for start in range(0,len(features),batch_size):
            logits=model(features[start:start+batch_size])
            if not torch.isfinite(logits).all():
                raise ValueError("nonfinite CNN logits")
            total_loss+=float(torch.nn.functional.cross_entropy(logits,labels[start:start+batch_size],reduction="sum"))
            predictions.append(logits.argmax(1).numpy())
    predicted=np.concatenate(predictions)
    summary={"accuracy":float(np.mean(predicted==labels.numpy())),"cross_entropy":total_loss/len(labels)}
    if detailed:
        summary.update(classification_summary(labels.numpy(),predicted))
    return summary,predicted


def extract_cnn(model,features, *, layers=("input","conv_features","penultimate"),batch_size=256):
    model.eval()
    values={name:[] for name in layers}
    with torch.no_grad():
        for start in range(0,len(features),batch_size):
            parts=model.forward_parts(features[start:start+batch_size])
            for name in layers:
                if not torch.isfinite(parts[name]).all():
                    raise ValueError("nonfinite CNN representation")
                values[name].append(parts[name].numpy())
    return {name:np.concatenate(chunks) for name,chunks in values.items()}


def snapshot_cnn(model,data,config,epoch,input_cache):
    result={}
    for split,item in data.items():
        selected=item["si_positions"]
        labels=item["y"].numpy()[selected]
        ids=item["ids"][selected]
        parts=extract_cnn(model,item["x"][selected])
        layers={}
        input_key=digest({"split":split,"ids":ids.tolist(),"labels":labels.tolist(),
                          "values_sha256":hashlib.sha256(parts["input"].tobytes()).hexdigest()})
        for layer,features in parts.items():
            if layer=="input" and input_key in input_cache:
                layers[layer]=copy.deepcopy(input_cache[input_key])
                continue
            measured=separation_index(features,labels,sample_ids=ids,batch_size=config["si_distance_batch_size"])
            value={"si":measured.score,"n":len(labels),"features":features.shape[1],
                "tie_fraction":measured.tie_fraction,"duplicate_vector_fraction":measured.duplicate_vector_fraction}
            if layer=="penultimate" and epoch in (0,config["primary_epoch"]):
                value["label_permutation"]=permutation_control(labels,measured.nearest_neighbor_indices,
                    seed=config["permutation_seeds"][split],count=config["permutation_count"])
            layers[layer]=value
            if layer=="input": input_cache[input_key]=copy.deepcopy(value)
        result[split]={"representations":layers}
    if epoch==config["primary_epoch"]:
        item=data["validation"]
        selected=item["sensitivity_positions"]
        parts=extract_cnn(model,item["x"][selected],layers=("penultimate",))
        measured=separation_index(parts["penultimate"],item["y"].numpy()[selected],sample_ids=item["ids"][selected],
                                 batch_size=config["si_distance_batch_size"])
        result["validation"]["sensitivity_penultimate"]={"si":measured.score,"n":len(selected),
            "features":32,"tie_fraction":measured.tie_fraction,"duplicate_vector_fraction":measured.duplicate_vector_fraction}
    return result


def fit_cnn_probes(model,data,config):
    parts={split:extract_cnn(model,item["x"],layers=("penultimate",))["penultimate"] for split,item in data.items()}
    records=[]
    for name,estimator in fixed_readouts(config,probes=True):
        result,_=fit_readout(estimator,parts["train"],data["train"]["y"].numpy(),
                            parts["validation"],data["validation"]["y"].numpy())
        records.append({"model":name,**result})
    return records


def restore_checkpoint(path,model,optimizer,generator,expected_fingerprint):
    saved=torch.load(path,map_location="cpu",weights_only=True)
    if saved["fingerprint"]!=expected_fingerprint or saved["record"]["fingerprint"]!=expected_fingerprint:
        raise ValueError("checkpoint source/config/split fingerprint mismatch")
    model.load_state_dict(saved["model"],strict=True)
    if model_state_hash(model)!=saved["model_state_sha256"]:
        raise ValueError("checkpoint tensor digest mismatch")
    optimizer.load_state_dict(saved["optimizer"])
    generator.set_state(saved["loader_rng"])
    torch.set_rng_state(saved["torch_rng"])
    return saved["record"]


def run_cnn(data,config,provenance,directory, *, setting,seed,input_cache=None,stop_after_epoch=None):
                                                                                         
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    record_path=directory/"metrics.json"
    checkpoint_path=directory/"resume.pt"
    fingerprint=digest({"config":config,"provenance":provenance,"setting":setting,"seed":seed})
    if record_path.exists():
        existing=json.loads(record_path.read_text(encoding="utf-8"))
        if existing["fingerprint"]!=fingerprint:
            raise ValueError("run directory belongs to another source/config/split")
        if existing["completed"]:
            bundle=json.loads((directory/"model_bundle.json").read_text(encoding="utf-8"))
            if bundle["metadata"]["fingerprint"]!=fingerprint or bundle["payload_sha256"]!=existing["model_bundle_payload_sha256"]:
                raise ValueError("completed CNN bundle provenance changed")
            if model_state_hash(cnn_from_bundle(bundle))!=existing["final_model_state_sha256"]:
                raise ValueError("completed CNN weights changed")
            return existing,"reused_completed"
    input_cache={} if input_cache is None else input_cache
    with torch.random.fork_rng(devices=[]),threadpool_limits(limits=1):
        torch.manual_seed(seed)
        model=build_fashion_cnn(seed)
        optimizer=torch.optim.AdamW(model.parameters(),lr=config["learning_rate"],weight_decay=config["weight_decay"])
        loader,generator=make_loader(data["train"]["x"],data["train"]["y"],data["train"]["ids"],
                                    seed=seed,batch_size=config["batch_size"])
        record={"run_id":f"fashion_{setting}_seed{seed}","setting":setting,"seed":seed,
            "training_n":len(data["train"]["y"]),"fingerprint":fingerprint,"config":config,"provenance":provenance,
            "initial_model_state_sha256":model_state_hash(model),"last_epoch":0,"history":[],"snapshots":{},
            "epoch_order_sha256":[],"training_seconds":0.0,"measurement_seconds":0.0,"checkpoint_seconds":0.0,
            "completed":False,"test_evaluated":False,"started_at_utc":datetime.now(timezone.utc).isoformat()}
        disposition="trained"
        if checkpoint_path.exists():
            record=restore_checkpoint(checkpoint_path,model,optimizer,generator,fingerprint)
            if record["config"]!=config or record["provenance"]!=provenance or record["completed"]:
                raise ValueError("partial checkpoint metadata mismatch")
            if len(record["epoch_order_sha256"])!=record["last_epoch"] or [h["epoch"] for h in record["history"]]!=list(range(record["last_epoch"]+1)):
                raise ValueError("partial checkpoint epoch/history mismatch")
                                                                                   
            if record_path.exists():
                latest=json.loads(record_path.read_text(encoding="utf-8"))
                comparable=lambda r:{k:v for k,v in r.items() if k!="checkpoint_seconds"}
                if latest["last_epoch"]>record["last_epoch"]:
                    raise ValueError("text record is ahead of saved optimizer state")
                if latest["last_epoch"]==record["last_epoch"] and comparable(latest)!=comparable(record):
                    raise ValueError("checkpoint and partial text record disagree")
                if latest["last_epoch"]==record["last_epoch"]:
                    record["checkpoint_seconds"]=latest["checkpoint_seconds"]
            disposition="resumed"

        def checkpoint():
            tick=perf_counter()
            save_checkpoint(checkpoint_path,{"fingerprint":fingerprint,"model":model.state_dict(),"model_state_sha256":model_state_hash(model),
                "optimizer":optimizer.state_dict(),"loader_rng":generator.get_state(),"torch_rng":torch.get_rng_state(),
                "record":record})
            record["checkpoint_seconds"]+=perf_counter()-tick
            save_json(record_path,record)

        def measure(epoch):
            tick=perf_counter()
            observed={split:evaluate_cnn(model,item["x"],item["y"])[0] for split,item in data.items()}
            record["history"].append({"epoch":epoch,**observed})
            if epoch in config["checkpoint_epochs"]:
                record["snapshots"][str(epoch)]=snapshot_cnn(model,data,config,epoch,input_cache)
            record["measurement_seconds"]+=perf_counter()-tick

        if not record["history"]:
            measure(0)
            checkpoint()
        steps={k:math.ceil(n/config["batch_size"])*config["epochs"] for k,n in config["budgets"].items()}
        per_run_training_cap=config["training_budget_seconds"]*steps[setting]/(sum(steps.values())*len(config["seeds"]))
        per_run_measurement_cap=config["measurement_budget_seconds"]/(len(config["seeds"])*len(config["budgets"]))
        for epoch in range(record["last_epoch"]+1,config["epochs"]+1):
            seconds,order=train_epoch(model,loader,optimizer)
            record["training_seconds"]+=seconds
            record["epoch_order_sha256"].append(order)
            measure(epoch)
            record["last_epoch"]=epoch
            checkpoint()
            if record["training_seconds"]>per_run_training_cap or record["measurement_seconds"]+record["checkpoint_seconds"]>per_run_measurement_cap:
                raise RuntimeError("predeclared per-run resource cap exceeded; preserve checkpoint and revise consistently")
            if stop_after_epoch is not None and epoch>=stop_after_epoch and epoch<config["epochs"]:
                return record,"interrupted_for_test"
        tick=perf_counter()
        record["final_prediction"]={split:evaluate_cnn(model,item["x"],item["y"],detailed=True)[0] for split,item in data.items()}
        record["probes"]=fit_cnn_probes(model,data,config)
        record["final_model_state_sha256"]=model_state_hash(model)
        bundle=cnn_bundle(model,{"run_id":record["run_id"],"fingerprint":fingerprint,"epoch":config["epochs"]})
        restored=cnn_from_bundle(bundle)
        with torch.no_grad():
            if not torch.equal(model(data["validation"]["x"][:32]),restored(data["validation"]["x"][:32])):
                raise ValueError("portable CNN round-trip changed logits")
        save_json(directory/"model_bundle.json",bundle)
        record["model_bundle_payload_sha256"]=bundle["payload_sha256"]
        record["measurement_seconds"]+=perf_counter()-tick
        if record["measurement_seconds"]+record["checkpoint_seconds"]>per_run_measurement_cap:
            raise RuntimeError("final probe/measurement work exceeded per-run resource cap")
        record["completed"]=True
        record["completed_at_utc"]=datetime.now(timezone.utc).isoformat()
        save_json(record_path,record)
        return record,disposition
