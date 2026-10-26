                                                                                      

from dataclasses import dataclass
import gzip
import hashlib
import io
import json
from pathlib import Path
import struct

import numpy as np
from sklearn.model_selection import train_test_split

PUBLISHED_MD5 = {
    "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
    "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
    "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
}


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class FashionData:
    images: np.ndarray
    labels: np.ndarray
    sample_ids: np.ndarray
    provenance: dict


def parse_idx_pair(image_bytes, label_bytes, *, expected_n=None, sample_id_start=0):
    if len(image_bytes)<16 or len(label_bytes)<8:
        raise ValueError("truncated IDX header")
    magic,n,height,width = struct.unpack(">IIII",image_bytes[:16])
    label_magic,label_n = struct.unpack(">II",label_bytes[:8])
    if magic!=2051 or label_magic!=2049 or (height,width)!=(28,28) or n!=label_n or n<1:
        raise ValueError("invalid IDX magic, dimensions or paired counts")
    if expected_n is not None and n!=expected_n:
        raise ValueError("unexpected official IDX count")
    if len(image_bytes)!=16+n*784 or len(label_bytes)!=8+n:
        raise ValueError("IDX payload length mismatch")
    if isinstance(sample_id_start,bool) or not isinstance(sample_id_start,int) or sample_id_start<0:
        raise ValueError("invalid sample ID offset")
    images=np.frombuffer(image_bytes,dtype=np.uint8,offset=16).reshape(n,28,28)
    labels=np.frombuffer(label_bytes,dtype=np.uint8,offset=8).astype(np.int64)
    if np.any(labels>9):
        raise ValueError("label outside 0..9")
    return FashionData(images,labels,np.arange(sample_id_start,sample_id_start+n,dtype=np.int64),
        {"decoded_images_sha256":hashlib.sha256(image_bytes).hexdigest(),
         "decoded_labels_sha256":hashlib.sha256(label_bytes).hexdigest()})


def _read_official(path, cap):
    payload=path.read_bytes()
    if hashlib.md5(payload).hexdigest()!=PUBLISHED_MD5[path.name]:
        raise ValueError("official compressed checksum mismatch: "+path.name)
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(payload)) as stream:
            decoded=stream.read(cap+1)
    except (OSError,EOFError) as error:
        raise ValueError("invalid gzip: "+path.name) from error
    if len(decoded)>cap:
        raise ValueError("IDX exceeds declared official payload size")
    return decoded,{"bytes":len(payload),"md5":hashlib.md5(payload).hexdigest(),
                    "sha256":hashlib.sha256(payload).hexdigest()}


def _load_pair(directory,prefix,n,offset):
    directory=Path(directory)
    image_name,label_name=prefix+"-images-idx3-ubyte.gz",prefix+"-labels-idx1-ubyte.gz"
    images,image_meta=_read_official(directory/image_name,16+n*784)
    labels,label_meta=_read_official(directory/label_name,8+n)
    data=parse_idx_pair(images,labels,expected_n=n,sample_id_start=offset)
    return FashionData(data.images,data.labels,data.sample_ids,
                       {**data.provenance,"files":{image_name:image_meta,label_name:label_meta}})


def load_training_pool(directory):
                                                              
    return _load_pair(directory,"train",60000,0)


def load_official_test_for_qa(directory):
                                                                             
    return _load_pair(directory,"t10k",10000,60000)


def _balanced_subset(rows,labels,per_class,seed):
    generator=np.random.default_rng(seed)
    chosen=[]
    for label in range(10):
        candidates=np.asarray(rows)[labels[rows]==label]
        if len(candidates)<per_class:
            raise ValueError("insufficient class for balanced subset")
        chosen.extend(generator.choice(candidates,per_class,replace=False).tolist())
    return sorted(chosen)


def make_manifest(data,config):
    labels=data.labels
    if len(labels)!=config["official_training_n"] or set(config["budgets"])!={"A","B"}:
        raise ValueError("unexpected training count or budget names")
    sizes=[config[k] for k in ("si_train_n","si_validation_primary_n","si_validation_sensitivity_n")]
    sizes.extend(config["budgets"].values())
    if any(isinstance(n,bool) or not isinstance(n,int) or n<=0 or n%10 for n in sizes):
        raise ValueError("balanced sizes must be positive multiples of ten")
    if config["budgets"]["A"]>config["budgets"]["B"] or config["si_train_n"]>config["budgets"]["A"]:
        raise ValueError("invalid nested budget sizes")
    if config["si_validation_primary_n"]>config["si_validation_sensitivity_n"]:
        raise ValueError("invalid nested validation SI sizes")
    rows=np.arange(len(labels),dtype=np.int64)
    if set(np.unique(labels))!=set(range(10)):
        raise ValueError("expected all ten classes")
    pool,val=train_test_split(rows,test_size=config["validation_n"],stratify=labels,random_state=config["split_seed"])
    pool,val=np.sort(pool),np.sort(val)
    generator=np.random.default_rng(config["training_budget_seed"])
    budgets={key:[] for key in config["budgets"]}
    for label in range(10):
        ordered=generator.permutation(pool[labels[pool]==label])
        for key,size in config["budgets"].items():
            if size%10 or size//10>len(ordered):
                raise ValueError("invalid balanced budget")
            budgets[key].extend(ordered[:size//10].tolist())
    budgets={key:sorted(value) for key,value in budgets.items()}
    sensitivity=_balanced_subset(val,labels,config["si_validation_sensitivity_n"]//10,config["si_validation_pool_seed"])
    primary=_balanced_subset(np.array(sensitivity),labels,config["si_validation_primary_n"]//10,config["si_validation_primary_seed"])
    train_si=_balanced_subset(np.array(budgets["A"]),labels,config["si_train_n"]//10,config["si_train_seed"])
    value={"version":1,"training_provenance":data.provenance,
        "configuration_sha256":digest(config),"official_training_n":len(labels),
        "pool_rows_definition":"ascending complement of validation_rows within [0, official_training_n)",
        "pool_n":len(pool),"pool_row_ids_sha256":hashlib.sha256(pool.astype("<i8").tobytes()).hexdigest(),
        "validation_rows":val.tolist(),"budget_rows":budgets,"si_train_rows":train_si,
        "si_validation_rows":primary,"si_validation_sensitivity_rows":sensitivity}
    return {**value,"manifest_sha256":digest(value)}


def validate_manifest(data,manifest,config):
    if manifest!=make_manifest(data,config):
        raise ValueError("manifest differs from deterministic source/config/split")


def exact_duplicate_audit(training,test,manifest):
                                                                                   
    validation=set(manifest["validation_rows"])
    budget_a=set(manifest["budget_rows"]["A"])
    budget_b=set(manifest["budget_rows"]["B"])
    groups={}
    for source,data in (("training",training),("test",test)):
        for row,image in enumerate(data.images):
            key=hashlib.sha256(image.tobytes()).hexdigest()
            roles=["test"] if source=="test" else ["validation" if row in validation else "pool"]
            if source=="training":
                if row in budget_a: roles.append("A")
                if row in budget_b: roles.append("B")
            groups.setdefault(key,[]).append({"source":source,"row":row,"sample_id":int(data.sample_ids[row]),
                                              "label":int(data.labels[row]),"roles":roles})
    duplicates=[]
    for key,members in groups.items():
        if len(members)<2:
            continue
        arrays=[(training if m["source"]=="training" else test).images[m["row"]] for m in members]
        if not all(np.array_equal(arrays[0],image) for image in arrays[1:]):
            raise ValueError("hash collision in exact duplicate audit")
        duplicates.append({"image_sha256":key,"conflicting_labels":len({m["label"] for m in members})>1,"members":members})
    boundaries={}
    for left,right in (("pool","validation"),("A","validation"),("B","validation"),
                       ("pool","test"),("validation","test"),("A","test"),("B","test")):
        selected=[g for g in duplicates if any(left in m["roles"] for m in g["members"])
                  and any(right in m["roles"] for m in g["members"])]
        boundaries[left+"__"+right]={"groups":len(selected),"pairs":sum(
            sum(left in m["roles"] for m in g["members"])*sum(right in m["roles"] for m in g["members"]) for g in selected),
            "conflicting_label_groups":sum(g["conflicting_labels"] for g in selected)}
    return {"method":"SHA256 candidates followed by exact uint8 image equality; no near-duplicate audit",
        "duplicate_groups":len(duplicates),"rows_in_duplicate_groups":sum(len(g["members"]) for g in duplicates),
        "conflicting_label_groups":sum(g["conflicting_labels"] for g in duplicates),
        "boundaries":boundaries,"groups":duplicates}
