                                                                                    

import base64
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
from time import perf_counter
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_limits
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .baselines import classification_summary
from .cpu_benchmark import build_mlp
from .metrics import separation_index


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def model_state_hash(model):
                                                                               
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(array.dtype.str.encode())
        digest.update(json.dumps(list(array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def model_bundle(model, metadata):
                                                                                     
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **{k: t.detach().cpu().numpy() for k, t in model.state_dict().items()})
    payload = buffer.getvalue()
    return {"format": "numpy_npz_base64_v1", "architecture": [64, 64, 32, 10],
            "metadata": metadata, "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "model_state_sha256": model_state_hash(model),
            "payload_base64": base64.b64encode(payload).decode("ascii")}


def model_from_bundle(bundle):
    if bundle["format"] != "numpy_npz_base64_v1" or bundle["architecture"] != [64, 64, 32, 10]:
        raise ValueError("unsupported model bundle")
    payload = base64.b64decode(bundle["payload_base64"], validate=True)
    if hashlib.sha256(payload).hexdigest() != bundle["payload_sha256"]:
        raise ValueError("model bundle payload hash mismatch")
                                                                        
    with torch.random.fork_rng(devices=[]):
        model = build_mlp()
    expected = model.state_dict()
    with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
        if set(arrays.files) != set(expected):
            raise ValueError("model bundle tensor names differ")
        state = {}
        for name, tensor in expected.items():
            array = arrays[name]
            if array.shape != tuple(tensor.shape) or array.dtype != np.float32 or not np.isfinite(array).all():
                raise ValueError("invalid tensor in model bundle: " + name)
            state[name] = torch.from_numpy(array.copy())
    model.load_state_dict(state, strict=True)
    if model_state_hash(model) != bundle["model_state_sha256"]:
        raise ValueError("model tensor-content hash mismatch")
    return model.eval()


def forward_parts(model, inputs):
    hidden1 = model[1](model[0](inputs))
    hidden2 = model[3](model[2](hidden1))
    return {"input": inputs, "hidden1": hidden1, "penultimate": hidden2, "logits": model[4](hidden2)}


def make_loader(features, labels, sample_ids, *, seed, batch_size):
    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(features, labels, torch.as_tensor(sample_ids, dtype=torch.int64))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0), generator


def train_epoch(model, loader, optimizer):
    model.train()
    order = hashlib.sha256()
    start = perf_counter()
    for features, labels, sample_ids in loader:
        order.update(sample_ids.numpy().astype("<i8").tobytes())
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(features), labels)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite training loss")
        loss.backward()
        optimizer.step()
    return perf_counter() - start, order.hexdigest()


def evaluate(model, features, labels, *, detailed=False):
    model.eval()
    with torch.no_grad():
        parts = forward_parts(model, features)
        logits = parts["logits"]
        if not torch.isfinite(logits).all():
            raise ValueError("nonfinite evaluation logits")
        predicted = logits.argmax(dim=1).numpy()
        result = {"accuracy": float(np.mean(predicted == labels.numpy())),
                  "cross_entropy": float(nn.functional.cross_entropy(logits, labels))}
        if detailed:
            result.update(classification_summary(labels.numpy(), predicted))
        return result, {k: v.numpy() for k, v in parts.items()}, predicted


def permutation_control(labels, neighbors, *, seed, count):
                                                                                 
    labels = np.asarray(labels)
    generator = np.random.default_rng(seed)
    scores = []
    for _ in range(count):
        shuffled = generator.permutation(labels)
        scores.append(float(np.mean(shuffled == shuffled[neighbors])))
    _, frequencies = np.unique(labels, return_counts=True)
    n = len(labels)
    expected = float(np.sum(frequencies * (frequencies - 1)) / (n * (n - 1)))
    return {"seed": seed, "permutations": count, "mean": float(np.mean(scores)),
            "std": float(np.std(scores, ddof=1)) if count > 1 else 0.0,
            "minimum": min(scores), "maximum": max(scores), "expected_random_label_agreement": expected}


def representation_metrics(parts, labels, sample_ids, subset, split, config, *, input_cache=None):
    records = {}
    for layer in ("input", "hidden1", "penultimate"):
        if layer == "input" and input_cache is not None:
            records[layer] = input_cache
            continue
        result = separation_index(parts[layer][subset], labels[subset], sample_ids=sample_ids[subset])
        records[layer] = {
            "si": result.score, "n": result.n_samples, "features": result.n_features,
            "tie_fraction": result.tie_fraction, "duplicate_vector_fraction": result.duplicate_vector_fraction,
            "label_permutation": permutation_control(
                labels[subset], result.nearest_neighbor_indices,
                seed=config["permutation_seeds"][split], count=config["permutation_count"],
            ),
        }
    return records


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def snapshot(model, data, config, input_cache):
    metrics, full_parts = {}, {}
    for split in ("train", "validation"):
        item = data[split]
        prediction_metrics, parts, _ = evaluate(model, item["x"], item["y"])
        metrics[split] = {"prediction": prediction_metrics,
            "representations": representation_metrics(parts, item["y"].numpy(), item["ids"], item["si_positions"],
                                                       split, config, input_cache=input_cache.get(split))}
        input_cache.setdefault(split, metrics[split]["representations"]["input"])
        full_parts[split] = parts
    return metrics, full_parts


def fit_probes(parts, data, config):
    records = []
    for name, model in (
        ("linear_probe", LogisticRegression(C=config["probe_logistic_C"], max_iter=config["probe_logistic_max_iter"],
                                             solver="lbfgs", random_state=2027)),
        ("cross_split_1nn_probe", KNeighborsClassifier(n_neighbors=1, algorithm="brute", metric="euclidean", n_jobs=1)),
    ):
        start = perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(parts["train"]["penultimate"], data["train"]["y"].numpy())
            prediction = model.predict(parts["validation"]["penultimate"])
        records.append({"model": name, "parameters": model.get_params(deep=False),
                        "validation": classification_summary(data["validation"]["y"].numpy(), prediction),
                        "fit_and_predict_seconds": perf_counter() - start,
                        "warnings": [str(w.message) for w in caught]})
    return records


def run_one(data, config, provenance, directory, *, setting, seed, input_cache):
                                                                                
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    run_id = f"{setting}_seed{seed}"
    fingerprint = canonical_hash({"config": config, "provenance": provenance, "setting": setting, "seed": seed})
    existing = directory / "metrics.json"
    if existing.exists():
        completed = json.loads(existing.read_text(encoding="utf-8"))
        if completed["fingerprint"] != fingerprint:
            raise ValueError("output directory belongs to different source/config: " + run_id)
        if completed.get("completed"):
            bundle = json.loads((directory / "model_bundle.json").read_text(encoding="utf-8"))
            if model_state_hash(model_from_bundle(bundle)) != completed["final_model_state_sha256"]:
                raise ValueError("completed model bundle changed")
            return completed, "reused_completed"

    with torch.random.fork_rng(devices=[]), threadpool_limits(limits=1):
        torch.manual_seed(seed)
        model = build_mlp()
        initial_hash = model_state_hash(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["settings"][setting])
        loader, generator = make_loader(data["train"]["x"], data["train"]["y"], data["train"]["ids"],
                                       seed=seed, batch_size=config["batch_size"])
        record = {"run_id": run_id, "setting": setting, "seed": seed, "weight_decay": config["settings"][setting],
                  "fingerprint": fingerprint, "provenance": provenance, "config": config,
                  "started_at_utc": datetime.now(timezone.utc).isoformat(), "initial_model_state_sha256": initial_hash,
                  "history": [], "snapshots": {}, "epoch_order_sha256": [], "training_seconds": 0.0,
                  "measurement_seconds": 0.0, "completed": False, "test_evaluated": False}
        last_epoch, disposition = 0, "trained"
        checkpoint_path = directory / "resume.pt"
        if checkpoint_path.exists():
            saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if saved["fingerprint"] != fingerprint:
                raise ValueError("checkpoint belongs to different source/config")
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            generator.set_state(saved["generator"])
            torch.set_rng_state(saved["torch_rng"])
            record, last_epoch = saved["record"], saved["epoch"]
            disposition = "resumed"

        def capture(epoch):
            start = perf_counter()
            measurement, _ = snapshot(model, data, config, input_cache)
            record["snapshots"][str(epoch)] = measurement
            record["measurement_seconds"] += perf_counter() - start

        def checkpoint(epoch):
            state = {"fingerprint": fingerprint, "epoch": epoch, "model": model.state_dict(),
                     "optimizer": optimizer.state_dict(), "generator": generator.get_state(),
                     "torch_rng": torch.get_rng_state(), "record": record}
            pending = checkpoint_path.with_suffix(".pending")
            torch.save(state, pending)
            pending.replace(checkpoint_path)
            save_json(existing, record)

        if not record["history"]:
            capture(0)
            record["history"].append({"epoch": 0, **{s: record["snapshots"]["0"][s]["prediction"] for s in data}})
            checkpoint(0)
        for epoch in range(last_epoch + 1, config["epochs"] + 1):
            seconds, order_hash = train_epoch(model, loader, optimizer)
            record["training_seconds"] += seconds
            record["epoch_order_sha256"].append(order_hash)
            if record["training_seconds"] > config["training_budget_seconds"] / 6:
                raise RuntimeError("per-run share of training budget exceeded; amend plan before proceeding")
            start = perf_counter()
            per_epoch = {s: evaluate(model, item["x"], item["y"])[0] for s, item in data.items()}
            record["measurement_seconds"] += perf_counter() - start
            record["history"].append({"epoch": epoch, **per_epoch})
            if epoch in config["checkpoint_epochs"]:
                capture(epoch)
                checkpoint(epoch)

        start = perf_counter()
        final_parts = {}
        record["final_prediction"] = {}
        for split, item in data.items():
            summary, parts, prediction = evaluate(model, item["x"], item["y"], detailed=True)
            record["final_prediction"][split] = summary
            final_parts[split] = parts
            if split == "validation":
                errors = np.flatnonzero(item["y"].numpy() != prediction)
                record["validation_error_count"] = len(errors)
                record["validation_errors_first20"] = [{"sample_id": int(item["ids"][i]),
                    "true": int(item["y"][i]), "predicted": int(prediction[i])} for i in errors[:20]]
        record["probes"] = fit_probes(final_parts, data, config)
        record["measurement_seconds"] += perf_counter() - start
        record["final_model_state_sha256"] = model_state_hash(model)
        bundle = model_bundle(model, {"run_id": run_id, "fingerprint": fingerprint, "seed": seed,
                                     "setting": setting, "weight_decay": config["settings"][setting], "epoch": config["epochs"]})
                                                                         
        restored = model_from_bundle(bundle)
        with torch.no_grad():
            if not torch.equal(model(data["validation"]["x"]), restored(data["validation"]["x"])):
                raise ValueError("portable model round-trip changed logits")
        save_json(directory / "model_bundle.json", bundle)
        record["model_bundle_payload_sha256"] = bundle["payload_sha256"]
        record["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        record["completed"] = True
        save_json(existing, record)
        return record, disposition
