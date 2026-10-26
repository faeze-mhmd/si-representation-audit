                                                                                  

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from si_audit.cnn import build_fashion_cnn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="omit for stdout")
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/fashion_cnn_v1.json").read_text(encoding="utf-8"))
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2040)
        inputs = torch.rand(config["batch_size"],1,28,28)
        labels = torch.arange(config["batch_size"]) % 10
        model = build_fashion_cnn(11)
        optimizer = torch.optim.AdamW(model.parameters(),lr=config["learning_rate"],weight_decay=config["weight_decay"])
        times = []
        for step in range(14):
            start = perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(inputs),labels)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite synthetic loss")
            loss.backward()
            optimizer.step()
            if step >= 2:
                times.append(perf_counter()-start)
        with torch.no_grad():
            parts = model.forward_parts(inputs[:3])
        steps = {k: math.ceil(n/config["batch_size"])*config["epochs"] for k,n in config["budgets"].items()}
        total_steps = sum(steps.values())*len(config["seeds"])
        seconds_per_step = sum(times)/len(times)
        projection = seconds_per_step*total_steps
        files = ["src/si_audit/cnn.py", "scripts/smoke_fashion_cnn.py", "tests/test_cnn.py",
                 "configs/fashion_cnn_v1.json", "FASHION_CNN_PROTOCOL.md"]
        result = {"executed_at_utc": datetime.now(timezone.utc).isoformat(), "stage": "synthetic preflight only",
            "environment": {"python": platform.python_version(), "torch": torch.__version__,
                            "platform": platform.platform(), "device": "cpu", "threads": torch.get_num_threads()},
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "representation_shapes": {k:list(v.shape) for k,v in parts.items()},
            "warmup_steps": 2, "measured_steps": len(times), "batch_size": config["batch_size"],
            "step_seconds": times, "mean_step_seconds": seconds_per_step,
            "planned_steps_per_run": steps, "planned_steps_all_six": total_steps,
            "synthetic_projected_optimizer_seconds": projection,
            "synthetic_projection_with_margin_seconds": projection*config["projection_safety_factor"],
            "training_budget_seconds": config["training_budget_seconds"],
            "synthetic_projection_within_training_cap": projection*config["projection_safety_factor"] <= config["training_budget_seconds"],
            "limitations": "Synthetic repeated-batch optimizer-only estimate, not real-data timing; excludes loading, evaluation, SI, probes, serialization, imports and interruptions. Real-training-only preflight is required.",
            "real_data_opened": False, "test_performance_evaluated": False, "weights_saved": False,
            "source_sha256": {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files}}
    payload = json.dumps(result,indent=2)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(payload,encoding="utf-8")
    else:
        print(payload,end="")


if __name__ == "__main__":
    main()
