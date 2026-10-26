                                                                               

from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def build_mlp():
                                                                             
    return nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))


def benchmark_cpu_training(pool, manifest):
                                                                                
    training = np.array(manifest["train_rows"])
    x = torch.from_numpy(pool.features[training].astype(np.float32) / 16.0)
    y = torch.from_numpy(pool.labels[training].copy())
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        model = build_mlp()
        generator = torch.Generator().manual_seed(11)
        loader = DataLoader(TensorDataset(x, y), batch_size=64, shuffle=True, generator=generator, num_workers=0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0)
        loss_function = nn.CrossEntropyLoss()
        times = []
        for _ in range(2):
            start = perf_counter()
            for inputs, labels in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = loss_function(model(inputs), labels)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite loss in CPU timing check")
                loss.backward()
                optimizer.step()
            times.append(perf_counter() - start)
    projection = float(np.mean(times) * 50 * 6)
    return {
        "purpose": "throwaway runtime check only; not a pilot/generalization result",
        "torch": torch.__version__, "device": "cpu", "torch_threads": torch.get_num_threads(),
        "training_rows": len(training), "epochs_measured": 2, "epoch_seconds": times,
        "projected_six_run_training_seconds": projection,
        "twofold_margin_training_seconds": 2 * projection,
        "training_budget_seconds": 7200,
        "within_budget_with_twofold_margin": 2 * projection <= 7200,
        "projection_limit": "two-epoch estimate; excludes imports, SI/probes, validation, logging, checkpointing and interruptions",
        "no_test_evaluation": True, "weights_saved": False,
    }
