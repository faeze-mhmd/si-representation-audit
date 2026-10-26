                                                                                     

import copy
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from si_audit.cpu_benchmark import build_mlp
from si_audit.training import forward_parts, make_loader, model_bundle, model_from_bundle, model_state_hash, permutation_control, train_epoch


class TrainingReproducibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def test_extracted_logits_match_model_and_hidden_shapes(self):
        torch.manual_seed(7)
        model = build_mlp().eval()
        inputs = torch.rand(5, 64)
        with torch.no_grad():
            parts = forward_parts(model, inputs)
            self.assertTrue(torch.equal(parts["logits"], model(inputs)))
        self.assertEqual(tuple(parts["hidden1"].shape), (5, 64))
        self.assertEqual(tuple(parts["penultimate"].shape), (5, 32))
        self.assertTrue(torch.all(parts["hidden1"] >= 0))
        self.assertTrue(torch.all(parts["penultimate"] >= 0))

    def test_portable_weights_preserve_logits_and_rng(self):
        torch.manual_seed(11)
        model = build_mlp().eval()
        with torch.no_grad():
            for parameter in model.parameters(): parameter.add_(0.03125)
        inputs = torch.rand(9, 64)
        state_before = torch.get_rng_state().clone()
        restored = model_from_bundle(model_bundle(model, {"run_id": "synthetic"}))
        self.assertTrue(torch.equal(state_before, torch.get_rng_state()))
        with torch.no_grad(): self.assertTrue(torch.equal(model(inputs), restored(inputs)))
        self.assertEqual(model_state_hash(model), model_state_hash(restored))

    def test_corrupt_bundle_hash_is_rejected(self):
        bundle = model_bundle(build_mlp(), {})
        bundle["payload_sha256"] = "wrong"
        with self.assertRaises(ValueError): model_from_bundle(bundle)

    def test_optimizer_and_generator_resume_reproduce_next_epoch(self):
        torch.manual_seed(22)
        initial = build_mlp()
        features = torch.rand(23, 64)
        labels = torch.arange(23) % 10
        ids = np.arange(23)
        continuous = copy.deepcopy(initial)
        loader, generator = make_loader(features, labels, ids, seed=22, batch_size=7)
        optimizer = torch.optim.AdamW(continuous.parameters(), lr=0.001, weight_decay=0.01)
        train_epoch(continuous, loader, optimizer)
        state = copy.deepcopy(continuous.state_dict())
        optimizer_state = copy.deepcopy(optimizer.state_dict())
        loader_state = generator.get_state().clone()
        _, expected_order = train_epoch(continuous, loader, optimizer)
        restored = copy.deepcopy(initial)
        restored.load_state_dict(state)
        restored_loader, restored_generator = make_loader(features, labels, ids, seed=999, batch_size=7)
        restored_generator.set_state(loader_state)
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.001, weight_decay=0.01)
        restored_optimizer.load_state_dict(optimizer_state)
        _, actual_order = train_epoch(restored, restored_loader, restored_optimizer)
        self.assertEqual(expected_order, actual_order)
        self.assertEqual(model_state_hash(continuous), model_state_hash(restored))

    def test_same_seed_pairs_initialization_and_batch_order(self):
        states, orders = [], []
        for decay in (0.0, 0.01):
            torch.manual_seed(33)
            model = build_mlp()
            states.append(model_state_hash(model))
            loader, _ = make_loader(torch.ones(13, 64), torch.arange(13) % 10, np.arange(13), seed=33, batch_size=5)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=decay)
            _, order = train_epoch(model, loader, optimizer)
            orders.append(order)
        self.assertEqual(states[0], states[1])
        self.assertEqual(orders[0], orders[1])

    def test_permutation_control_handles_exact_small_cases(self):
        for labels, expected in (([0, 1], 0.0), ([1, 1], 1.0)):
            result = permutation_control(np.array(labels), np.array([1, 0]), seed=2030, count=20)
            self.assertEqual(result["mean"], expected)
            self.assertEqual(result["expected_random_label_agreement"], expected)


if __name__ == "__main__":
    unittest.main()
