                                                                              

import copy
import io
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from si_audit.cnn import build_fashion_cnn


class FashionCNNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def test_primary_representation_is_actual_head_input(self):
        model = build_fashion_cnn(11).eval()
        inputs = torch.rand(5, 1, 28, 28)
        with torch.no_grad():
            parts = model.forward_parts(inputs)
            self.assertTrue(torch.equal(model.classifier(parts["penultimate"]), model(inputs)))
        self.assertEqual(tuple(parts["conv_features"].shape), (5, 256))
        self.assertEqual(tuple(parts["penultimate"].shape), (5, 32))
        self.assertTrue(torch.all(parts["penultimate"] >= 0))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 9802)

    def test_paired_initialization_preserves_caller_rng(self):
        before = torch.get_rng_state().clone()
        first = build_fashion_cnn(11)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        second = build_fashion_cnn(11)
        other = build_fashion_cnn(22)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(first.parameters(),second.parameters())))
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(first.parameters(),other.parameters())))

    def test_invalid_image_shape_and_dtype_rejected(self):
        model = build_fashion_cnn(11)
        for x in (torch.zeros(2,784), torch.zeros(2,3,28,28), torch.zeros(0,1,28,28),
                  torch.zeros(2,1,27,28), torch.zeros(2,1,28,28,dtype=torch.float64)):
            with self.assertRaises(ValueError):
                model(x)

    def test_training_reaches_conv_layers_and_saved_weights_preserve_logits(self):
        model = build_fashion_cnn(11)
        before = copy.deepcopy(model.state_dict())
        inputs = torch.rand(12,1,28,28)
        labels = torch.arange(12) % 10
        optimizer = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(inputs),labels)
        loss.backward()
        self.assertGreater(float(model.features[0].weight.grad.abs().sum()),0)
        optimizer.step()
        self.assertFalse(torch.equal(before["features.0.weight"],model.features[0].weight))
        self.assertFalse(torch.equal(before["classifier.weight"],model.classifier.weight))
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
        buffer = io.BytesIO()
        torch.save(model.state_dict(),buffer)
        buffer.seek(0)
        restored = build_fashion_cnn(99)
        restored.load_state_dict(torch.load(buffer,weights_only=True),strict=True)
        with torch.no_grad():
            self.assertTrue(torch.equal(model(inputs),restored(inputs)))


if __name__ == "__main__":
    unittest.main()
