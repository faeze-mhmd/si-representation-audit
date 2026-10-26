                                                                              

import torch
from torch import nn


class SmallFashionCNN(nn.Module):
                                                                              

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
        )
        self.embedding = nn.Sequential(nn.Linear(256, 32), nn.ReLU())
        self.classifier = nn.Linear(32, 10)

    def forward_parts(self, inputs):
        if inputs.ndim != 4 or inputs.shape[0] < 1 or tuple(inputs.shape[1:]) != (1, 28, 28):
            raise ValueError("expected nonempty N x 1 x 28 x 28 images")
        if inputs.dtype != torch.float32:
            raise ValueError("expected float32 images")
        pooled = self.features(inputs)
        penultimate = self.embedding(pooled)
        return {"input": inputs.flatten(1), "conv_features": pooled,
                "penultimate": penultimate, "logits": self.classifier(penultimate)}

    def forward(self, inputs):
        return self.forward_parts(inputs)["logits"]


def build_fashion_cnn(seed):
                                                                             
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return SmallFashionCNN()
