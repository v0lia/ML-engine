import torch
import torch.nn as nn
# import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # 1 input channel (gray)
        self.classifier = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10)
        )
    def forward(self, x):
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x    # logits