import torch
import torch.nn as nn
# import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # 1 input channel (gray), 6 features to find, 5x5 kernel
        self.features = nn.Sequential(
            nn.Conv2d(1,6,5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),    # kernel 2x2; in 28x28, out 14x14
            
            nn.Conv2d(6,16,5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),    # in 14x14, out 7x7
        )
        self.classifier = nn.Sequential(
            nn.Linear(16*7*7, 120), # 7x7 as (28x28)/2/2
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,10)
    )
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x    # logits