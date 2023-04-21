import torch
import torch.nn as nn

class QualityDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resnext = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=True)
        self.fc = nn.Linear(1000, 8) # output dim of resnext = 1000

    def forward(self, x):
        x = self.resnext(x)
        x = self.fc(x)
        return x
