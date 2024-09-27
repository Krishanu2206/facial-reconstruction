from torch import nn
import torch.nn.functional as F


class RefinementNetwork(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(RefinementNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        return out + residual

class StackedRefinementSR(nn.Module):
    def __init__(self, base_model, refinement_model):
        super(StackedRefinementSR, self).__init__()
        self.base_model = base_model
        self.refinement_model = refinement_model

    def forward(self, x):
        base_output = self.base_model(x)
        refined_output = self.refinement_model(base_output)
        return refined_output