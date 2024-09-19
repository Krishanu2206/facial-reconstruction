import torch
from torch import nn


class LandMarkModel(nn.Module):
    def __init__(self, in_features: int = 3, out_features: int = 8) -> None:
        super().__init__()
        
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(3, 128, 3, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.clf = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 62 * 62, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 8)
        )
        
    def forward(self, x: torch.Tensor):
        return self.clf(self.conv_block_2(self.conv_block_1(x)))
    
    
    
    
    
if __name__ == "__main__":
    model = LandMarkModel()
    sample = torch.randn((1, 3, 256, 256))
    op = model(sample)
    print(op.shape)
    print(op)