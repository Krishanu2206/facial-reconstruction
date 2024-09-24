import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path

class CreateDataset(Dataset):
    def __init__(self, lowResImagesPath: str | Path, highResImagesPath: str | Path, 
                 feature_transform: transforms.Compose, target_transform: transforms.Compose):
        self.lowResImagesPath = Path(lowResImagesPath)
        self.highResImagesPath = Path(highResImagesPath)
        
        print(f"Low-res path: {self.lowResImagesPath}")
        print(f"High-res path: {self.highResImagesPath}")
        
        self.lowResImages = [f for f in os.listdir(lowResImagesPath) if f.endswith('.npy')]
        self.highResImages = [f for f in os.listdir(highResImagesPath) if f.endswith('.npy')]
        
        print(f"Number of low-res images found: {len(self.lowResImages)}")
        print(f"Number of high-res images found: {len(self.highResImages)}")
        
        if len(self.lowResImages) == 0:
            print("Warning: No low-res .npy files found!")
        if len(self.highResImages) == 0:
            print("Warning: No high-res .npy files found!")
        
        self.transform = feature_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.lowResImages)

    def __getitem__(self, idx):
        low_image_name = self.lowResImages[idx]
        low_res_image_path = self.lowResImagesPath / low_image_name
        
        subject = low_image_name.split("_")[0]
        high_image_name = f"{subject}_frontal_rgb_heatmaps.npy"
        high_image_path = self.highResImagesPath / high_image_name

        low_res_img = np.load(low_res_image_path)
        high_res_img = np.load(high_image_path)
        
        if self.transform:
            low_res_img = self.transform.process(low_res_img)
        if self.target_transform:
            high_res_img = self.target_transform.process(high_res_img)
            
        return low_res_img, high_res_img
    
    
if __name__ == "__main__":
    from preprocess import ProcessFeatures, ProcessTarget
    train_dataset = CreateDataset(
        lowResImagesPath="data/processed/train/low_res",
        highResImagesPath="data/processed/train/high_res",
        feature_transform=ProcessFeatures,
        target_transform=ProcessTarget
    )
    
    test_dataset = CreateDataset(lowResImagesPath="data/processed/test/low_res",
                    highResImagesPath="data/processed/test/high_res",
                    feature_transform=ProcessFeatures,
                    target_transform=ProcessTarget)

    
    for low, high in train_dataset:
        print(low.shape, high.shape)
        
    for low, high in test_dataset:
        print(low.shape, high.shape)