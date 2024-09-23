import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path

class CreateDataset(Dataset):
    def __init__(self, lowResImagesPath: str | Path, highResImagesPath: str | Path, 
                 feature_transform: transforms.Compose, target_transform: transforms.Compose):
        """
        Initialize the CreateDataset dataset class.

        Args:
            lowResImagesPath (str | Path): The path to the folder containing the low-resolution .npy files.
            highResImagesPath (str | Path): The path to the folder containing the high-resolution .npy files.
            feature_transform (torchvision.transforms.Compose): The transform to apply to the low-resolution arrays.
            target_transform (torchvision.transforms.Compose): The transform to apply to the high-resolution arrays.
        """
        self.lowResImagesPath = Path(lowResImagesPath)
        self.highResImagesPath = Path(highResImagesPath)
        self.lowResImages = [f for f in os.listdir(lowResImagesPath) if f.endswith('.npy')]
        self.highResImages = [f for f in os.listdir(highResImagesPath) if f.endswith('.npy')]
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
            low_res_img = self.transform(low_res_img)
        if self.target_transform:
            high_res_img = self.target_transform(high_res_img)

        return low_res_img, high_res_img