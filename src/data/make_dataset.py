import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path


class CreateDataset(Dataset):
    def __init__(self, lowResImagesPath: str | Path, 
                 highResImagesPath: str | Path, 
                 feature_transform: transforms.Compose, 
                 target_transform: transforms.Compose):
        
        self.lowResImagesPath = lowResImagesPath
        self.highResImagesPath = highResImagesPath
        self.lowResImages = os.listdir(lowResImagesPath)
        self.highResImages = os.listdir(highResImagesPath)

        self.transform = feature_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.lowResImages)

    def __getitem__(self, idx):
        _low_image_name = self.lowResImages[idx]
        _low_res_image_path = os.path.join(self.lowResImagesPath, _low_image_name)
        _subject = _low_image_name.split("_")[0]
        _high_image_name = _subject + "_frontal.jpg"
        _high_image_path = os.path.join(self.highResImagesPath, _high_image_name)

        _low_res_img = Image.open(_low_res_image_path)
        _high_res_img = Image.open(_high_image_path)
        if self.transform:
            _low_res_img = self.transform.process(_low_res_img)
        if self.target_transform:
            _high_res_img = self.target_transform.process(_high_res_img)

        return _low_res_img, _high_res_img