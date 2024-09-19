import os
import sys
import numpy as np
from PIL import Image
from typing import Dict
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.transforms import Compose

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from preprocess import ProcessFeatures, ProcessTarget



class create_dataset(Dataset):
    def __init__(self, annotations_path: str | Path = Path("data/raw/cctv_footage/annotations/all.txt"),
                 images_path: str | Path = Path("data/processed"),
                 faeture_transforms: Compose = ProcessFeatures,
                 target_transfomrs: Compose = ProcessTarget) -> None:
        """
        Initialize the create_dataset dataset class.

        Args:
            annotations_path (str | Path, optional): The path to the file containing the annotations.
                Defaults to Path("data/raw/cctv_footage/annotations/all.txt").
            images_path (str | Path, optional): The path to the folder containing the images.
                Defaults to Path("data/processed").
            feature_transforms (Compose, optional): The transform to apply to the low-resolution images.
                Defaults to ProcessFeatures.
            target_transforms (Compose, optional): The transform to apply to the high-resolution images.
                Defaults to ProcessTarget.
        """
        super().__init__()
        self.annotations_path = annotations_path
        self.images_path = images_path
        self.faeture_transform = faeture_transforms
        self.target_transform = target_transfomrs
        
        self.annotations = self._load_txt(self.annotations_path)
        self.images = list(self.images_path.glob("*/*/*.jpg"))
        
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        img_pil = Image.open(img_path)
        
        img = np.array(img_pil)
        annot = self.annotations[img_path.stem]
        
        if self.target_transform:
            annot = self.target_transform.process(landmarks=annot, img_size=img.shape, new_size=(256, 256, 3))
        
        if self.faeture_transform:
            img = self.faeture_transform.process(img_pil)
        
        return img, annot
    
    def _load_txt(self, annotations_path) -> Dict:
        img2annot = dict()
        for annotation in np.loadtxt(annotations_path, dtype=str):
            img2annot[annotation[0]] = list(map(int, annotation[1:]))

        return img2annot
    
    
    
if __name__ == "__main__":
    dataset = create_dataset()
    
    print(dataset[0])