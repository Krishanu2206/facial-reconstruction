import os
import sys
from pathlib import Path
from torch.utils.data import Dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

cctv_all = Path("data/raw/cctv_footage/surveillance_cameras_all")
frontal_images = Path("data/raw/high_quality_images/mugshot_frontal_original_all")
annotations = Path("data/raw/cctv_footage/annotations/all.txt")


class create_dataset(Dataset):
    def __init__(self, annotations_path: str | Path, images_path) -> None:
        super().__init__()