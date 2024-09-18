import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

cctv_all = Path("data/raw/cctv_footage/surveillance_cameras_all")
frontal_images = Path("data/raw/high_quality_images/mugshot_frontal_original_all")
annotations = Path("data/raw/cctv_footage/annotations/all.txt")

img2annot = dict()
for annotation in np.loadtxt(annotations, dtype=str):
    img2annot[annotation[0]] = list(map(int, annotation[1:]))


# print(img2annot.keys())
image_list = [img + ".jpg" for img in img2annot.keys()]

def check_files():
    for img in image_list:
        if "frontal" in img:
            img_path = frontal_images / img
            assert img_path.is_file(), f"error at {img_path}"
        else:
            img_path = cctv_all / img
            assert img_path.is_file(), f"error at {img_path}"
    print("all the paths exist")
    
    
    
if __name__ == "__main__":
    check_files()