import random
import os
import sys
import shutil
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

lowResImagesPath = Path("data/raw/cctv_footage/surveillance_cameras_all")
lowResImages = list(lowResImagesPath.glob("*.jpg"))  # 2860

highResImagesPath = Path("data/raw/high_quality_images/mugshot_frontal_original_all")
highResImages = list(highResImagesPath.glob("*.jpg"))  # 130

train_split = 0.8
train_dir = Path("data/processed/train")
test_dir = Path("data/processed/test")
train_dir.mkdir(exist_ok=True, parents=True)
test_dir.mkdir(exist_ok=True, parents=True)

train_high_res = random.sample(highResImages, int(train_split * len(highResImages)))
test_high_res = [img for img in highResImages if img not in train_high_res]

train_low_res = [img for img in lowResImages if highResImagesPath / (img.stem.split("_")[0] + "_frontal.jpg") in train_high_res]
test_low_res = [img for img in lowResImages if img not in train_low_res]

print(len(train_high_res), len(test_high_res))
print(len(train_low_res), len(test_low_res))
print(train_high_res[:4])
print(train_low_res[:3])


def copy_image(src, dest):
    if not dest.exists():
        dest.mkdir(exist_ok=True, parents=True)
        print(f"Destination folder created: {dest}")

    for idx, img in enumerate(src):
        dest_img_path = dest / (img.stem.split('_')[0] + "_" + "_".join(img.stem.split('_')[1:]) + ".jpg")
        shutil.copy(str(img), str(dest_img_path))

        if idx % 100 == 0:
            print(f"Completed copying image {idx + 1}/{len(src)}")
            print(f"From {img} to {dest_img_path}")


if __name__ == "__main__":

    copy_image(train_high_res, train_dir / "high_res")
    copy_image(train_low_res, train_dir / "low_res")

    copy_image(test_high_res, test_dir / "high_res")
    copy_image(test_low_res, test_dir / "low_res")
