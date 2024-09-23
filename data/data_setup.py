import random
import os
import shutil
from pathlib import Path
import argparse

def setup_data(lowResImagesPath, highResImagesPath, train_dir, test_dir):
    lowResImages = list(lowResImagesPath.glob("*.npy"))
    highResImages = list(highResImagesPath.glob("*.npy"))

    train_split = 0.8
    train_dir.mkdir(exist_ok=True, parents=True)
    test_dir.mkdir(exist_ok=True, parents=True)

    # Splitting high-res images
    train_high_res = random.sample(highResImages, int(train_split * len(highResImages)))
    test_high_res = [img for img in highResImages if img not in train_high_res]

    # Matching low-res images to high-res splits
    train_low_res = [img for img in lowResImages if highResImagesPath / (img.stem.split("_")[0] + "_frontal_rgb_heatmaps" + img.suffix) in train_high_res]
    test_low_res = [img for img in lowResImages if img not in train_low_res]

    print(f"Train high-res: {len(train_high_res)}, Test high-res: {len(test_high_res)}")
    print(f"Train low-res: {len(train_low_res)}, Test low-res: {len(test_low_res)}")
    print(f"Sample train high-res: {train_high_res[:4]}")
    print(f"Sample train low-res: {train_low_res[:3]}")

    copy_files(train_high_res, train_dir / "high_res")
    copy_files(train_low_res, train_dir / "low_res")
    copy_files(test_high_res, test_dir / "high_res")
    copy_files(test_low_res, test_dir / "low_res")

def copy_files(src, dest):
    if not dest.exists():
        dest.mkdir(exist_ok=True, parents=True)
        print(f"Destination folder created: {dest}")

    for idx, file in enumerate(src):
        dest_file_path = dest / file.name
        shutil.copy(str(file), str(dest_file_path))

        if idx % 100 == 0:
            print(f"Completed copying file {idx + 1}/{len(src)}")
            print(f"From {file} to {dest_file_path}")

if __name__ == "__main__":
    # lowResImagesPath = Path("data/rgb&heatMap/lowres")
    # highResImagesPath = Path("data/rgb&heatMap/highres")
    # train_dir = Path("data/processed/train")
    # test_dir = Path("data/processed/test")

    # use argparse to take these 4 arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--lowResImagesPath", type=str, required=True)
    parser.add_argument("--highResImagesPath", type=str, required=True)
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)
    args = parser.parse_args()

    lowResImagesPath = Path(args.lowResImagesPath)
    highResImagesPath = Path(args.highResImagesPath)
    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)
    

    setup_data(lowResImagesPath, highResImagesPath, train_dir, test_dir)
