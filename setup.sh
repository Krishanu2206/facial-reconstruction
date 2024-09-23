#!/bin/bash

set -e

if [ ! -d "data" ]; then
    mkdir data
    echo "data folder created"
    echo "now download the data"
    exit 1
fi

cd data

zip_file="scface.zip"
if [ ! -f "$zip_file" ]; then
    echo "$zip_file not found. Please download the data and try again."
else
    unzip "$zip_file" .
    echo "Data unzipped"
fi

high_quality_images="SCface_database/mugshot_frontal_original_all"
low_quality_images="SCface_database/surveillance_cameras_all"

if [ ! -d "$high_quality_images" ]; then
    echo "High-quality images not found at $high_quality_images"
    exit 2
elif [ ! -d "$low_quality_images" ]; then
    echo "Low-quality images not found at $low_quality_images"
    exit 3
fi

rgb_heatmap_low="rgb&heatMap/lowres"
rgb_heatmap_high="rgb&heatMap/highres"

python process_raw.py --input_dir "$low_quality_images" --output_dir "$rgb_heatmap_low" || { echo "Processing low-quality images failed"; exit 4; }
python process_raw.py --input_dir "$high_quality_images" --output_dir "$rgb_heatmap_high" || { echo "Processing high-quality images failed"; exit 5; }

lowResImagesPath="$rgb_heatmap_low"
highResImagesPath="$rgb_heatmap_high"
train_dir="processed/train"
test_dir="processed/test"

python data_setup.py --lowResImagesPath "$lowResImagesPath" --highResImagesPath "$highResImagesPath" --train_dir "$train_dir" --test_dir "$test_dir" || { echo "Training failed"; exit 6; }
