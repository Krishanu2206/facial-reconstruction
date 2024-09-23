#!/bin/bash

set -e

if ! command -v python3 &> /dev/null; then
    echo "Python3 is not installed. Please install Python3 first."
    exit 1
fi

# Create a virtual environment named 'venv' if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment 'venv' created."
else
    echo "Virtual environment 'venv' already exists."
fi

# Activate the virtual environment
source venv/bin/activate
echo "Virtual environment 'venv' activated."

# Check if requirements.txt exists
if [ -f "requirements.txt" ]; then
    touch logs.txt
    pip install -r requirements.txt >> logs.txt
    echo "Dependencies installed from requirements.txt."
else
    echo "requirements.txt not found. Please make sure it exists."
    deactivate
    exit 2
fi

if [ ! -d "data" ]; then
    mkdir data
    echo "data folder created"
fi

cd data

zip_file="scface.zip"
if [ ! -f "$zip_file" ]; then
    echo "$zip_file not found. Downloading..."
    gdown --id 1mxbAgil0-Lbka9FnNTRyrxHKlVB0vgGQ
    echo "Data downloaded"
    unzip "$zip_file" .
    echo "Data unzipped"
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

rm -r $rgb_heatmap_low $rgb_heatmap_high

echo "Setup complete"