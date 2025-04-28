#!/bin/bash

# Simple training script for GraphSAGE model
# Usage: ./train.sh <input_folder> <output_model>

# Check if correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_folder> <output_model>"
    exit 1
fi

INPUT_FOLDER=$1
OUTPUT_MODEL=$2

echo "Starting training process..."
echo "Input folder: $INPUT_FOLDER"
echo "Output model: $OUTPUT_MODEL"

# Run the training script
python src/train1_d1.py --input_folder "$INPUT_FOLDER" --output_model "$OUTPUT_MODEL"

# Check if training was successful
if [ $? -eq 0 ]; then
    echo "Training completed successfully."
    echo "Model saved to: $OUTPUT_MODEL"
else
    echo "Training failed."
    exit 1
fi