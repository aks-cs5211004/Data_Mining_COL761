#!/bin/bash

# Simple testing script for GraphSAGE model
# Usage: ./test.sh <test_folder> <model_path> <output_prediction>

# Check if correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <test_folder> <model_path> <output_prediction>"
    exit 1
fi

TEST_FOLDER=$1
MODEL_PATH=$2
OUTPUT_PREDICTION=$3

echo "Starting testing process..."
echo "Test folder: $TEST_FOLDER"
echo "Model path: $MODEL_PATH"
echo "Output prediction: $OUTPUT_PREDICTION"

# Run the testing script
python src/test1_d2.py --input_dir "$TEST_FOLDER" --model_path "$MODEL_PATH" --output_csv "$OUTPUT_PREDICTION"

# Check if testing was successful
if [ $? -eq 0 ]; then
    echo "Testing completed successfully."
    echo "Predictions saved to: $OUTPUT_PREDICTION"
else
    echo "Testing failed."
    exit 1
fi