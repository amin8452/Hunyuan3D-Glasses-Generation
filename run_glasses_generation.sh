#!/bin/bash

echo "Hunyuan3D Glasses Generation"
echo "==========================="
echo

if [ -z "$1" ]; then
    echo "Error: Please provide an input image."
    echo "Usage: ./run_glasses_generation.sh your_image.jpg [output_model.glb]"
    exit 1
fi

INPUT_IMAGE="$1"

if [ -z "$2" ]; then
    OUTPUT_MODEL="output.glb"
else
    OUTPUT_MODEL="$2"
fi

echo "Input image: $INPUT_IMAGE"
echo "Output model: $OUTPUT_MODEL"
echo
echo "Running glasses generation..."
echo

python run_glasses_generation.py --input_image "$INPUT_IMAGE" --output_model "$OUTPUT_MODEL"

echo
if [ $? -eq 0 ]; then
    echo "Process completed successfully!"
else
    echo "Process failed with error code $?"
fi
