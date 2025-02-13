#!/bin/bash

# Define the image and bind paths
IMAGE="./hyak_container.sif"  # Assuming you have converted your container to a SIF file
SRC_DIR="$PWD/src"
WORK_DIR="$PWD/work"
DATA_DIR="$PWD/src/data/en/"
OUTPUT_DIR="$PWD/output"
SCRIPT_PATH="/job/src/predict.sh"
INPUT_FILE="/job/data/input.txt"
OUTPUT_FILE="/job/output/pred.txt"

# Check if "--nv" is passed as an argument
USE_GPU=0
for arg in "$@"; do
    if [[ "$arg" == "--nv" ]]; then
        USE_GPU=1
    fi
done

# Construct the command
CMD="apptainer run"
if [[ $USE_GPU -eq 1 ]]; then
    CMD+=" --nv"
fi

CMD+=" --bind \"$SRC_DIR:/job/src\" \
       --bind \"$WORK_DIR:/job/work\" \
       --bind \"$DATA_DIR:/job/data\" \
       --bind \"$OUTPUT_DIR:/job/output\" \
       \"$IMAGE\" bash \"$SCRIPT_PATH\" \"$INPUT_FILE\" \"$OUTPUT_FILE\""

# Execute the command
eval $CMD
