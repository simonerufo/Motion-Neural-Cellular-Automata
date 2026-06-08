#!/bin/bash

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
    echo "Usage: ./run.sh [ACTION] [EXPERIMENT] [TARGET_IMAGE] [--save_gif]"
    echo ""
    echo "Available Actions: train1, train2, train_all, inference1, inference2"
    echo "Available Experiments: chemotaxis, chemotaxis_obs, ecosystem"
    echo ""
    echo "Example for Training (Chemotaxis):"
    echo "  ./run.sh train_all chemotaxis targets/salamander32.png"
    echo ""
    echo "Examples for Inference:"
    echo "  1. Base Chemotaxis (No obstacles):"
    echo "     ./run.sh inference2 chemotaxis targets/salamander32.png"
    echo ""
    echo "  2. Chemotaxis with Obstacles:"
    echo "     ./run.sh inference2 chemotaxis_obs targets/salamander32.png"
    echo ""
    echo "  3. Ecosystem (Random Exploration & Mitosis):"
    echo "     ./run.sh inference2 ecosystem targets/jelly32_1.png"
    echo ""
    echo "To save as GIF instead of showing on screen, add --save_gif at the end:"
    echo "     ./run.sh inference2 ecosystem targets/jelly32_1.png --save_gif"
    exit 1
fi

ACTION=$1
EXPERIMENT=$2
TARGET=$3
SAVE_GIF=$4

echo "Starting Script with Action=$ACTION | Experiment=$EXPERIMENT | Image=$TARGET ${SAVE_GIF:+(Saving GIF)}"
python main.py --action "$ACTION" --experiment "$EXPERIMENT" --target "$TARGET" $SAVE_GIF
