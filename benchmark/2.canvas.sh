#!/bin/bash

DATASET=$1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate canvas

export PYTHONPATH="${PYTHONPATH}:$(pwd)/_packages/CANVAS"

# preprocessing
echo "PREPROCESSING"
python _packages/CANVAS/canvas/run_preprocess.py \
    --config_root _data/$DATASET/canvas/configs \
    --data_root _data/$DATASET/canvas/data

# training
echo "TRAINING"
python _packages/CANVAS/canvas/run_training.py \
    --epoch 200 \
    --config_root _data/$DATASET/canvas/configs \
    --data_root _data/$DATASET/canvas/data

echo "INFERENCE"
python _packages/CANVAS/canvas/run_inference.py --ckpt_num 199 \
    --config_root _data/$DATASET/canvas/configs \
    --data_root _data/$DATASET/canvas/data

echo "ANALYSIS"
python _packages/CANVAS/canvas/run_analysis.py \
    --config_root _data/$DATASET/canvas/configs \
    --data_root _data/$DATASET/canvas/data