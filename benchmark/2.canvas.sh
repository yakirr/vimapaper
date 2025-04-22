#!/bin/bash

DATASET=$1

conda activate canvas

# preprocessing
python _packages/CANVAS/canvas/run_preprocess.py \
    --config_root _data/$DATASET/canvas/configs \
    --data_root _data/$DATASET/canvas/data

# training
python packages/CANVAS/canvas/run_training.py \
    --epoch 200 \
    --config_root _data/$DATASET/canvas/configs \
    --data_root _data/$DATASET/canvas/data

python packages/CANVAS/canvas/run_inference.py --ckpt_num 199 \
    --config_root _data/$DATASET/canvas/configs \
    --data_root _data/$DATASET/canvas/data

python packages/CANVAS/canvas/run_analysis.py \
    --config_root _data/$DATASET/canvas/configs \
    --data_root _data/$DATASET/canvas/data