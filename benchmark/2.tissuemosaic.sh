#!/bin/bash

DATASET=$1

conda activate tissue_mosaic 

# Model training 

training_config_file='./config_dino_ssl_alz.yaml'
python packages/TissueMosaic/run/main_1_train_ssl.py \
	--config $training_config_file \
	--data_folder "_data/$DATASET/sample_ads"

# Model inference: passing all patches through model to obtain low-dimensional representations

python packages/TissueMosaic/run/main_2_featurize.py \
        --anndata_in "_data/$DATASET/sample_ads"  \
        --anndata_out "_data/$DATASET/sample_ads_featurized" \ # add # of epochs trained to featurized if wished 
        --ckpt_in "./ckpt_out.pt" \
        --feature_key dino \
        --ncv_k 10 25 100 \
        --suffix featurized
