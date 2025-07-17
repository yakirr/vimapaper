#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 2-12:00
#SBATCH -c 1
#SBATCH --exclude=compute-g-16-25[4-5]
#SBATCH --mem=50G
#SBATCH --open-mode=append
#SBATCH --job-name=train_tm_700_UC 
#SBATCH -o /n/data1/hms/dbmi/raychaudhuri/lab/lakshay/vimapaper/benchmark/_data/ALZ/train_tm_UC_700_epochs.out

DATASET=$1

source ~/miniforge3/etc/profile.d/mamba.sh 
mamba activate tm

# Creating sample AnnData objects 
python "./2.pre_tissuemosaic.py" $DATASET 

# Model training 
training_config_file="_data/${DATASET}/config_dino_ssl_${DATASET}.yaml"
python _packages/TissueMosaic/run/main_1_train_ssl.py \
	--config $training_config_file \
	--data_folder "_data/$DATASET/sample_ads"

mkdir -p "_data/$DATASET/sample_ads_featurized" 

# Model inference: passing all patches through model to obtain low-dimensional representations
python _packages/TissueMosaic/run/main_2_featurize.py \
        --anndata_in "_data/$DATASET/sample_ads"  \
        --anndata_out "_data/$DATASET/sample_ads_featurized" \ # add # of epochs trained to featurized if wished 
        --ckpt_in "_data/${DATASET}/${DATASET}_ckpt_out.pt" \
        --feature_key dino \
        --ncv_k 10 25 100 \
        --suffix featurized

python "./2.post_tissuemosaic.py" $DATASET 
