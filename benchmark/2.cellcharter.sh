#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 0-08:00
#SBATCH -c 1
#SBATCH --mem=50G
#SBATCH --open-mode=append
#SBATCH --job-name=cc_uc  
#SBATCH --exclude=compute-g-16-25[4-5]
#SBATCH -o /n/data1/hms/dbmi/raychaudhuri/lab/lakshay/vimapaper/benchmark/_data/UC/cc_uc.out 
#SBATCH -e /n/data1/hms/dbmi/raychaudhuri/lab/lakshay/vimapaper/benchmark/_data/UC/cc_uc.err

DATASET=$1 
python "./2.cellcharter.py" $DATASET
