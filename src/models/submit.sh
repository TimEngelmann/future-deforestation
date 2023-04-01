#!/bin/bash

#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --time=24:00:00

python3 train_model.py
 