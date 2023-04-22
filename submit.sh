#!/bin/bash

#SBATCH --mem-per-cpu=64G
#SBATCH --tmp=300G  
#SBATCH --gpus=1
#SBATCH --gres=gpumem:8G
#SBATCH --time=4:00:00

python3 src/models/train_model.py $1
 