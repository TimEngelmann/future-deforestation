#!/bin/bash

#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=40G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:8G
#SBATCH --time=8:00:00

python3 src/models/train_model.py $1
 