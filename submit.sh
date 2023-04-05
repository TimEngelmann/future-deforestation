#!/bin/bash

#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --time=24:00:00

python3 src/models/train_model.py $1
 