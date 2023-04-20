#!/bin/bash

#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=1G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:8g
#SBATCH --time=24:00:00

python3 src/models/train_model.py $1
 