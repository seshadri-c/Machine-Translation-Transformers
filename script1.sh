#!/bin/bash
#SBATCH -A seshadri_c
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --output=op_file.txt
#SBATCH --nodelist=gnode55

source ~/home/projects/TEST/env_test/bin/activate

python train.py

