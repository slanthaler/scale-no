#!/bin/bash

#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --job-name="symmetry-baseline"
#SBATCH --output=slurm-%x.%j.out
#SBATCH --error=slurm-%x.%j.out
#SBATCH --mail-user=zongyili@caltech.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# initialize conda environment
source /home/zongyili/.bashrc_2022
conda activate cuda12

# run code
python3 /central/groups/astuart/zongyi/symmetry-no/symmetry_no/train_darcy.py -n helmholtz

