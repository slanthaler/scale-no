#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --job-name="scale-ns"
#SBATCH --output=slurm-%x.%j.out
#SBATCH --error=slurm-%x.%j.out
#SBATCH --mail-user=zongyili@caltech.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# initialize conda environment
source /home/zongyili/.bashrc_2022
conda activate cuda12

export PYTHONPATH="/central/groups/astuart/zongyi/scale-no:$PYTHONPATH"

# run code
python3 /central/groups/astuart/zongyi/scale-no/scale_no/train_ns_ablations.py --config config/ns_ablations/config_ns_scale_nofreq_ufno.yaml

