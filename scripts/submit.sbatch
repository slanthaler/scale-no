#!/bin/bash

#SBATCH --time=00:30
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --job-name="test-0-0"
#SBATCH --output=slurm-%x.%j.out
#SBATCH --error=slurm-%x.%j.out
#SBATCH --mail-user=slanth@caltech.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --qos=debug

# initialize conda environment
source /groups/astuart/slanth/miniconda3/etc/profile.d/conda.sh
conda activate ml

# run code
python ../../symmetry-no/train.py -c ./config.yaml

#
echo done
