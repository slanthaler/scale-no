#!/bin/bash

#SBATCH --time=10:55:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -C gpu&hbm80g
#SBATCH -c 2
#SBATCH --gpus-per-task=1
#SBATCH --mem=224G
#SBATCH -A m4505_g
#SBATCH -q regular
#SBATCH -J "helmholtz-train-uno"
#SBATCH --output=slurm-%x.%j.out
#SBATCH --error=slurm-%x.%j.out
#SBATCH --mail-user=mhchen@caltech.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

cd /pscratch/sd/z/zongyili/mhchen/symmetry-no-ablations-2D-NSandHelm/

CONDA_PATH=/pscratch/sd/z/zongyili/anaconda3
GINO_ENV_PATH=$CONDA_PATH/envs/gino
export PATH=$CONDA_PATH/bin:$PATH
export PYTHONPATH=$GINO_ENV_PATH/lib/python3.7/site-packages:$PYTHONPATH
export PYTHONPATH=/pscratch/sd/z/zongyili/mhchen/neuraloperator:$PYTHONPATH
export PYTHONPATH=/pscratch/sd/z/zongyili/mhchen/symmetry-no-ablations:$PYTHONPATH
export PYTHONPATH=/pscratch/sd/z/zongyili/mhchen/UNO:$PYTHONPATH
export PYTHONPATH=/pscratch/sd/z/zongyili/mhchen:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

$GINO_ENV_PATH/bin/python symmetry_no/train_helm.py --config config/helmholtz/config_helmholtz_uno.yaml
