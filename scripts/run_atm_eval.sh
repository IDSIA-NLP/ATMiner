#!/bin/bash

#SBATCH --time=2-00:00:00  # 2 days 12 hour of wall time
#SBATCH --nodes=1        # 1 GPU node
#SBATCH --partition=gpu # GPU partition
#SBATCH --ntasks=1       # 1 CPU core to drive GPU
#SBATCH --gres=gpu:1     # Request 1 GPU
#SBATCH --output=./log.out # output log file
#SBATCH --error=./error.err  # error file
#SBATCH -w gnode03

echo "Job started at "`date`
module purge

source ~/miniconda3/etc/profile.d/conda.sh
conda activate atminer

echo $CONDA_DEFAULT_ENV

cd ../src/
python run_atm_eval.py