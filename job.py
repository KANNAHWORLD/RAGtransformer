#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128GB
#SBATCH --time=12:15:00
#SBATCH --account=robinjia_1265
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
# SBATCH --array=1-4

module purge
module load gcc/11.3.0
module load python

eval "$(conda shell.bash hook)"
deactivate
source csci467/bin/activate
./script.zsh

# python3 .py --input=input/input_${SLURM_ARRAY_TASK_ID}
