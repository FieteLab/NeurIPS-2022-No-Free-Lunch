#!/bin/bash
#SBATCH -p fiete
#SBATCH -n 1                    # two cores
#SBATCH --mem=12G               # RAM
#SBATCH --gres=gpu:1
#SBATCH --time=99:99:99         # total run time limit (HH:MM:SS)
#SBATCH --mail-type=FAIL

module load openmind/cuda/11.2
module load openmind/cudnn/11.5-v8.3.3.40

id=${1}

# This allegedly helps with memory fragmentation.
TF_GPU_ALLOCATOR=cuda_malloc_async

# Activate virtual environment.
source mec_hpc_venv/bin/activate
export PYTHONPATH=.

# write the executed command to the slurm output file for easy reproduction
# https://stackoverflow.com/questions/5750450/how-can-i-print-each-command-before-executing
set -x

wandb agent rylan/mec-hpc-investigations/${id}
