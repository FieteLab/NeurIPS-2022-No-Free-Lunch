#!/bin/bash
#SBATCH -p fiete
#SBATCH -n 2                    # two cores
#SBATCH --mem=16G               # RAM
#SBATCH --time=99:99:99         # total run time limit (HH:MM:SS)
#SBATCH --mail-type=FAIL

module load openmind/cuda/11.2
module load openmind/cudnn/11.5-v8.3.3.40

id=${1}

# update
source torch_tem_venv/bin/activate
#export PYTHONPATH=.

# write the executed command to the slurm output file for easy reproduction
# https://stackoverflow.com/questions/5750450/how-can-i-print-each-command-before-executing
set -x

python -u run.py
