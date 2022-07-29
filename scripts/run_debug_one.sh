#!/bin/bash
#SBATCH -p fiete
#SBATCH -n 1                    # two cores
#SBATCH --mem=8G               # RAM
#SBATCH --time=99:99:99         # total run time limit (HH:MM:SS)
#SBATCH --mail-type=FAIL

id=${1}

# Activate virtual environment.
source mec_hpc_venv/bin/activate
export PYTHONPATH=.

# write the executed command to the slurm output file for easy reproduction
# https://stackoverflow.com/questions/5750450/how-can-i-print-each-command-before-executing
set -x

python -u mec_hpc_investigations/models/run_trainer_rylan.py
