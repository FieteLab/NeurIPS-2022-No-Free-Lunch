#!/bin/bash
#SBATCH -p fiete
#SBATCH -n 1                    # one CPU
#SBATCH --mem=8G                # RAM
#SBATCH --time=99:99:99         # total run time limit (HH:MM:SS)
#SBATCH --mail-type=FAIL

id=${1}

# Activate virtual environment.
source mec_hpc_venv/bin/activate
export PYTHONPATH=.

# write the executed command to the slurm output file for easy reproduction
# https://stackoverflow.com/questions/5750450/how-can-i-print-each-command-before-executing
set -x

wandb agent rylan/mec-hpc-investigations/${id}
