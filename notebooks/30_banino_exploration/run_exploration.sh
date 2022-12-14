#!/bin/bash
#SBATCH -p fiete
#SBATCH -n 1                    # one core
#SBATCH --mem=4G               # RAM
#SBATCH --time=99:99:99         # total run time limit (HH:MM:SS)
#SBATCH --mail-type=FAIL

# Activate virtual environment.
source mec_hpc_venv/bin/activate

# Make source code importable.
export PYTHONPATH=.

# write the executed command to the slurm output file for easy reproduction
# https://stackoverflow.com/questions/5750450/how-can-i-print-each-command-before-executing
set -x

python -u notebooks/30_banino_exploration/30_banino_exploration.py
