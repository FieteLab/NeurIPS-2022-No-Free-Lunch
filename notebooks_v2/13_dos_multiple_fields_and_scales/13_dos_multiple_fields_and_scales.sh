#!/bin/bash
#SBATCH -p fiete
#SBATCH -n 1                    # one core
#SBATCH --mem=10G               # RAM
#SBATCH --time=24:99:99         # total run time limit (HH:MM:SS)
#SBATCH --mail-type=FAIL

# Activate virtual environment.
source mec_hpc_venv/bin/activate

# Make source code importable.
export PYTHONPATH=.

# write the executed command to the slurm output file for easy reproduction
# https://stackoverflow.com/questions/5750450/how-can-i-print-each-command-before-executing
set -x

python -u notebooks_v2/13_dos_multiple_fields_and_scales/13_dos_multiple_fields_and_scales.py
