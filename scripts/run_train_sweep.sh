#!/bin/bash
#SBATCH -p fiete
#SBATCH -n 1                    # two cores
#SBATCH --mem=1G                # RAM
#SBATCH --time=01:99:99         # total run time limit (HH:MM:SS)
#SBATCH --mail-type=FAIL


# Run this, then pipe sweep ID to each individual run
# source mec_hpc_venv/bin/activate
# wandb sweep sweeps/sweep_position.yaml

for i in {1..10}
do
  sbatch scripts/run_train_one.sh khwhl11c
  sleep 1
done