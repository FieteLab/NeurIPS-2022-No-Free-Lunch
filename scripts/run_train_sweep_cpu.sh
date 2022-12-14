#!/bin/bash
#SBATCH -p fiete
#SBATCH -n 1                    # two cores
#SBATCH --mem=1G                # RAM
#SBATCH --time=01:99:99         # total run time limit (HH:MM:SS)
#SBATCH --mail-type=FAIL


# Run this, then pipe sweep ID to each individual run
# source mec_hpc_venv/bin/activate
# wandb sweep sweeps/sweep_position.yaml

for i in {1..6}
do
  sbatch scripts/run_train_one_cpu.sh 2gz0ks4o
  sleep 3
done