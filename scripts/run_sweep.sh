#!/bin/bash
#SBATCH -p fiete
#SBATCH -n 1                    # two cores
#SBATCH --mem=1G                # RAM
#SBATCH --time=99:99:99         # total run time limit (HH:MM:SS)
#SBATCH --mail-type=FAIL


# Run this, then pipe sweep ID to each individual run
# source mec_hpc_venv/bin/activate
# wandb sweep configs/sweep_complete.yaml
# wandb sweep configs/sweep_test.yaml



for i in {1..10}
do
  sbatch scripts/run_one.sh afo5qty6
  sleep 5
done