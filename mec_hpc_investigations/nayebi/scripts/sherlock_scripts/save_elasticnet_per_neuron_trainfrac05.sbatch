#!/bin/bash
#
#set the job name (output file and error file)
#SBATCH --job-name=save_elasticnet_per_neuron_trainfrac05
#SBATCH --output=outs/save_elasticnet_per_neuron_trainfrac05.out
#SBATCH --error=errs/save_elasticnet_per_neuron_trainfrac05.error
#################
#set time needed
#SBATCH --time=02:00:00
#################
#SBATCH -p yamins,owners,normal
#################
#SBATCH --nodes=1
#################
#request 256GB RAM on CPU
#SBATCH --mem=256000
#################
#SBATCH --array=0-619

module load python/3.6.1
srun ${GROUP_HOME}/mec_env/bin/python ${GROUP_HOME}/NormativeMEC/mec/neural_fits/save_elasticnet_per_neuron_trainfrac.py --train_frac=0.5
