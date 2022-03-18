#!/bin/bash
#
#set the job name (output file and error file)
#SBATCH --job-name=sample_elasticnet_per_neuron_trainfrac01_valfrac01
#SBATCH --output=outs/sample_elasticnet_per_neuron_trainfrac01_valfrac01.out
#SBATCH --error=errs/sample_elasticnet_per_neuron_trainfrac01_valfrac01.error
#################
#set time needed
#SBATCH --time=06:00:00
#################
#SBATCH -p yamins,owners,normal
#################
#SBATCH -c 20
#################
#SBATCH --nodes=1
#################
#request 128GB RAM on CPU
#SBATCH --mem=128000
#################
#SBATCH --array=0-619

module load python/3.6.1
srun ${GROUP_HOME}/mec_env/bin/python ${GROUP_HOME}/NormativeMEC/mec/neural_fits/sample_elasticnet_per_neuron.py --train_frac_range="0.1" --val_frac=0.1
