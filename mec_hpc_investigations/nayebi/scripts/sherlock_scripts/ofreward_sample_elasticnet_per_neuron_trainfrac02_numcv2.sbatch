#!/bin/bash
#
#set the job name (output file and error file)
#SBATCH --job-name=ofreward_sample_elasticnet_per_neuron_trainfrac02_numcv2
#SBATCH --output=outs/ofreward_sample_elasticnet_per_neuron_trainfrac02_numcv2.out
#SBATCH --error=errs/ofreward_sample_elasticnet_per_neuron_trainfrac02_numcv2.error
#################
#set time needed
#SBATCH --time=01:00:00
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
#SBATCH --array=0-597

module load python/3.6.1
srun ${GROUP_HOME}/mec_env/bin/python ${GROUP_HOME}/NormativeMEC/mec/neural_fits/sample_elasticnet_per_neuron.py --dataset_name="ofreward_combined" --train_frac_range="0.2" --num_cv_splits=2
