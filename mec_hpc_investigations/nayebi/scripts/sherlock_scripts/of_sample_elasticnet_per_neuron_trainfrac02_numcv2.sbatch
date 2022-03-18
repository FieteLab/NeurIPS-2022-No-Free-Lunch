#!/bin/bash
#
#set the job name (output file and error file)
#SBATCH --job-name=of_sample_elasticnet_per_neuron_trainfrac02_numcv2
#SBATCH --output=outs/of_sample_elasticnet_per_neuron_trainfrac02_numcv2.out
#SBATCH --error=errs/of_sample_elasticnet_per_neuron_trainfrac02_numcv2.error
#################
#set time needed
#SBATCH --time=01:00:00
#################
#SBATCH -p owners,normal
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
srun ${GROUP_HOME}/mec_env/bin/python ${GROUP_HOME}/NormativeMEC/mec/neural_fits/sample_elasticnet_per_neuron.py --dataset_name="of_only" --train_frac_range="0.2" --num_cv_splits=2
