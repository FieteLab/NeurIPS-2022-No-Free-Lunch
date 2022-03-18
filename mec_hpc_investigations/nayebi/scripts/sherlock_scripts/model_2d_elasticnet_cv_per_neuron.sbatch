#!/bin/bash
#
#set the job name (output file and error file)
#SBATCH --job-name=model_2d_elasticnet_cv_per_neuron
#SBATCH --output=outs/model_2d_elasticnet_cv_per_neuron.out
#SBATCH --error=errs/model_2d_elasticnet_cv_per_neuron.error
#################
#set time needed
#SBATCH --time=48:00:00
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
#SBATCH --array=0-47

module load python/3.6.1
srun ${GROUP_HOME}/mec_env/bin/python ${GROUP_HOME}/NormativeMEC/mec/neural_fits/model_2d_cv_per_neuron.py --cv_type "elasticnet" --cv_n_jobs 20