#!/bin/bash
#
#set the job name (output file and error file)
#SBATCH --job-name=sample_elasticnet_interanimal_pairwise_caitlin_1d_vrdata
#SBATCH --output=outs/sample_elasticnet_interanimal_pairwise_caitlin_1d_vrdata.out
#SBATCH --error=errs/sample_elasticnet_interanimal_pairwise_caitlin_1d_vrdata.error
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
#SBATCH --array=0-271

module load python/3.6.1
srun ${GROUP_HOME}/mec_env/bin/python ${GROUP_HOME}/NormativeMEC/mec/neural_fits/sample_elasticnet_per_neuron_caitlin_1d_vrdata.py --n_jobs 20
