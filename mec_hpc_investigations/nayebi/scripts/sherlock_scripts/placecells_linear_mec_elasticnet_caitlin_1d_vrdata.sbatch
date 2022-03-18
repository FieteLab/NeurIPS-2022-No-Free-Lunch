#!/bin/bash
#
#set the job name (output file and error file)
#SBATCH --job-name=placecells_linear_mec_elasticnet_caitlin_1d_vrdata
#SBATCH --output=outs/placecells_linear_mec_elasticnet_caitlin_1d_vrdata.out
#SBATCH --error=errs/placecells_linear_mec_elasticnet_caitlin_1d_vrdata.error
#################
#set time needed
#SBATCH --time=12:00:00
#################
#SBATCH -p owners,normal
#################
#SBATCH -c 5
#################
#SBATCH --nodes=1
#################
#request 128GB RAM on CPU
#SBATCH --mem=128000
#################
#SBATCH --array=0-271

module load python/3.6.1
srun ${GROUP_HOME}/mec_env/bin/python ${GROUP_HOME}/NormativeMEC/mec/neural_fits/interanimal_noise_ceiling_elasticnet_caitlin_1d_vrdata.py --first_N_name "placecells_none_env1d_activations_1d"
