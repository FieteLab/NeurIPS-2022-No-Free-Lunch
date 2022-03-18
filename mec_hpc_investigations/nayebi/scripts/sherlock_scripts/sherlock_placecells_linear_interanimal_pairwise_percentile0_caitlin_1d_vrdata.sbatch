#!/bin/bash
#
#set the job name (output file and error file)
#SBATCH --job-name=sherlock_placecells_linear_interanimal_pairwise_percentile0_caitlin_1d_vrdata
#SBATCH --output=outs/sherlock_placecells_linear_interanimal_pairwise_percentile0_caitlin_1d_vrdata.out
#SBATCH --error=errs/sherlock_placecells_linear_interanimal_pairwise_percentile0_caitlin_1d_vrdata.error
#################
#set time needed
#SBATCH --time=12:00:00
#################
#SBATCH -p yamins,owners,normal
#################
#SBATCH --nodes=1
#################
#request 128GB RAM on CPU
#SBATCH --mem=128000
#################
#SBATCH --array=0-271

module load python/3.6.1
srun ${GROUP_HOME}/mec_env/bin/python ${GROUP_HOME}/NormativeMEC/mec/neural_fits/interanimal_noise_ceiling_caitlin_1d_vrdata.py --map_type percentile --percentile 0 --mode pairwise --first_N_name placecells_none_env1d_activations_1d --n_jobs 8
