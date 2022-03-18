#!/bin/bash
#
#set the job name (output file and error file)
#SBATCH --job-name=sherlock_internal_consistencies_caitlin_1d_vrdata
#SBATCH --output=outs/sherlock_internal_consistencies_caitlin_1d_vrdata.out
#SBATCH --error=errs/sherlock_internal_consistencies_caitlin_1d_vrdata.error
#################
#set time needed
#SBATCH --time=01:00:00
#################
#SBATCH -p yamins,owners,normal
#################
#SBATCH --nodes=1
#################
#request 128GB RAM on CPU
#SBATCH --mem=128000
#################
#SBATCH --array=0-17

module load python/3.6.1
srun ${GROUP_HOME}/mec_env/bin/python ${GROUP_HOME}/NormativeMEC/mec/neural_fits/interanimal_noise_ceiling_caitlin_1d_vrdata.py --mode internal --n_jobs 8
