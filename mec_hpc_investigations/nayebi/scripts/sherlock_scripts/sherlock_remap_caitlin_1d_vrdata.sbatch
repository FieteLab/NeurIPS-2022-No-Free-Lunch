#!/bin/bash
#
#set the job name (output file and error file)
#SBATCH --job-name=sherlock_remap_caitlin_1d_vrdata
#SBATCH --output=outs/sherlock_remap_caitlin_1d_vrdata.out
#SBATCH --error=errs/sherlock_remap_caitlin_1d_vrdata.error
#################
#set time needed
#SBATCH --time=2:00:00
#################
#SBATCH -p yamins,owners,normal
#################
#SBATCH --nodes=1
#################
#request 128GB RAM on CPU
#SBATCH --mem=128000
#################
#SBATCH --array=0-109

module load python/3.6.1
srun ${GROUP_HOME}/mec_env/bin/python ${GROUP_HOME}/NormativeMEC/mec/neural_data/remap_caitlin_1d_vrdata.py
