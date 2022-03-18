#!/bin/bash
#
#set the job name (output file and error file)
#SBATCH --job-name=sherlock_model_2d_trainfrac
#SBATCH --output=outs/sherlock_model_2d_trainfrac.out
#SBATCH --error=errs/sherlock_model_2d_trainfrac.error
#################
#set time needed
#SBATCH --time=48:00:00
#################
#SBATCH -p yamins,owners,normal
#################
#SBATCH --nodes=1
#################
#request 128GB RAM on CPU
#SBATCH --mem=128000
#################
#SBATCH --array=0-563

module load python/3.6.1
srun ${GROUP_HOME}/mec_env/bin/python ${GROUP_HOME}/NormativeMEC/mec/neural_fits/model_2d_trainfrac.py
