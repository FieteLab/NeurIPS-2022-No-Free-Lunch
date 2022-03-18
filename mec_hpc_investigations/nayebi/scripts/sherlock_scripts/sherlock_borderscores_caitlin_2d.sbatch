#!/bin/bash
#
#set the job name (output file and error file)
#SBATCH --job-name=sherlock_borderscores_caitlin_2d
#SBATCH --output=outs/sherlock_borderscores_caitlin_2d.out
#SBATCH --error=errs/sherlock_borderscores_caitlin_2d.error
#################
#set time needed
#SBATCH --time=02:00:00
#################
#SBATCH -p yamins,owners,normal
#################
#SBATCH --nodes=1
#################
#request 128GB RAM on CPU
#SBATCH --mem=128000
#################
#SBATCH --array=0-619

module load python/3.6.1
srun ${GROUP_HOME}/mec_env/bin/python ${GROUP_HOME}/NormativeMEC/mec/neural_data/borderscore_permtest_caitlin2d.py