#!/bin/bash
#
#set the job name (output file and error file)
#SBATCH --job-name=shapemetrics_model_2d_proc_euclidean
#SBATCH --output=outs/shapemetrics_model_2d_proc_euclidean.out
#SBATCH --error=errs/shapemetrics_model_2d_proc_euclidean.error
#################
#set time needed
#SBATCH --time=48:00:00
#################
#SBATCH -p hns,normal,yamins,owners
#################
#SBATCH --nodes=1
#################
#request 128GB RAM on CPU
#SBATCH --mem=128000
#################
#SBATCH --array=0-89

module load python/3.6.1
srun ${GROUP_HOME}/mec_env/bin/python ${GROUP_HOME}/NormativeMEC/mec/neural_fits/model_2d_shapemetrics.py --map_type="proc" --train_frac=0.2 --score_method="euclidean"
