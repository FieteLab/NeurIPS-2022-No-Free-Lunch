#!/bin/bash
#
#set the job name (output file and error file)
#SBATCH --job-name=missing_models_elasticnet_caitlin_1d_vrdata
#SBATCH --output=outs/missing_models_elasticnet_caitlin_1d_vrdata.out
#SBATCH --error=errs/missing_models_elasticnet_caitlin_1d_vrdata.error
#################
#set time needed
#SBATCH --time=24:00:00
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
#SBATCH --array=0-11

module load python/3.6.1
srun ${GROUP_HOME}/mec_env/bin/python ${GROUP_HOME}/NormativeMEC/mec/neural_fits/model_comparisons_elasticnet_caitlin_1d_vrdata.py --missing_fn "missing_sherlock_jobs_1d.npz"
