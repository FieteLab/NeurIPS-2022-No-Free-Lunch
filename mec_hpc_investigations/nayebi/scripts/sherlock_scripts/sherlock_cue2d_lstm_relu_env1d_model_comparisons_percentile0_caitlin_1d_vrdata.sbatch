#!/bin/bash
#
#set the job name (output file and error file)
#SBATCH --job-name=sherlock_cue2d_lstm_relu_env1d_model_comparisons_percentile0_caitlin_1d_vrdata
#SBATCH --output=outs/sherlock_cue2d_lstm_relu_env1d_model_comparisons_percentile0_caitlin_1d_vrdata.out
#SBATCH --error=errs/sherlock_cue2d_lstm_relu_env1d_model_comparisons_percentile0_caitlin_1d_vrdata.error
#################
#set time needed
#SBATCH --time=36:00:00
#################
#SBATCH -p yamins,owners,normal
#################
#SBATCH --nodes=1
#################
#request 128GB RAM on CPU
#SBATCH --mem=128000
#################
#SBATCH --array=0-107

module load python/3.6.1
srun ${GROUP_HOME}/mec_env/bin/python ${GROUP_HOME}/NormativeMEC/mec/neural_fits/model_comparisons_caitlin_1d_vrdata.py --model_types "rewardlstm,rewardlstm2" --task_modes "fixedcue2d_prob05,fixedcue2d_prob1,randomcue2d" --activations "relu" --map_type percentile --percentile 0 --n_jobs 8
