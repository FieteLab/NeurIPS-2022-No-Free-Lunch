#!/bin/bash
#
#set the job name (output file and error file)
#SBATCH --job-name=sherlock_interanimal_pairwise_percentile0_caitlin_1d_vrdata_noncue95
#SBATCH --output=outs/sherlock_interanimal_pairwise_percentile0_caitlin_1d_vrdata_noncue95.out
#SBATCH --error=errs/sherlock_interanimal_pairwise_percentile0_caitlin_1d_vrdata_noncue95.error
#################
#set time needed
#SBATCH --time=06:00:00
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
srun ${GROUP_HOME}/mec_env/bin/python ${GROUP_HOME}/NormativeMEC/mec/neural_fits/interanimal_noise_ceiling_caitlin_1d_vrdata.py --map_type percentile --percentile 0 --mode pairwise --sel_mask_name noncue_cells_95 --n_jobs 8
