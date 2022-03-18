#!/bin/bash
#
#set the job name (output file and error file)
#SBATCH --job-name=sherlock_percentile_cv_per_neuron
#SBATCH --output=outs/sherlock_percentile_cv_per_neuron.out
#SBATCH --error=errs/sherlock_percentile_cv_per_neuron.error
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
#SBATCH --array=0-619

module load python/3.6.1
srun ${GROUP_HOME}/mec_env/bin/python ${GROUP_HOME}/NormativeMEC/mec/neural_fits/percentile_cv_per_neuron.py
