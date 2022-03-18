#!/bin/bash
#
#set the job name (output file and error file)
#SBATCH --job-name=Ringo_trainfrac_cv_per_neuron
#SBATCH --output=outs/Ringo_trainfrac_cv_per_neuron.out
#SBATCH --error=errs/Ringo_trainfrac_cv_per_neuron.error
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
#SBATCH --array=0-374

module load python/3.6.1
srun ${GROUP_HOME}/mec_env/bin/python ${GROUP_HOME}/NormativeMEC/mec/neural_fits/trainfrac_cv_per_neuron.py --target_animal "Ringo"
