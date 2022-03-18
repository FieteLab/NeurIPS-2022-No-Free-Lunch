#!/bin/bash
#
#set the job name (output file and error file)
#SBATCH --job-name=fixedinteranimal_model_2d_elasticnet_max_valfrac01_cv_per_neuron
#SBATCH --output=outs/fixedinteranimal_model_2d_elasticnet_max_valfrac01_cv_per_neuron.out
#SBATCH --error=errs/fixedinteranimal_model_2d_elasticnet_max_valfrac01_cv_per_neuron.error
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
#SBATCH --array=0-90

module load python/3.6.1
srun ${GROUP_HOME}/mec_env/bin/python ${GROUP_HOME}/NormativeMEC/mec/neural_fits/model_2d_cv_per_neuron_fixedinteranimal.py --cv_type="elasticnet_max" --train_frac=0.1 --val_frac=0.1
