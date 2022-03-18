#!/bin/bash
#
#set the job name (output file and error file)
#SBATCH --job-name=sherlock_specific_interanimal_2d_trainfrac
#SBATCH --output=outs/sherlock_specific_interanimal_2d_trainfrac.out
#SBATCH --error=errs/sherlock_specific_interanimal_2d_trainfrac.error
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
#SBATCH --array=0-8

module load python/3.6.1
srun ${GROUP_HOME}/mec_env/bin/python ${GROUP_HOME}/NormativeMEC/mec/neural_fits/interanimal_2d_trainfrac.py --map_types "ridge,pls100,1-1" --train_fracs="0.1,0.25,0.4"
