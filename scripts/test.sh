#!/bin/bash
#SBATCH -t 02:00:00
#SBATCH -c 2
#SBATCH --mem=2G
#SBATCH --gres=gpu:1

module load openmind/cuda/11.2
module load openmind/cudnn/11.5-v8.3.3.40

nvidia-smi    # Check GPU status before the program runs