#!/bin/bash

#SBATCH --partition=long
#SBATCH --time=3-00:00:00
#SBATCH --account=bela

#SBATCH --gres=gpu:4
#SBATCH --mem=100G

#SBATCH --job-name=bela
#SBATCH --output=%j.out    
#SBATCH --error=%j.out

##
## my prologue
##

pwd
echo $GPU_DEVICE_ORDINAL
echo $CUDA_VISIBLE_DEVICES
echo $SLURM_VISIBLE_GPUS

##
## my script
##
exec "$@"
