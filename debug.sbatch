#!/bin/bash

#SBATCH --partition=debug
#SBATCH --time=00:15:00
#SBATCH --account=bela

#SBATCH --gres=gpu:4
#SBATCH --mem=100G

#SBATCH --job-name=bela_debug
#SBATCH --output=%j.out    
#SBATCH --error=%j.out

##
## my prologue
##
echo $GPU_DEVICE_ORDINAL

pwd
echo $CUDA_VISIBLE_DEVICES
echo $SLURM_VISIBLE_GPUS

##
## my script
##
exec "$@"
