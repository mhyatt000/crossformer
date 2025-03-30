#!/bin/bash

#SBATCH --partition=cpu
#SBATCH --time=3-00:00:00
#SBATCH --account=bela

#SBATCH --nodes=2
#SBATCH --mem=100G

#SBATCH --job-name=bela
#SBATCH --output=%j.out    
#SBATCH --error=%j.out    

# #SBATCH --gres=gpu:4


##
## my prologue
##
echo $GPU_DEVICE_ORDINAL

pwd
echo $CUDA_VISIBLE_DEVICES
echo $SLURM_VISIBLE_GPUS

env | grep SURM_JOB

##
## my script
##
mpirun -n $SLURM_JOB_NUM_NODES "$@"
