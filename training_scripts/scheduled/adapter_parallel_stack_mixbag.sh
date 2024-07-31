#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=2
#SBATCH --time=4-00:00:00
#SBATCH --job-name Certified-Robustness.job

ulimit -s unlimited
ulimit -c unlimited

source activate /scratch/ravihm.scee.iitmandi/pytorch
cd /home/ravihm.scee.iitmandi/Certified-Robustness

accelerate launch train_adapter_parallel_stack.py --noise_sd -1 --name selector-adapter-mixbag-0.1