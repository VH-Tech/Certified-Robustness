#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --time=4-00:00:00
#SBATCH --job-name GPU1-Certified-Robustness.job

ulimit -s unlimited
ulimit -c unlimited

source activate /scratch/ravihm.scee.iitmandi/pytorch

cd /home/ravihm.scee.iitmandi/Certified-Robustness

python /scratch/ravihm.scee.iitmandi/Vatsal/Certified-Robustness/hold.py