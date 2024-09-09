#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --time=4-00:00:00
#SBATCH --job-name RS-adapters-certify.job

ulimit -s unlimited
ulimit -c unlimited

source activate /scratch/ravihm.scee.iitmandi/robustness

cd /scratch/ravihm.scee.iitmandi/Vatsal/Certified-Robustness/

python certify_adapters.py --dataset cifar10 --base_classifier vit --sigma 1.0 --outfile certify_adapters_RS_1.0.txt --N 10000 --skip 20 --adapter /scratch/ravihm.scee.iitmandi/models/cifar10/vit/adapters_compacter_1.0/1.0/
