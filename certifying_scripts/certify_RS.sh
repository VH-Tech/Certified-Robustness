#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=2
#SBATCH --time=4-00:00:00
#SBATCH --job-name RS-certify.job

ulimit -s unlimited
ulimit -c unlimited

source activate /scratch/ravihm.scee.iitmandi/robustness

cd /scratch/ravihm.scee.iitmandi/Vatsal/Certified-Robustness/

python certify.py --dataset cifar10 --base_classifier /scratch/ravihm.scee.iitmandi/models/cifar10/vit/RS_1.0/0.5/checkpoint.pth.tar --sigma 0.5 --outfile certify_RS_0.5.txt --N 10000 --skip 20 --batch 1000
python certify.py --dataset cifar10 --base_classifier /scratch/ravihm.scee.iitmandi/models/cifar10/vit/RS_1.0/0.75/checkpoint.pth.tar --sigma 0.75 --outfile certify_RS_0.75.txt --N 10000 --skip 20 --batch 1000
python certify.py --dataset cifar10 --base_classifier /scratch/ravihm.scee.iitmandi/models/cifar10/vit/RS_1.0/1.0/checkpoint.pth.tar --sigma 1.0 --outfile certify_RS_1.0.txt --N 10000 --skip 20 --batch 1000
