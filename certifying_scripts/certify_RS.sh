#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --time=4-00:00:00
#SBATCH --job-name RS-certify.job

ulimit -s unlimited
ulimit -c unlimited

source activate /scratch/ravihm.scee.iitmandi/pytorch

cd /scratch/ravihm.scee.iitmandi/Vatsal/Certified-Robustness/

python certify.py --dataset cifar10 --base_classifier /scratch/ravihm.scee.iitmandi/models/cifar10/vit/1.0/full_0.25/checkpoint.pth.tar --sigma 0.25 --outfile certify_RS_0.25.txt --N 1000 --skip 10
python certify.py --dataset cifar10 --base_classifier /scratch/ravihm.scee.iitmandi/models/cifar10/vit/1.0/full_0.5/checkpoint.pth.tar --sigma 0.5 --outfile certify_RS_0.5.txt --N 1000 --skip 10
python certify.py --dataset cifar10 --base_classifier /scratch/ravihm.scee.iitmandi/models/cifar10/vit/1.0/full_0.75/checkpoint.pth.tar --sigma 0.75 --outfile certify_RS_0.75.txt --N 1000 --skip 10
python certify.py --dataset cifar10 --base_classifier /scratch/ravihm.scee.iitmandi/models/cifar10/vit/1.0/full_1.0/checkpoint.pth.tar --sigma 1.0 --outfile certify_RS_1.0.txt --N 1000 --skip 10
