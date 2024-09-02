#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --time=4-00:00:00
#SBATCH --job-name Denoisers-certify.job

ulimit -s unlimited
ulimit -c unlimited

source activate /scratch/ravihm.scee.iitmandi/robustness

cd /scratch/ravihm.scee.iitmandi/Vatsal/Certified-Robustness/

python certify.py --dataset cifar10 --base_classifier vit --sigma 1.0 --outfile certify_denoiser_1.0.txt --denoiser /scratch/ravihm.scee.iitmandi/models/cifar10/vit/denoiser_1.0_1.0/checkpoint.pth.tar --N 1000 --batch 512 --skip 20 
python certify.py --dataset cifar10 --base_classifier vit --sigma 0.75 --outfile certify_denoiser_0.75.txt --denoiser /scratch/ravihm.scee.iitmandi/models/cifar10/vit/denoiser_1.0_0.75/checkpoint.pth.tar --N 1000 --batch 512 --skip 20
python certify.py --dataset cifar10 --base_classifier vit --sigma 0.5 --outfile certify_denoiser_0.5.txt --denoiser /scratch/ravihm.scee.iitmandi/models/cifar10/vit/denoiser_1.0_0.5/checkpoint.pth.tar --N 1000 --batch 512 --skip 20
python certify.py --dataset cifar10 --base_classifier vit --sigma 0.25 --outfile certify_denoiser_0.25.txt --denoiser /scratch/ravihm.scee.iitmandi/models/cifar10/vit/denoiser_1.0_0.25/checkpoint.pth.tar --N 1000 --batch 512 --skip 20
