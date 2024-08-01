#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=2
#SBATCH --time=4-00:00:00
#SBATCH --job-name 0.25-Full-Finetune.job

ulimit -s unlimited
ulimit -c unlimited

source activate /scratch/ravihm.scee.iitmandi/pytorch

cd /scratch/ravihm.scee.iitmandi/Vatsal/Certified-Robustness/

accelerate launch full_finetune.py --dataset cifar10 --arch vit --outdir /scratch/ravihm.scee.iitmandi/models/cifar10/vit --noise_sd 0.25 --data_dir /scratch/ravihm.scee.iitmandi/dataset/cifar10 --dataset_fraction 1.0 --epochs 120
accelerate launch full_finetune.py --dataset cifar10 --arch vit --outdir /scratch/ravihm.scee.iitmandi/models/cifar10/vit --noise_sd 0.25 --data_dir /scratch/ravihm.scee.iitmandi/dataset/cifar10 --dataset_fraction 0.5 --epochs 120
accelerate launch full_finetune.py --dataset cifar10 --arch vit --outdir /scratch/ravihm.scee.iitmandi/models/cifar10/vit --noise_sd 0.25 --data_dir /scratch/ravihm.scee.iitmandi/dataset/cifar10 --dataset_fraction 0.1 --epochs 120
