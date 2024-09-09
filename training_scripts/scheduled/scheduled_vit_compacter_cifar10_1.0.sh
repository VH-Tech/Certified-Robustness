#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --time=4-00:00:00
#SBATCH --job-name compacter-0.1.job

ulimit -s unlimited
ulimit -c unlimited

source activate /scratch/ravihm.scee.iitmandi/robustness

cd /scratch/ravihm.scee.iitmandi/Vatsal/Certified-Robustness/

accelerate launch train_denoiser_adapter.py --dataset cifar10 --arch vit --outdir /scratch/ravihm.scee.iitmandi/models/cifar10/vit --batch 128 --data_dir /scratch/ravihm.scee.iitmandi/dataset/cifar10  --workers 8 --noise_sd 0.25 --dataset_fraction 0.1 --scheduler step --lr_step_size 40 --epochs 120 --lr 0.1
accelerate launch train_denoiser_adapter.py --dataset cifar10 --arch vit --outdir /scratch/ravihm.scee.iitmandi/models/cifar10/vit --batch 128 --data_dir /scratch/ravihm.scee.iitmandi/dataset/cifar10  --workers 8 --noise_sd 0.5 --dataset_fraction 0.1 --scheduler step --lr_step_size 40 --epochs 120 --lr 0.1



