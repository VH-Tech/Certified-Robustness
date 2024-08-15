#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=2
#SBATCH --time=4-00:00:00
#SBATCH --job-name 0.25-denoiser.job

ulimit -s unlimited
ulimit -c unlimited

source activate /scratch/ravihm.scee.iitmandi/pytorch

cd /scratch/ravihm.scee.iitmandi/Vatsal/Certified-Robustness/

accelerate launch train_denoiser.py --dataset cifar10 --dataset_fraction 1.0 --outdir /scratch/ravihm.scee.iitmandi/models/cifar10/vit/denoiser_1.0_0.25 --data_dir /scratch/ravihm.scee.iitmandi/dataset/cifar10 --objective stability --arch cifar_dncnn --classifier vit --noise_sd 0.25 --workers 8
accelerate launch train_denoiser.py --dataset cifar10 --dataset_fraction 1.0 --outdir /scratch/ravihm.scee.iitmandi/models/cifar10/vit/denoiser_1.0_0.25 --data_dir /scratch/ravihm.scee.iitmandi/dataset/cifar10 --objective stability --arch cifar_dncnn --classifier vit --noise_sd 0.5 --workers 8
accelerate launch train_denoiser.py --dataset cifar10 --dataset_fraction 1.0 --outdir /scratch/ravihm.scee.iitmandi/models/cifar10/vit/denoiser_1.0_0.25 --data_dir /scratch/ravihm.scee.iitmandi/dataset/cifar10 --objective stability --arch cifar_dncnn --classifier vit --noise_sd 0.75 --workers 8
accelerate launch train_denoiser.py --dataset cifar10 --dataset_fraction 1.0 --outdir /scratch/ravihm.scee.iitmandi/models/cifar10/vit/denoiser_1.0_0.25 --data_dir /scratch/ravihm.scee.iitmandi/dataset/cifar10 --objective stability --arch cifar_dncnn --classifier vit --noise_sd 1.0 --workers 8
