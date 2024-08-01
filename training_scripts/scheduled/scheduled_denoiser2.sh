#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=2
#SBATCH --time=4-00:00:00
#SBATCH --job-name 1.0-denoiser.job

ulimit -s unlimited
ulimit -c unlimited

source activate /scratch/ravihm.scee.iitmandi/pytorch

cd /scratch/ravihm.scee.iitmandi/Vatsal/Certified-Robustness/

accelerate launch train_denoiser.py --dataset cifar10 --dataset_fraction 1.0 --outdir /scratch/ravihm.scee.iitmandi/models/cifar10/vit/denoiser_1.0_1.0 --data_dir /scratch/ravihm.scee.iitmandi/dataset/cifar10 --objective stability --arch cifar_dncnn --classifier vit --noise_sd 1.0 --workers 8
accelerate launch train_denoiser.py --dataset cifar10 --dataset_fraction 0.5 --outdir /scratch/ravihm.scee.iitmandi/models/cifar10/vit/denoiser_0.5_1.0 --data_dir /scratch/ravihm.scee.iitmandi/dataset/cifar10 --objective stability --arch cifar_dncnn --classifier vit --noise_sd 1.0 --workers 8
accelerate launch train_denoiser.py --dataset cifar10 --dataset_fraction 0.1 --outdir /scratch/ravihm.scee.iitmandi/models/cifar10/vit/denoiser_0.1_1.0 --data_dir /scratch/ravihm.scee.iitmandi/dataset/cifar10 --objective stability --arch cifar_dncnn --classifier vit --noise_sd 1.0 --workers 8
accelerate launch train_denoiser.py --dataset cifar10 --dataset_fraction 0.01 --outdir /scratch/ravihm.scee.iitmandi/models/cifar10/vit/denoiser_0.01_1.0 --data_dir /scratch/ravihm.scee.iitmandi/dataset/cifar10 --objective stability --arch cifar_dncnn --classifier vit --noise_sd 1.0 --workers 8
accelerate launch train_denoiser.py --dataset cifar10 --dataset_fraction 0.001 --outdir /scratch/ravihm.scee.iitmandi/models/cifar10/vit/denoiser_0.001_1.0 --data_dir /scratch/ravihm.scee.iitmandi/dataset/cifar10 --objective stability --arch cifar_dncnn --classifier vit --noise_sd 1.0 --workers 8
accelerate launch train_denoiser.py --dataset cifar10 --dataset_fraction 0.0002 --outdir /scratch/ravihm.scee.iitmandi/models/cifar10/vit/denoiser_0.0002_1.0 --data_dir /scratch/ravihm.scee.iitmandi/dataset/cifar10 --objective stability --arch cifar_dncnn --classifier vit --noise_sd 1.0 --workers 8
accelerate launch train_denoiser.py --dataset cifar10 --dataset_fraction 0.0005 --outdir /scratch/ravihm.scee.iitmandi/models/cifar10/vit/denoiser_0.0005_1.0 --data_dir /scratch/ravihm.scee.iitmandi/dataset/cifar10 --objective stability --arch cifar_dncnn --classifier vit --noise_sd 1.0 --workers 8