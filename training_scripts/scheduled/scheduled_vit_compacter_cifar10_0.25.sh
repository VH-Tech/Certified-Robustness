#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --time=4-00:00:00
#SBATCH --job-name 0.25-Certified-Robustness.job

ulimit -s unlimited
ulimit -c unlimited

source activate /scratch/ravihm.scee.iitmandi/pytorch

cd /scratch/ravihm.scee.iitmandi/Vatsal/Certified-Robustness/

accelerate launch train_denoiser_adapter.py --dataset cifar10 --arch vit --outdir /scratch/ravihm.scee.iitmandi/models/cifar10/vit --batch 128 --data_dir /scratch/ravihm.scee.iitmandi/dataset/cifar10  --workers 8 --noise_sd 0.25 --dataset_fraction 1.0 --scheduler step --lr_step_size 60 --epochs 180

accelerate launch train_denoiser_adapter.py --dataset cifar10 --arch vit --outdir /scratch/ravihm.scee.iitmandi/models/cifar10/vit --batch 128 --data_dir /scratch/ravihm.scee.iitmandi/dataset/cifar10  --workers 8 --noise_sd 0.25 --dataset_fraction 0.5 --scheduler step --lr_step_size 60 --epochs 180

accelerate launch train_denoiser_adapter.py --dataset cifar10 --arch vit --outdir /scratch/ravihm.scee.iitmandi/models/cifar10/vit --batch 128 --data_dir /scratch/ravihm.scee.iitmandi/dataset/cifar10  --workers 8 --noise_sd 0.25 --dataset_fraction 0.1 --scheduler step --lr_step_size 60 --epochs 180

accelerate launch train_denoiser_adapter.py --dataset cifar10 --arch vit --outdir /scratch/ravihm.scee.iitmandi/models/cifar10/vit --batch 128 --data_dir /scratch/ravihm.scee.iitmandi/dataset/cifar10  --workers 8 --noise_sd 0.25 --dataset_fraction 0.01 --scheduler step --lr_step_size 60 --epochs 180

accelerate launch train_denoiser_adapter.py --dataset cifar10 --arch vit --outdir /scratch/ravihm.scee.iitmandi/models/cifar10/vit --batch 128 --data_dir /scratch/ravihm.scee.iitmandi/dataset/cifar10  --workers 8 --noise_sd 0.25 --dataset_fraction 0.001 --scheduler step --lr_step_size 60 --epochs 180

accelerate launch train_denoiser_adapter.py --dataset cifar10 --arch vit --outdir /scratch/ravihm.scee.iitmandi/models/cifar10/vit --batch 128 --data_dir /scratch/ravihm.scee.iitmandi/dataset/cifar10  --workers 8 --noise_sd 0.25 --dataset_fraction 0.0005 --scheduler step --lr_step_size 60 --epochs 180

accelerate launch train_denoiser_adapter.py --dataset cifar10 --arch vit --outdir /scratch/ravihm.scee.iitmandi/models/cifar10/vit --batch 128 --data_dir /scratch/ravihm.scee.iitmandi/dataset/cifar10  --workers 8 --noise_sd 0.25 --dataset_fraction 0.0002 --scheduler step --lr_step_size 60 --epochs 180
