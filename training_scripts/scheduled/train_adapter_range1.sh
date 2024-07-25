#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --time=4-00:00:00
#SBATCH --job-name Certified-Robustness.job

ulimit -s unlimited
ulimit -c unlimited

source activate /scratch/ravihm.scee.iitmandi/pytorch
cd /home/ravihm.scee.iitmandi/Certified-Robustness

python train_denoiser_adapter.py --dataset cifar10 --arch vit --outdir /scratch/ravihm.scee.iitmandi/models/cifar10/vit/ --batch 128 --noise_sd 0.25 --data_dir /scratch/ravihm.scee.iitmandi/dataset/cifar10/ --dataset_fraction 0.1 --train_range 1
python train_denoiser_adapter.py --dataset cifar10 --arch vit --outdir /scratch/ravihm.scee.iitmandi/models/cifar10/vit/ --batch 128 --noise_sd 0.5 --data_dir /scratch/ravihm.scee.iitmandi/dataset/cifar10/ --dataset_fraction 0.1 --train_range 1