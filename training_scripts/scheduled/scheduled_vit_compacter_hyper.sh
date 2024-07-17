#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=2
#SBATCH --time=4-00:00:00
#SBATCH --job-name Certified-Robustness.job

ulimit -s unlimited
ulimit -c unlimited

source activate /scratch/ravihm.scee.iitmandi/pytorch

cd /home/ravihm.scee.iitmandi/Certified-Robustness

python train_classifier.py --dataset hyper --arch vit --outdir /scratch/ravihm.scee.iitmandi/models/hyper/vit --workers 8 --data_dir /scratch/ravihm.scee.iitmandi/dataset/hyper

CUDA_VISIBLE_DEVICES=0,1 accelerate launch train_denoiser_adapter.py --dataset hyper --arch vit --outdir /scratch/ravihm.scee.iitmandi/models/hyper/vit --batch 64 --data_dir /scratch/ravihm.scee.iitmandi/dataset/hyper  --workers 8 --noise_sd 0.25 --dataset_fraction 0.001 --scheduler step

CUDA_VISIBLE_DEVICES=0,1 accelerate launch train_denoiser_adapter.py --dataset hyper --arch vit --outdir /scratch/ravihm.scee.iitmandi/models/hyper/vit --batch 64 --data_dir /scratch/ravihm.scee.iitmandi/dataset/hyper  --workers 8 --noise_sd 1.0 --dataset_fraction 0.001 --scheduler step

CUDA_VISIBLE_DEVICES=0,1 accelerate launch train_denoiser_adapter.py --dataset hyper --arch vit --outdir /scratch/ravihm.scee.iitmandi/models/hyper/vit --batch 64 --data_dir /scratch/ravihm.scee.iitmandi/dataset/hyper  --workers 8 --noise_sd 0.25 --dataset_fraction 0.01 --scheduler step

CUDA_VISIBLE_DEVICES=0,1 accelerate launch train_denoiser_adapter.py --dataset hyper --arch vit --outdir /scratch/ravihm.scee.iitmandi/models/hyper/vit --batch 64 --data_dir /scratch/ravihm.scee.iitmandi/dataset/hyper  --workers 8 --noise_sd 1.0 --dataset_fraction 0.01 --scheduler step