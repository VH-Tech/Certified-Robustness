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

CUDA_VISIBLE_DEVICES=0,1 accelerate launch train_classifier.py --dataset hyper --arch vit_custom --outdir /scratch/ravihm.scee.iitmandi/models/hyper/vit --batch 64 --data_dir /scratch/ravihm.scee.iitmandi/dataset/hyper --workers 6 --lr 1e-2

CUDA_VISIBLE_DEVICES=0,1 accelerate launch train_classifier.py --dataset hyper --arch vit_custom --outdir /scratch/ravihm.scee.iitmandi/models/hyper/vit_custom --batch 64 --data_dir /scratch/ravihm.scee.iitmandi/dataset/hyper --workers 6 --tuning_method compacter --noise_sd 0.25 --lr 1e-2

CUDA_VISIBLE_DEVICES=0,1 accelerate launch train_classifier.py --dataset hyper --arch vit_custom --outdir /scratch/ravihm.scee.iitmandi/models/hyper/vit_custom --batch 64 --data_dir /scratch/ravihm.scee.iitmandi/dataset/hyper --workers 6 --tuning_method compacter --noise_sd 1.0 --lr 1e-2

CUDA_VISIBLE_DEVICES=0,1 accelerate launch train_classifier.py --dataset hyper --arch vit_custom --outdir /scratch/ravihm.scee.iitmandi/models/hyper/vit_custom --batch 64 --data_dir /scratch/ravihm.scee.iitmandi/dataset/hyper --workers 6 --tuning_method compacter --noise_sd 0.25 --dataset_fraction 0.5 --lr 1e-2

CUDA_VISIBLE_DEVICES=0,1 accelerate launch train_classifier.py --dataset hyper --arch vit_custom --outdir /scratch/ravihm.scee.iitmandi/models/hyper/vit_custom --batch 64 --data_dir /scratch/ravihm.scee.iitmandi/dataset/hyper --workers 6 --tuning_method compacter --noise_sd 1.0 --dataset_fraction 0.5 --lr 1e-2

CUDA_VISIBLE_DEVICES=0,1 accelerate launch train_classifier.py --dataset hyper --arch vit_custom --outdir /scratch/ravihm.scee.iitmandi/models/hyper/vit_custom --batch 64 --data_dir /scratch/ravihm.scee.iitmandi/dataset/hyper --workers 6 --tuning_method compacter --noise_sd 0.25 --dataset_fraction 0.1 --lr 1e-2

CUDA_VISIBLE_DEVICES=0,1 accelerate launch train_classifier.py --dataset hyper --arch vit_custom --outdir /scratch/ravihm.scee.iitmandi/models/hyper/vit_custom --batch 64 --data_dir /scratch/ravihm.scee.iitmandi/dataset/hyper --workers 6 --tuning_method compacter --noise_sd 1.0 --dataset_fraction 0.1 --lr 1e-2

CUDA_VISIBLE_DEVICES=0,1 accelerate launch train_classifier.py --dataset hyper --arch vit_custom --outdir /scratch/ravihm.scee.iitmandi/models/hyper/vit_custom --batch 64 --data_dir /scratch/ravihm.scee.iitmandi/dataset/hyper --workers 6 --tuning_method compacter --noise_sd 0.25 --dataset_fraction 0.01 --lr 1e-2

CUDA_VISIBLE_DEVICES=0,1 accelerate launch train_classifier.py --dataset hyper --arch vit_custom --outdir /scratch/ravihm.scee.iitmandi/models/hyper/vit_custom --batch 64 --data_dir /scratch/ravihm.scee.iitmandi/dataset/hyper --workers 6 --tuning_method compacter --noise_sd 1.0 --dataset_fraction 0.01 --lr 1e-2


CUDA_VISIBLE_DEVICES=0,1 accelerate launch train_classifier.py --dataset hyper --arch swin --outdir /scratch/ravihm.scee.iitmandi/models/hyper/swin --batch 64 --data_dir /scratch/ravihm.scee.iitmandi/dataset/hyper --workers 6 --tuning_method compacter --noise_sd 0.25  --epochs 20 --lr 5e-4 

CUDA_VISIBLE_DEVICES=0,1 accelerate launch train_classifier.py --dataset hyper --arch swin --outdir /scratch/ravihm.scee.iitmandi/models/hyper/swin --batch 64 --data_dir /scratch/ravihm.scee.iitmandi/dataset/hyper --workers 6 --tuning_method compacter --noise_sd 1.0  --epochs 20 --lr 5e-4

CUDA_VISIBLE_DEVICES=0,1 accelerate launch train_classifier.py --dataset hyper --arch swin --outdir /scratch/ravihm.scee.iitmandi/models/hyper/swin --batch 64 --data_dir /scratch/ravihm.scee.iitmandi/dataset/hyper --workers 6 --tuning_method compacter --noise_sd 0.25 --dataset_fraction 0.5  --epochs 20 --lr 5e-4

CUDA_VISIBLE_DEVICES=0,1 accelerate launch train_classifier.py --dataset hyper --arch swin --outdir /scratch/ravihm.scee.iitmandi/models/hyper/swin --batch 64 --data_dir /scratch/ravihm.scee.iitmandi/dataset/hyper --workers 6 --tuning_method compacter --noise_sd 1.0 --dataset_fraction 0.5 --epochs 20 --lr 5e-4

CUDA_VISIBLE_DEVICES=0,1 accelerate launch train_classifier.py --dataset hyper --arch swin --outdir /scratch/ravihm.scee.iitmandi/models/hyper/swin --batch 64 --data_dir /scratch/ravihm.scee.iitmandi/dataset/hyper --workers 6 --tuning_method compacter --noise_sd 0.25 --dataset_fraction 0.1 --epochs 20 --lr 5e-4

CUDA_VISIBLE_DEVICES=0,1 accelerate launch train_classifier.py --dataset hyper --arch swin --outdir /scratch/ravihm.scee.iitmandi/models/hyper/swin --batch 64 --data_dir /scratch/ravihm.scee.iitmandi/dataset/hyper --workers 6 --tuning_method compacter --noise_sd 1.0 --dataset_fraction 0.1 --epochs 20 --lr 5e-4

CUDA_VISIBLE_DEVICES=0,1 accelerate launch train_classifier.py --dataset hyper --arch swin --outdir /scratch/ravihm.scee.iitmandi/models/hyper/swin --batch 64 --data_dir /scratch/ravihm.scee.iitmandi/dataset/hyper --workers 6 --tuning_method compacter --noise_sd 0.25 --dataset_fraction 0.01 --epochs 20 --lr 5e-4

CUDA_VISIBLE_DEVICES=0,1 accelerate launch train_classifier.py --dataset hyper --arch swin --outdir /scratch/ravihm.scee.iitmandi/models/hyper/swin --batch 64 --data_dir /scratch/ravihm.scee.iitmandi/dataset/hyper --workers 6 --tuning_method compacter --noise_sd 1.0 --dataset_fraction 0.01 --epochs 20 --lr 5e-4