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


accelerate launch train_denoiser_adapter.py --dataset cifar10 --arch vit --outdir /scratch/ravihm.scee.iitmandi/models/cifar10/vit --batch 64 --data_dir /scratch/ravihm.scee.iitmandi/dataset/cifar10  --workers 8 --noise_sd 0.25 --dataset_fraction 0.5 --do_fourier 1 --gap 1 --fourier_location adapter --scheduler cosine --invert_domain 1
accelerate launch train_denoiser_adapter.py --dataset cifar10 --arch vit --outdir /scratch/ravihm.scee.iitmandi/models/cifar10/vit --batch 64 --data_dir /scratch/ravihm.scee.iitmandi/dataset/cifar10  --workers 8 --noise_sd 0.25 --dataset_fraction 0.5 --do_fourier 1 --gap 1 --fourier_location adapter --scheduler cosine --lr 1e-3 --invert_domain 1


accelerate launch train_denoiser_adapter.py --dataset cifar10 --arch vit --outdir /scratch/ravihm.scee.iitmandi/models/cifar10/vit --batch 64 --data_dir /scratch/ravihm.scee.iitmandi/dataset/cifar10  --workers 8 --noise_sd 0.25 --dataset_fraction 0.5 --do_fourier 1 --gap 1 --fourier_location adapter --adapter_config seq_bn --scheduler cosine --invert_domain 1
accelerate launch train_denoiser_adapter.py --dataset cifar10 --arch vit --outdir /scratch/ravihm.scee.iitmandi/models/cifar10/vit --batch 64 --data_dir /scratch/ravihm.scee.iitmandi/dataset/cifar10  --workers 8 --noise_sd 0.25 --dataset_fraction 0.5 --do_fourier 1 --gap 1 --fourier_location adapter --adapter_config seq_bn --scheduler cosine --lr 1e-3 --invert_domain 1


accelerate launch train_denoiser_adapter.py --dataset cifar10 --arch vit --outdir /scratch/ravihm.scee.iitmandi/models/cifar10/vit --batch 64 --data_dir /scratch/ravihm.scee.iitmandi/dataset/cifar10  --workers 8 --noise_sd 0.25 --dataset_fraction 0.5 --do_fourier 1 --gap 1 --fourier_location adapter --adapter_config par_bn --scheduler cosine --invert_domain 1
accelerate launch train_denoiser_adapter.py --dataset cifar10 --arch vit --outdir /scratch/ravihm.scee.iitmandi/models/cifar10/vit --batch 64 --data_dir /scratch/ravihm.scee.iitmandi/dataset/cifar10  --workers 8 --noise_sd 0.25 --dataset_fraction 0.5 --do_fourier 1 --gap 1 --fourier_location adapter --adapter_config par_bn  --scheduler cosine --lr 1e-3 --invert_domain 1
