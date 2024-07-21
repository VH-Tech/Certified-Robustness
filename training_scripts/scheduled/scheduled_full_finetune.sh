#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --time=4-00:00:00
#SBATCH --job-name Full-Finetune.job

ulimit -s unlimited
ulimit -c unlimited

source activate /scratch/ravihm.scee.iitmandi/pytorch

cd /home/ravihm.scee.iitmandi/Certified-Robustness

python full_finetune.py --dataset cifar10 --arch vit --outdir /scratch/ravihm.scee.iitmandi/models/cifar10/vit --noise_sd 0.25 --data_dir /scratch/ravihm.scee.iitmandi/dataset/cifar10 --dataset_fraction 0.01
python full_finetune.py --dataset cifar10 --arch vit --outdir /scratch/ravihm.scee.iitmandi/models/cifar10/vit --noise_sd 0.25 --data_dir /scratch/ravihm.scee.iitmandi/dataset/cifar10 --dataset_fraction 0.001
python full_finetune.py --dataset cifar10 --arch vit --outdir /scratch/ravihm.scee.iitmandi/models/cifar10/vit --noise_sd 0.25 --data_dir /scratch/ravihm.scee.iitmandi/dataset/cifar10 --dataset_fraction 0.0005
python full_finetune.py --dataset cifar10 --arch vit --outdir /scratch/ravihm.scee.iitmandi/models/cifar10/vit --noise_sd 0.25 --data_dir /scratch/ravihm.scee.iitmandi/dataset/cifar10 --dataset_fraction 0.0002

python full_finetune.py --dataset cifar10 --arch vit --outdir /scratch/ravihm.scee.iitmandi/models/cifar10/vit --noise_sd 1.0 --data_dir /scratch/ravihm.scee.iitmandi/dataset/cifar10 --dataset_fraction 0.01
python full_finetune.py --dataset cifar10 --arch vit --outdir /scratch/ravihm.scee.iitmandi/models/cifar10/vit --noise_sd 1.0 --data_dir /scratch/ravihm.scee.iitmandi/dataset/cifar10 --dataset_fraction 0.001
python full_finetune.py --dataset cifar10 --arch vit --outdir /scratch/ravihm.scee.iitmandi/models/cifar10/vit --noise_sd 1.0 --data_dir /scratch/ravihm.scee.iitmandi/dataset/cifar10 --dataset_fraction 0.0005
python full_finetune.py --dataset cifar10 --arch vit --outdir /scratch/ravihm.scee.iitmandi/models/cifar10/vit --noise_sd 1.0 --data_dir /scratch/ravihm.scee.iitmandi/dataset/cifar10 --dataset_fraction 0.0002