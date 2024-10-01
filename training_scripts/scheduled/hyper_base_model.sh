#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=2
#SBATCH --time=4-00:00:00
#SBATCH --job-name hyper-base.job

ulimit -s unlimited
ulimit -c unlimited

source activate /home/ravihm.scee.iitmandi/.conda/envs/nemo

cd /scratch/ravihm.scee.iitmandi/Vatsal/Certified-Robustness/

accelerate launch train_classifier.py --dataset hyper --arch vit --outdir /scratch/ravihm.scee.iitmandi/models/hyper/vit/ --data_dir /scratch/ravihm.scee.iitmandi/dataset/hyper/