#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=2
#SBATCH --time=1-00:00:00
#SBATCH --job-name Diffusion-Certified-Robustness.job

ulimit -s unlimited
ulimit -c unlimited

source activate /scratch/ravihm.scee.iitmandi/pytorch
cd /scratch/ravihm.scee.iitmandi/Vatsal/improved-diffusion/

MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"
TRAIN_FLAGS="--lr 1e-4 --batch_size 128"
OPENAI_LOGDIR="/scratch/ravihm.scee.iitmandi/models/cifar10/diffusion/0.0005"

mpiexec -n 2 python scripts/image_train.py --data_dir /scratch/ravihm.scee.iitmandi/dataset/cifar10_0.0005 $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
