CUDA_VISIBLE_DEVICES=0,1 accelerate launch train_classifier.py --dataset hyper --arch vit --outdir /scratch/ravihm.scee.iitmandi/models/hyper/vit --batch 64 --data_dir /scratch/ravihm.scee.iitmandi/dataset/hyper --workers 6

CUDA_VISIBLE_DEVICES=0,1 accelerate launch train_classifier.py --dataset hyper --arch swin --outdir /scratch/ravihm.scee.iitmandi/models/hyper/swin --batch 16 --data_dir /scratch/ravihm.scee.iitmandi/dataset/hyper 

CUDA_VISIBLE_DEVICES=0,1 accelerate launch train_classifier.py --dataset hyper --arch resnet50 --outdir /scratch/ravihm.scee.iitmandi/models/hyper/resnet50 --batch 64 --data_dir /scratch/ravihm.scee.iitmandi/dataset/hyper --workers 6