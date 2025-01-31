# this file is based on code publicly available at
#   https://github.com/bearpaw/pytorch-classification
# written by Wei Yang.

from accelerate import Accelerator
from architectures import CLASSIFIERS_ARCHITECTURES, get_architecture
from datasets import get_dataset, DATASETS
from loss import FocalLoss
from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from train_utils import AverageMeter, accuracy, init_logfile, log, copy_code
from mixup_utils import ComboLoader, get_combo_loader
import torch.nn.functional as F

from utils import count_parameters_total, count_parameters_trainable
import numpy as np
from transformers import TrainingArguments, EvalPrediction
from adapters import AdapterTrainer

from torch.utils.data import Dataset
import json
import adapters
import argparse
import datetime
import numpy as np
import os
import time
import torch
import torchvision
import random

from adapters import ParBnConfig, SeqBnConfig, SeqBnInvConfig, PrefixTuningConfig, CompacterConfig, LoRAConfig, IA3Config
accelerator = Accelerator()
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--arch', type=str, default="google/vit-base-patch16-224-in21k",
                    choices=CLASSIFIERS_ARCHITECTURES)
parser.add_argument('--outdir', type=str, 
                    help='folder to save model and training log)')

parser.add_argument('--noise_sd', default=0.0, type=float,
                    help="standard deviation of noise distribution for data augmentation")
parser.add_argument('--resume', action='store_true',
                    help='if true, tries to resume training from an existing checkpoint')

parser.add_argument("--adapter_config", type=str, help="Adapter config")

parser.add_argument('--mixup_lam', type=float, default=0.1, 
                    help='mixup lambda')
parser.add_argument('--mixup_mode', type=str, default='class', 
                    help='sampling mode (instance, class, sqrt, prog)')
parser.add_argument('--mixup', type=int, default=0, 
                    help='do mixup')
parser.add_argument('--ssl_like', type=int, default=0, 
                    help='do ssl like criterion')

args = parser.parse_args()


torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
VIT = False

def main():
    if args.gpu:
        # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        pass
    folder = "/adapters"

    if args.mixup:
        folder += "_mixup"

    if args.focal :
        folder += "_focal"

    config = "seq_bn"
    if args.adapter_config:
        folder += "_"+args.adapter_config

        if args.adapter_config == "par_bn":
            config = ParBnConfig()

        elif args.adapter_config == "seq_bn":
            config = SeqBnConfig()

        elif args.adapter_config == "seq_bn_inv":
            config = SeqBnInvConfig()  

        elif args.adapter_config == "prefix_tuning":
            config = PrefixTuningConfig()

        elif args.adapter_config == "compacter":
            config = CompacterConfig()

        elif args.adapter_config == "lora":
            config = LoRAConfig()

        elif args.adapter_config == "ia3":
            config = IA3Config()

    
    if not os.path.exists(args.outdir+folder+"/"+str(args.noise_sd)):
        print("path not found")
        return

    model = get_architecture(args.arch, args.dataset)
    _ , model = model
    adapters.init(model)
    model.add_adapter("denoising-adapter", config=config)

    adapter_path = args.outdir+folder+"/"+str(args.noise_sd)+"/checkpoint.pth.tar"
    checkpoint = torch.load(adapter_path,
                                map_location=lambda storage, loc: storage)
    # starting_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    #set active adapter
    model.set_active_adapters("denoising-adapter")
    model.train_adapter("denoising-adapter")
    model.save_adapter(args.outdir+folder+"/"+str(args.noise_sd), "denoising-adapter")
    torch.save({
                'epoch': checkpoint["epoch"],
                'arch': checkpoint["arch"],
                'optimizer': checkpoint["optimizer"],
            }, os.path.join(args.outdir, 'checkpoint.pth.tar'))

if __name__ == "__main__":
    main()
