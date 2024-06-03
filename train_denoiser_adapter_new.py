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
from torchvision import transforms, datasets
from adapters import ParBnConfig, SeqBnConfig, SeqBnInvConfig, PrefixTuningConfig, CompacterConfig, LoRAConfig, IA3Config
from adapters import AdapterTrainer


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset', type=str, 
                    choices=DATASETS)
parser.add_argument('--arch', type=str, default="google/vit-base-patch16-224-in21k",
                    choices=CLASSIFIERS_ARCHITECTURES)
parser.add_argument('--outdir', type=str, 
                    help='folder to save model and training log)')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch', default=64, type=int, metavar='N',
                    help='batchsize (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--lr_step_size', type=int, default=30,
                    help='How often to decrease learning by gamma.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--gpu', default=None, type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--noise_sd', default=0.0, type=float,
                    help="standard deviation of noise distribution for data augmentation")
parser.add_argument('--resume', action='store_true',
                    help='if true, tries to resume training from an existing checkpoint')
parser.add_argument('--azure_datastore_path', type=str, default='',
                    help='Path to imagenet on azure')
parser.add_argument('--philly_imagenet_path', type=str, default='',
                    help='Path to imagenet on philly')
parser.add_argument('--focal', default=0, type=int,
                    help='use focal loss')
parser.add_argument('--data_dir', type=str, default='./data',
                    help='Path to data directory')
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
        os.makedirs(args.outdir+folder+"/"+str(args.noise_sd))

    train_dataset = get_dataset(args.dataset, 'train', args.data_dir, noise_sd=args.noise_sd)
    test_dataset = get_dataset(args.dataset, 'test', args.data_dir, noise_sd=args.noise_sd)

    pin_memory = (args.dataset == "imagenet")
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch,
                              num_workers=args.workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch,
                             num_workers=args.workers, pin_memory=pin_memory)

    model = get_architecture(args.arch, args.dataset)

    if "vit" in args.arch:
        global VIT
        VIT = True

    if args.focal:

        class_frequency = torch.unique(torch.tensor(train_dataset.targets), return_counts=True)[1] if args.dataset == 'pneumonia' else train_dataset.get_frequency()
        w = torch.tensor([sum(class_frequency) / (len(class_frequency) * class_frequency[i]) for i in range(len(class_frequency))])
        print('Class Frequency and weights for train dataset: ', class_frequency, w)
        criterion = FocalLoss(alpha=w.cuda(), gamma=2).cuda() 
    else:
        criterion = CrossEntropyLoss().cuda()


    starting_epoch = 0

    logfilename = os.path.join(args.outdir+folder+"/"+str(args.noise_sd), 'log.txt')

    ## Resume from checkpoint if exists and if resume flag is True
    adapter_path = args.outdir+folder+"/"+str(args.noise_sd)
    model_path = os.path.join(args.outdir, 'checkpoint.pth.tar')

    if os.path.isfile(model_path):
        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path,
                                map_location=lambda storage, loc: storage)
        # starting_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

        print("=> loaded checkpoint '{}' (epoch {})"
                        .format(model_path, checkpoint['epoch']))
        
        #add adapters
        # print(model)
        normalize_layer, model = model
        adapters.init(model)

        if args.resume : 
            # load adapter from path
            model.load_adapter(adapter_path)
            # checkpoint_adapter = torch.load(os.path.join(adapter_path, 'checkpoint.pth.tar'),
            #                     map_location=lambda storage, loc: storage)
            # # starting_epoch = checkpoint_adapter['epoch']
            # best = checkpoint_adapter['test_acc']

        else:
            model.add_adapter("denoising-adapter", config=config)
            init_logfile(logfilename, "epoch\ttime\tlr\ttrainloss\ttestloss\ttrainAcc\ttestAcc")
            # best = 0.0 

        #set active adapter
        model.set_active_adapters("denoising-adapter")
        model.train_adapter("denoising-adapter")

    else:
        print("=> no checkpoint found at '{}'".format(model_path))
        

    class CustomTrainer(AdapterTrainer):
    #Add a parameter for Criterion
        def __init__(self, model, args, train_dataset, eval_dataset, criterion, compute_metrics):
            super().__init__(model=model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset, compute_metrics=compute_metrics)
            self.criterion = criterion

        def compute_loss(self, model, inputs, return_outputs=False):
            outputs = model(**inputs)
            # Compute custom loss here
            custom_loss = self.criterion(outputs.logits, inputs['labels'])
            return (custom_loss, outputs) if return_outputs else custom_loss
        
    training_args = TrainingArguments(
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        logging_steps=200,
        output_dir="./training_output",
        overwrite_output_dir=True,
        # The next line is important to ensure the dataset labels are properly passed to the model
        remove_unused_columns=False,
    )

    def compute_accuracy(p: EvalPrediction):
        preds = np.argmax(p.predictions, axis=1)
        return {"acc": (preds == p.label_ids).mean()}
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        # Add the criterion to the trainer
        criterion=criterion,
        compute_metrics=compute_accuracy,
    )

    trainer.train()
    trainer.evaluate()

    # Save adapter
    model.save_adapter( args.outdir+folder+"/"+str(args.noise_sd), "denoising-adapter")




        
