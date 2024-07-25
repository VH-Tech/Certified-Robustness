# this file is based on code publicly available at
#   https://github.com/bearpaw/pytorch-classification
# written by Wei Yang.
from accelerate import Accelerator
from architectures import CLASSIFIERS_ARCHITECTURES, get_architecture
from datasets import get_dataset, DATASETS
from loss import FocalLoss
from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from train_utils import AverageMeter, accuracy, init_logfile, log, copy_code
from mixup_utils import ComboLoader, get_combo_loader
import torch.nn.functional as F

from utils import count_parameters_total, count_parameters_trainable
import numpy as np
from transformers import TrainingArguments, EvalPrediction
from adapters import AdapterTrainer
import torch.nn as nn
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
import wandb
import adapters.composition as ac
from adapters import ParBnConfig, SeqBnConfig, SeqBnInvConfig, PrefixTuningConfig, CompacterConfig, LoRAConfig, IA3Config, Fuse
accelerator = Accelerator()

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--noise_sd', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-1)
parser.add_argument('--batch', type=int, default=128)
parser.add_argument('--name', type=str, default="selector-adapter")
parser.add_argument('--adapter_config', type=str, default="compacter")
args = parser.parse_args()


def train(loader: DataLoader, model: torch.nn.Module, criterion, optimizer: Optimizer, epoch: int, noise_sd: float):
    """
    Function to do one training epoch
        :param loader:DataLoader: dataloader (train) 
        :param model:torch.nn.Module: the classifer being trained
        :param criterion: the loss function
        :param optimizer:Optimizer: the optimizer used during trainined
        :param epoch:int: the current epoch number (for logging)
        :param noise_sd:float: the std-dev of the Guassian noise perturbation of the input
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    # top5 = AverageMeter()
    end = time.time()  

    # switch to train mode
    model.train()

    for i, (inputs, targets) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs = inputs.cuda()
        targets = targets.cuda()

        # augment inputs with noise
        # if noise_sd == -1:
        #     #choose randomly a value between 0 and 1
        #     noise_sd = random.random()
        if noise_sd < 0:
            choices = np.array([0.25, 0.5, 0.75, 1.0])
            noise_sd = np.random.choice(choices)

        inputs = inputs + torch.randn_like(inputs, device='cuda') * noise_sd

        # compute output
        outputs = model(inputs)
        if VIT == True :
            outputs = outputs.logits
        
        # print(outputs.shape, targets.shape)
        
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        acc1 = accuracy(outputs, targets, topk=(1,))[0]
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1.item(), inputs.size(0))
        # top5.update(acc5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        # loss.backward()
        accelerator.backward(loss)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))

    return (losses.avg, top1.avg)


def test(loader: DataLoader, model: torch.nn.Module, criterion, noise_sd: float):
    """
    Function to evaluate the trained model
        :param loader:DataLoader: dataloader (train)
        :param model:torch.nn.Module: the classifer being evaluated
        :param criterion: the loss function
        :param noise_sd:float: the std-dev of the Guassian noise perturbation of the input
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    # top5 = AverageMeter()
    end = time.time()

    # switch to eval mode
    model.eval()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            # measure data loading time
            data_time.update(time.time() - end)

            inputs = inputs.cuda()
            targets = targets.cuda()

            # # augment inputs with noise
            # if noise_sd == -1:
            #     #choose randomly a value between 0 and 1
            #     noise_sd = random.random()

            if noise_sd < 0:
                choices = np.array([0.25, 0.5, 0.75, 1.0])
                noise_sd = np.random.choice(choices)
                
            inputs = inputs + torch.randn_like(inputs, device='cuda') * noise_sd

            # compute output
            outputs = model(inputs)
            if VIT == True :
                outputs = outputs.logits
                
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            acc1 = accuracy(outputs, targets, topk=(1,))[0]
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
            # top5.update(acc5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1))

        return (losses.avg, top1.avg)


train_dataset = get_dataset("cifar10", 'train', "/scratch/ravihm.scee.iitmandi/dataset/cifar10", model_input=224)
test_dataset = get_dataset("cifar10", 'test', "/scratch/ravihm.scee.iitmandi/dataset/cifar10", model_input=224)


# Define the desired subset size
subset_size = int(len(train_dataset) * 0.1)

# Create a subset of the CIFAR10 dataset
subset_indices = torch.randperm(len(train_dataset))[:subset_size]
subset_train_dataset = Subset(train_dataset, subset_indices)
pin_memory = ("cifar10" == "imagenet")
    
print("creating train dataloader with dataset of size : ",len(subset_train_dataset))
train_loader = DataLoader(subset_train_dataset, shuffle=True, batch_size=args.batch,
                              num_workers=8, pin_memory=pin_memory)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch,
                             num_workers=8, pin_memory=pin_memory)

model = get_architecture("vit", "cifar10")
_, model = model
adapters.init(model)
model.load_adapter("/scratch/ravihm.scee.iitmandi/models/cifar10/vit/adapters_compacter_0.1/0.25", with_head=False)
model.load_adapter("/scratch/ravihm.scee.iitmandi/models/cifar10/vit/adapters_compacter_0.1/0.5", with_head=False)
model.load_adapter("/scratch/ravihm.scee.iitmandi/models/cifar10/vit/adapters_compacter_0.1/0.75", with_head=False)
model.load_adapter("/scratch/ravihm.scee.iitmandi/models/cifar10/vit/adapters_compacter_0.1/1.0", with_head=False)
model.add_adapter("selection-adapter", config=args.adapter_config)
model.active_adapters = ac.Stack(ac.Parallel('denoising-adapter-75', 'denoising-adapter-25', 'denoising-adapter-50', 'denoising-adapter-100'), "selection-adapter")
model.train_adapter("selection-adapter")

global VIT
VIT = True
criterion = CrossEntropyLoss().cuda()

starting_epoch = 0
logfilename = os.path.join('/scratch/ravihm.scee.iitmandi/models/cifar10/vit/selection_adapter_0.1/log.txt')
os.makedirs('/scratch/ravihm.scee.iitmandi/models/cifar10/vit/selection_adapter_0.1', exist_ok=True)

optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = StepLR(optimizer, step_size=60, gamma=0.1)
model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)
model = model.cuda()

wandb.init(
    # set the wandb project where this run will be logged
    project=args.name,

    # track hyperparameters and run metadata
    config={
    "learning_rate": 1e-1,
    "architecture": "vit",
    "dataset": "cifar10",
    "epochs": 90,
    }
)
print("Training " +  str((count_parameters_trainable(model)/(count_parameters_total(model)))*100)  +"% of the parameters")
print("starting training")
best = 0
for epoch in range(starting_epoch, 180):
    before = time.time()


    train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, args.noise_sd)

    test_loss, test_acc = test(test_loader, model, criterion, args.noise_sd)
    after = time.time()

    log(logfilename, "{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}".format(
            epoch, after - before,
            scheduler.get_lr()[0], train_loss, test_loss, train_acc, test_acc))

    scheduler.step(epoch)

    if test_acc > best and ((epoch > 0 and args.noise_sd > 0) or (epoch > 0 and args.noise_sd < 0)):
        print(f'New Best Found: {test_acc}%')
        best = test_acc
        # if "cifar10" not in ['cifar10']:
        #     normalize_layer, model = model

        # # Save fusion
        # # model.save_adapter_fusion("/scratch/ravihm.scee.iitmandi/models/cifar10/vit/adapters_fusion_0.1/", "denoising-adapter-75,denoising-adapter-25,denoising-adapter-50,denoising-adapter-100")
    
        model.save_adapter('/scratch/ravihm.scee.iitmandi/models/cifar10/vit/selection_adapter_compacter_0.1/', "selection-adapter")

        # if "cifar10" not in ['cifar10']:
        #     model = torch.nn.Sequential(normalize_layer, model)


    wandb.log({"train_loss": train_loss, "test_loss": test_loss, "train_acc": train_acc, "test_acc": test_acc, "best" : best, "lr" : scheduler.get_lr()[0]})

