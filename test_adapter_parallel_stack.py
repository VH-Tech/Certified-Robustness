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

import numpy as np

import time
import torch

import random
import adapters.composition as ac

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--noise_sd', type=float, default=1)
parser.add_argument('--lr', type=float, default=1e-1)
parser.add_argument('--batch', type=int, default=128)
parser.add_argument('--name', type=str, default="selector-adapter")
parser.add_argument('--adapter_config', type=str, default="compacter")
args = parser.parse_args()

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


test_dataset = get_dataset("cifar10", 'test', "/scratch/ravihm.scee.iitmandi/dataset/cifar10", model_input=224)


pin_memory = ("cifar10" == "imagenet")

test_loader = DataLoader(test_dataset, shuffle=False, batch_size=256,
                             num_workers=8, pin_memory=pin_memory)

model = get_architecture("vit", "cifar10")
_, model = model
adapters.init(model)
model.load_adapter("/scratch/ravihm.scee.iitmandi/models/cifar10/vit/adapters_compacter_range_0.1/0.25", with_head=False)
model.load_adapter("/scratch/ravihm.scee.iitmandi/models/cifar10/vit/adapters_compacter_range_0.1/0.5", with_head=False)
model.load_adapter("/scratch/ravihm.scee.iitmandi/models/cifar10/vit/adapters_compacter_range_0.1/0.75", with_head=False)
model.load_adapter("/scratch/ravihm.scee.iitmandi/models/cifar10/vit/adapters_compacter_range_0.1/1.0", with_head=False)
model.add_adapter("selection-adapter", config=args.adapter_config)
model.active_adapters = ac.Stack(ac.Parallel('denoising-adapter-75', 'denoising-adapter-25', 'denoising-adapter-50', 'denoising-adapter-100'), "selection-adapter")
model.train_adapter("selection-adapter")


checkpoint = torch.load('/scratch/ravihm.scee.iitmandi/models/cifar10/vit/selection_adapter_compacter_0.1/checkpoint.pth.tar', map_location=lambda storage, loc: storage)
new_state_dict = {}
for k, v in checkpoint["state_dict"].items():
    if k.startswith('module.'):
        new_key = k.replace('module.', '')
        new_state_dict[new_key] = v
    else:
        new_state_dict[k] = v

model.load_state_dict(new_state_dict)
print("loaded everything")

global VIT
VIT = True
criterion = CrossEntropyLoss().cuda()

starting_epoch = 0
model = model.cuda()


# print("Training " +  str((count_parameters_trainable(model)/(count_parameters_total(model)))*100)  +"% of the parameters")
print("starting testing")

test_loss, test_acc = test(test_loader, model, criterion, 0.25)
print(test_loss, test_acc)
test_loss, test_acc = test(test_loader, model, criterion, 0.5)
print(test_loss, test_acc)
test_loss, test_acc = test(test_loader, model, criterion, 0.75)
print(test_loss, test_acc)
test_loss, test_acc = test(test_loader, model, criterion, 1.0)
print(test_loss, test_acc)
test_loss, test_acc = test(test_loader, model, criterion, 1.25)
print(test_loss, test_acc)
test_loss, test_acc = test(test_loader, model, criterion, 1.5)
print(test_loss, test_acc)
test_loss, test_acc = test(test_loader, model, criterion, 1.75)
print(test_loss, test_acc)
test_loss, test_acc = test(test_loader, model, criterion, 2.0)
print(test_loss, test_acc)

