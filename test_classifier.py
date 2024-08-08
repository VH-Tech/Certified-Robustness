# this file is based on code publicly available at
#   https://github.com/bearpaw/pytorch-classification
# written by Wei Yang.


from architectures import CLASSIFIERS_ARCHITECTURES, get_architecture
from datasets import get_dataset, DATASETS
from loss import FocalLoss
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from train_utils import AverageMeter, accuracy, init_logfile, log, copy_code

import argparse
import datetime
import numpy as np
import os
import time
import torch
import random
from adapters import ParBnConfig, SeqBnConfig, SeqBnInvConfig, PrefixTuningConfig, CompacterConfig, LoRAConfig, IA3Config
import adapters

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset', type=str, 
                    choices=DATASETS)
parser.add_argument('--arch', type=str, default="google/vit-base-patch16-224-in21k",
                    choices=CLASSIFIERS_ARCHITECTURES)
parser.add_argument('--outdir', type=str, 
                    help='folder to save model and training log)')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('--batch', default=64, type=int, metavar='N',
                    help='batchsize (default: 256)')

parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--noise_sd', default=0.0, type=float,
                    help="standard deviation of noise distribution for data augmentation")
parser.add_argument('--resume', action='store_true',
                    help='if true, tries to resume training from an existing checkpoint')

parser.add_argument('--focal', default=0, type=int,
                    help='use focal loss')
parser.add_argument('--data_dir', type=str, default='./data',
                    help='Path to data directory')

parser.add_argument('--dataset_fraction', type=float, default='1.0',
                    help='Path to data directory')
# adapters
parser.add_argument('--adapter_config', type=str, default=None,
                    help='Adapter name')

parser.add_argument('--mixup', type=str, default=None,
                    help='Adapter name')

args = parser.parse_args()

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
global VIT
VIT = False

def main():

    if "vit" in args.arch or "swin" in args.arch:
        global VIT
        VIT = True
        train_dataset = get_dataset(args.dataset, 'train', args.data_dir, model_input=224)
        test_dataset = get_dataset(args.dataset, 'test', args.data_dir, model_input=224)

    else:
        train_dataset = get_dataset(args.dataset, 'train', args.data_dir)
        test_dataset = get_dataset(args.dataset, 'test', args.data_dir)

    pin_memory = (args.dataset == "imagenet")

    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch,
                             num_workers=args.workers, pin_memory=pin_memory)

    model = get_architecture(args.arch, args.dataset)
    normalize_layer, model = model
    print("removed normalization layer")
    if args.focal:

        class_frequency = torch.unique(torch.tensor(train_dataset.targets), return_counts=True)[1] if args.dataset == 'pneumonia' else train_dataset.get_frequency()
        w = torch.tensor([sum(class_frequency) / (len(class_frequency) * class_frequency[i]) for i in range(len(class_frequency))])
        print('Class Frequency and weights for train dataset: ', class_frequency, w)
        criterion = FocalLoss(alpha=w.cuda(), gamma=2).cuda() 
    else:
        criterion = CrossEntropyLoss().cuda()

    ## Load Weights if required
    if args.dataset not in ["imagenet","cifar10"]:

        model_path = os.path.join(args.outdir, 'checkpoint.pth.tar')
        if os.path.isfile(model_path):
            print("=> loading checkpoint '{}'".format(model_path))
            checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
            new_state_dict = {}
            for key, value in checkpoint["state_dict"].items():
                if key.startswith("1.module."):
                    new_key = key[9:]  # Remove the "1.module." prefix
                else:
                    new_key = key
                new_state_dict[new_key] = value
            model.load_state_dict(new_state_dict)
            print("=> loaded checkpoint '{}' (epoch {})".format(model_path, checkpoint['epoch']))

        else:
            print("=> no checkpoint found at '{}', please check the path provided".format(args.outdir))
            return
    
    if args.adapter_config is not None:
        folder = "/adapters"

    if args.mixup:
        folder += "_mixup"

    if args.focal :
        folder += "_focal"

    if args.adapter_config:
        folder += "_"+args.adapter_config
        folder += "_"+str(args.dataset_fraction)
        adapter_path = args.outdir+folder+"/"+str(args.noise_sd)
        
        adapters.init(model)
        model.load_adapter(adapter_path)
        model.set_active_adapters("denoising-adapter-"+str(int(args.noise_sd*100)))

    # model = torch.nn.Sequential(normalize_layer, model)
    model.to('cuda')
    test_loss, test_acc = test(test_loader, model, criterion, args.noise_sd)
    print(test_loss, test_acc)




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

            if noise_sd == -1:
                #choose randomly a value between 0 and 1
                noise_sd = random.random()

            # augment inputs with noise
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

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1))

        return (losses.avg, top1.avg)


if __name__ == "__main__":
    main()
