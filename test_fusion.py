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
from adapters import ParBnConfig, SeqBnConfig, SeqBnInvConfig, PrefixTuningConfig, CompacterConfig, LoRAConfig, IA3Config, Fuse
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

parser.add_argument('--dataset_fraction', type=float, default='1.0',
                    help='Path to data directory')
# adapters
parser.add_argument('--adapter_config', type=str, default=None,
                    help='Adapter name')

parser.add_argument('--mixup', type=str, default=None,
                    help='Adapter name')

args = parser.parse_args()

if args.azure_datastore_path:
    os.environ['IMAGENET_DIR_AZURE'] = os.path.join(args.azure_datastore_path, 'datasets/imagenet_zipped')
if args.philly_imagenet_path:
    os.environ['IMAGENET_DIR_PHILLY'] = os.path.join(args.philly_imagenet_path, './')

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
global VIT
VIT = False

def main():
    if args.gpu:
        # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        pass
    
    # Copy code to output directory
    # copy_code(args.outdir)

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
            checkpoint = torch.load(model_path,
                                    map_location=lambda storage, loc: storage)

            model.load_state_dict(checkpoint['state_dict'])

            print("=> loaded checkpoint '{}' (epoch {})"
                            .format(model_path, checkpoint['epoch']))

        else:
            print("=> no checkpoint found at '{}', please check the path provided".format(args.outdir))
            return
    
    if args.adapter_config is not None:
        folder = "/adapters"

    if args.mixup:
        folder += "_mixup"

    if args.focal :
        folder += "_focal"
    if args.adapter_config == "fusion":
        print("model created")
        adapters.init(model)
        model.load_adapter("/scratch/ravihm.scee.iitmandi/models/cifar10/vit/adapters_compacter_range_0.1/0.25", with_head=False)
        model.load_adapter("/scratch/ravihm.scee.iitmandi/models/cifar10/vit/adapters_compacter_range_0.1/0.5", with_head=False)
        model.load_adapter("/scratch/ravihm.scee.iitmandi/models/cifar10/vit/adapters_compacter_range_0.1/0.75", with_head=False)
        model.load_adapter("/scratch/ravihm.scee.iitmandi/models/cifar10/vit/adapters_compacter_range_0.1/1.0", with_head=False)

        # Add a fusion layer for all loaded adapters
        adapter_setup = Fuse('denoising-adapter-25', 'denoising-adapter-50', 'denoising-adapter-75', 'denoising-adapter-100')
        model.add_adapter_fusion(adapter_setup)

        # Unfreeze and activate fusion setup
        model.train_adapter_fusion(adapter_setup)
        
        checkpoint = torch.load(os.path.join('/scratch/ravihm.scee.iitmandi/models/cifar10/vit/adapters_fusion_range_0.1/', 'checkpoint.pth.tar'), map_location=lambda storage, loc: storage)

        # Create a new state dict with modified keys
        new_state_dict = {}
        for k, v in checkpoint["state_dict"].items():
            if k.startswith('module.'):
                new_key = k.replace('module.', '')
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict)
        print("loaded everything")

    elif args.adapter_config:
        folder += "_"+args.adapter_config
        folder += "_"+str(args.dataset_fraction)
        adapter_path = args.outdir+folder+"/"+str(args.noise_sd)
        
        adapters.init(model)
        model.load_adapter(adapter_path)
        model.set_active_adapters("denoising-adapter-"+str(int(args.noise_sd*100)))
    # normalize_layer, model = model
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
