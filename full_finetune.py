# this file is based on code publicly available at
#   https://github.com/bearpaw/pytorch-classification
# written by Wei Yang.
from utils import count_parameters_trainable, count_parameters_total
from accelerate import Accelerator
from architectures import CLASSIFIERS_ARCHITECTURES, get_architecture
from datasets import get_dataset, DATASETS
from loss import FocalLoss
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from train_utils import AverageMeter, accuracy, init_logfile, log, copy_code, cosine_scheduler
#import Subset
from torch.utils.data import Subset
import json
import wandb
import argparse
import datetime
import numpy as np
import os
import time
import torch
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset', type=str, 
                    choices=DATASETS)
parser.add_argument('--arch', type=str, default="google/vit-base-patch16-224-in21k",
                    choices=CLASSIFIERS_ARCHITECTURES)
parser.add_argument('--outdir', type=str, 
                    help='folder to save model and training log)')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=60, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch', default=64, type=int, metavar='N',
                    help='batchsize (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--lr_step_size', type=int, default=40,
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
parser.add_argument('--tuning_method', type=str, default='full',
                    help='Path to finetune model')
parser.add_argument("--dataset_fraction", type=float, default=1.0,
                    help="Fraction of dataset to use")
args = parser.parse_args()

accelerator = Accelerator()

if args.azure_datastore_path:
    os.environ['IMAGENET_DIR_AZURE'] = os.path.join(args.azure_datastore_path, 'datasets/imagenet_zipped')
if args.philly_imagenet_path:
    os.environ['IMAGENET_DIR_PHILLY'] = os.path.join(args.philly_imagenet_path, './')

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
VIT = False

def main():
    if args.gpu:
        # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        pass
    
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # Copy code to output directory
    # copy_code(args.outdir)
    if "vit" in args.arch :
        global VIT
        VIT = True
    if "vit" in args.arch or "swin" in args.arch:
        train_dataset = get_dataset(args.dataset, 'train', args.data_dir)
        test_dataset = get_dataset(args.dataset, 'test', args.data_dir)
    else:
        train_dataset = get_dataset(args.dataset, 'train', args.data_dir)
        test_dataset = get_dataset(args.dataset, 'test', args.data_dir)

    subset_size = int(len(train_dataset) * args.dataset_fraction)

    # Create a subset of the CIFAR10 dataset
    subset_indices = torch.randperm(len(train_dataset))[:subset_size]
    train_dataset = Subset(train_dataset, subset_indices)
    print("train dataset length : ", len(train_dataset))


    pin_memory = (args.dataset == "imagenet")
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch,
                              num_workers=args.workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch,
                             num_workers=args.workers, pin_memory=pin_memory)

    model = get_architecture(args.arch, args.dataset, tuning_method=args.tuning_method)
    _, model = model # removed normalization layer


    if args.focal:

        class_frequency = torch.unique(torch.tensor(train_dataset.targets), return_counts=True)[1] if args.dataset == 'pneumonia' else train_dataset.get_frequency()
        w = torch.tensor([sum(class_frequency) / (len(class_frequency) * class_frequency[i]) for i in range(len(class_frequency))])
        print('Class Frequency and weights for train dataset: ', class_frequency, w)
        criterion = FocalLoss(alpha=w.cuda(), gamma=2).cuda() 
    else:
        criterion = CrossEntropyLoss().cuda()


    starting_epoch = 0
    logfilename = os.path.join(args.outdir, 'log.txt')

    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)
    # scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr/1000)

    ## Resume from checkpoint if exists and if resume flag is True
    model_path = os.path.join(args.outdir, 'checkpoint.pth.tar')
    if args.tuning_method == 'full':
        args.outdir = os.path.join(args.outdir, str(args.dataset_fraction), 'full_' + str(args.noise_sd))
        os.makedirs(args.outdir, exist_ok=True)

    if args.resume and os.path.isfile(model_path):
        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path,
                                map_location=lambda storage, loc: storage)
        # starting_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
                        .format(model_path, checkpoint['epoch']))
        
    else:
        if args.resume: print("=> no checkpoint found at '{}'".format(args.outdir))
        os.makedirs(args.outdir, exist_ok=True)
        init_logfile(logfilename, "epoch\ttime\tlr\ttrainloss\ttestloss\ttrainAcc\ttestAcc")

    if os.path.isfile(model_path):
        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path,map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print("=> loaded to finetune '{}' (epoch {})"
                        .format(model_path, checkpoint['epoch']))
                # freeze parameters if needed
        tuning_method = args.tuning_method
        if tuning_method == 'full':
            # all parameters are trainable
            pass 
        elif tuning_method == 'prompt':
            for name, param in model.named_parameters():
                if name.startswith('head'):
                    continue

                if name.startswith('norm'):
                    continue

                if 'tuning_module' in name:
                    continue

                param.requires_grad = False
        elif tuning_method == 'adapter':
            raise NotImplementedError
        elif tuning_method == 'sidetune':
            raise NotImplementedError
        elif tuning_method == 'linear':
            for name, param in model.named_parameters():
                if name.startswith('head'):
                    continue
                
                if name.startswith('norm'):
                    continue

                param.requires_grad = False
        elif tuning_method == 'norm':
            for name, param in model.named_parameters():
                if name.startswith('head'):
                    continue

                if 'bn' in name:
                    continue

                if 'gn' in name:
                    continue

                if 'norm' in name:
                    continue
                
                # adjust last group norm
                if 'before_head' in name:
                    continue

                param.requires_grad = False    
        elif tuning_method == 'bias':
            for name, param in model.named_parameters():
                if name.startswith('head'):
                    continue
                
                if name.startswith('norm'):
                    continue

                if 'bias' in name:
                    continue

                param.requires_grad = False
        elif tuning_method == 'conv_adapt' or tuning_method == 'repnet':
            for name, param in model.named_parameters():
                if name.startswith('head'):
                    continue
                
                if 'tuning_module' in name:
                    continue
                
                # add a norm layer before average pooling
                if 'norm' in name:
                    continue

                param.requires_grad = False
        elif tuning_method == 'conv_adapt_norm':
            for name, param in model.named_parameters():
                if name.startswith('head'):
                    continue
                
                if 'tuning_module' in name:
                    continue

                if 'bn' in name:
                    continue

                if 'gn' in name:
                    continue

                if 'norm' in name:
                    continue
                
                # adjust last group norm
                if 'before_head' in name:
                    continue

                param.requires_grad = False    
        elif tuning_method == 'conv_adapt_bias' or tuning_method == 'repnet_bias':
            for name, param in model.named_parameters():
                if name.startswith('head'):
                    continue
                
                if 'tuning_module' in name:
                    continue

                if 'bias' in name:
                    continue
                
                # add a norm layer before average pooling
                if name.startswith('norm'):
                    continue

                param.requires_grad = False
        elif tuning_method == 'compacter':
            for name, param in model.named_parameters():
                if name.startswith('head'):
                    continue
                
                if 'tuning_module' in name:
                    continue
                param.requires_grad = False

        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"{name} is trainable")

    best = 0.0 
    if args.tuning_method != 'full':
        args.outdir = os.path.join(args.outdir, args.tuning_method, str(args.noise_sd), str(args.dataset_fraction))
        os.makedirs(args.outdir, exist_ok=True)
    
    print("training ", count_parameters_trainable(model)/count_parameters_total(model), " of parameters")

    wandb.init(
    # set the wandb project where this run will be logged
    project=args.arch+"_"+args.dataset+"_"+args.tuning_method+"_"+str(args.dataset_fraction)+"_"+str(args.noise_sd),

    # track hyperparameters and run metadata
    config={
    "learning_rate": args.lr,
    "architecture": args.arch,
    "dataset": args.dataset,
    "epochs": args.epochs,
    }
    )
    model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)
    model = model.cuda()
    for epoch in range(starting_epoch, args.epochs):
        before = time.time()
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, args.noise_sd)
        test_loss, test_acc = test(test_loader, model, criterion, args.noise_sd)
        after = time.time()

        log(logfilename, "{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}".format(
            epoch, after - before,
            scheduler.get_lr()[0], train_loss, test_loss, train_acc, test_acc))
        scheduler.step(epoch)
       
        if test_acc > best:
            print(f'New Best Found: {test_acc}%')
            best = test_acc
            torch.save({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(args.outdir, 'checkpoint.pth.tar'))

        wandb.log({"train_loss": train_loss, "test_loss": test_loss, "train_acc": train_acc, "test_acc": test_acc, "best" : best, "lr" : scheduler.get_lr()[0]})
        

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

    resize_transform = transforms.Resize((224, 224))

    for i, (inputs, targets) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs = inputs.cuda()
        targets = targets.cuda()

        # augment inputs with noise
        inputs = inputs + torch.randn_like(inputs, device='cuda') * noise_sd
        inputs = resize_transform(inputs)

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
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
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

    resize_transform = transforms.Resize((224, 224))
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            # measure data loading time
            data_time.update(time.time() - end)

            inputs = inputs.cuda()
            targets = targets.cuda()

            # augment inputs with noise
            inputs = inputs + torch.randn_like(inputs, device='cuda') * noise_sd
            inputs = resize_transform(inputs)
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
