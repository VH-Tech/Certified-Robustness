# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# File for training denoisers with at most one classifier attached to
from accelerate import Accelerator
from architectures import DENOISERS_ARCHITECTURES, get_architecture, IMAGENET_CLASSIFIERS, CIFAR10_CLASSIFIERS
from datasets import get_dataset, DATASETS
from mixup_utils import get_combo_loader
from loss import *
from test_denoiser import test, test_with_classifier
from torch.nn import MSELoss, CrossEntropyLoss
from torch.optim import SGD, Optimizer, Adam
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from train_utils import AverageMeter, accuracy, init_logfile, log, copy_code, requires_grad_
import wandb
import argparse
import datetime
import numpy as np
import os
import time
import torch
import torchvision
from torch.utils.data import Subset
import torchvision.transforms as transforms

accelerator = Accelerator()
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset', type=str, choices=DATASETS)
parser.add_argument('--arch', type=str, choices=DENOISERS_ARCHITECTURES)
parser.add_argument('--outdir', type=str, help='folder to save denoiser and training log)')
parser.add_argument('--data_dir', type=str, help='folder to load training data from')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--dataset_fraction', default=1.0, type=float, metavar='N',
                    help='dataset fraction')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch', default=30, type=int, metavar='N',
                    help='batchsize (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=1.5e-3, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--lr_step_size', type=int, default=60,
                    help='How often to decrease learning by gamma.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--gpu', default=None, type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--noise_sd', default=0.0, type=float,
                    help="standard deviation of noise distribution for data augmentation")
parser.add_argument('--objective', default='denoising', type=str,
                    help="the objective that is used to train the denoiser",
                    choices=['denoising', 'classification', 'stability', 'focal', 'ldam'])
parser.add_argument('--classifier', default='', type=str,
                    help='path to the classifier used with the `classificaiton`'
                     'or `stability` objectives of the denoiser.')
parser.add_argument('--pretrained-denoiser', default='', type=str,
                    help='path to a pretrained denoiser')
parser.add_argument('--optimizer', default='Adam', type=str,
                    help='SGD, Adam, or Adam then SGD', choices=['SGD', 'Adam','AdamThenSGD'])
parser.add_argument('--start-sgd-epoch', default=50, type=int,
                    help='[Relevent only to AdamThenSGD.] Epoch at which adam switches to SGD')
parser.add_argument('--start-sgd-lr', default=1e-3, type=float,
                    help='[Relevent only to AdamThenSGD.] LR at which SGD starts after Adam')
parser.add_argument('--resume', action='store_true',
                    help='if true, tries to resume training from an existing checkpoint')
parser.add_argument('--azure_datastore_path', type=str, default='',
                    help='Path to imagenet on azure')
parser.add_argument('--philly_imagenet_path', type=str, default='',
                    help='Path to imagenet on philly')
parser.add_argument('--drw', type=int, default=0,
                    help='drw for ldam')

parser.add_argument('--mixup_lam', type=float, default=0.1, 
                    help='mixup lambda')
parser.add_argument('--mixup_mode', type=str, default='class', 
                    help='sampling mode (instance, class, sqrt, prog)')
parser.add_argument('--mixup', type=int, default=0, 
                    help='do mixup')
parser.add_argument('--ssl_like', type=int, default=0, 
                    help='do ssl like criterion')
parser.add_argument('--both', type=int, default=0, 
                    help='do both')
args = parser.parse_args()

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

toPilImage = ToPILImage()

def main():

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir, exist_ok=True)

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

    if args.pretrained_denoiser:
        checkpoint = torch.load(args.pretrained_denoiser)
        assert checkpoint['arch'] == args.arch
        denoiser = get_architecture(checkpoint['arch'], args.dataset)
        denoiser.load_state_dict(checkpoint['state_dict'])
    else:
        denoiser = get_architecture(args.arch, args.dataset)

    if args.optimizer == 'Adam':
        optimizer = Adam(denoiser.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = SGD(denoiser.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamThenSGD':
        optimizer = Adam(denoiser.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)

    starting_epoch = 0
    logfilename = os.path.join(args.outdir, 'log.txt')

    ## Resume from checkpoint if exists and if resume flag is True
    denoiser_path = os.path.join(args.outdir, 'checkpoint.pth.tar')
    if args.resume and os.path.isfile(denoiser_path):
        print("=> loading checkpoint '{}'".format(denoiser_path))
        checkpoint = torch.load(denoiser_path,
                                map_location=lambda storage, loc: storage)
        assert checkpoint['arch'] == args.arch
        # starting_epoch = checkpoint['epoch']
        # new_state_dict = {}
        # for k, v in checkpoint['state_dict'].items():
        #     name = k[7:] # remove `module.`
        #     new_state_dict[name] = v
        # denoiser.load_state_dict(new_state_dict)
        denoiser.load_state_dict(checkpoint['state_dict'])
        if starting_epoch >= args.start_sgd_epoch and args.optimizer == 'AdamThenSGD ': # Do adam for few steps thaen continue SGD
            print("-->[Switching from Adam to SGD.]")
            args.lr = args.start_sgd_lr
            optimizer = SGD(denoiser.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)
        
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
                        .format(denoiser_path, checkpoint['epoch']))
    else:
        if args.resume: print("=> no checkpoint found at '{}'".format(args.outdir))
        init_logfile(logfilename, "epoch\ttime\tlr\ttrainloss\ttestloss\ttestAcc")


    if args.objective == 'denoising':
        criterion = MSELoss(size_average=None, reduce=None, reduction = 'mean').cuda()
        best_loss = 1e6

    elif args.objective in ['classification', 'stability', 'focal', 'ldam']:
        assert args.classifier != '', "Please specify a path to the classifier you want to attach the denoiser to."
        if args.classifier == 'resnet50':
            clf = get_architecture(args.classifier, args.dataset, pytorch_pretrained=True)

        elif args.classifier in IMAGENET_CLASSIFIERS and args.dataset == 'imagenet':
            clf = get_architecture(args.classifier, args.dataset, pytorch_pretrained=True)

        elif args.classifier in CIFAR10_CLASSIFIERS and args.dataset == 'cifar10':
            clf = get_architecture(args.classifier, args.dataset, pytorch_pretrained=True)
            norm_layer , clf = clf
            print("removed norm layer and loaded cifar10 ViT")

        else:
            checkpoint = torch.load(args.classifier)
            clf = get_architecture(checkpoint['arch'], args.dataset)
            _,clf=clf
            clf.load_state_dict(checkpoint['state_dict'])


        clf.cuda().eval()
        requires_grad_(clf, False)
        print("frozen clf")

        if args.objective == 'focal':
            class_frequency = torch.unique(torch.tensor(train_dataset.targets), return_counts=True)[1] if args.dataset == 'pneumonia' else train_dataset.get_frequency()
            w = torch.tensor([sum(class_frequency) / (len(class_frequency) * class_frequency[i]) for i in range(len(class_frequency))])
            print('Class Frequency and weights for train dataset: ', class_frequency, w)
            criterion = FocalLoss(alpha=w.cuda(), gamma=2).cuda() 
        elif args.objective == 'ldam':
            class_frequency = list(torch.unique(torch.tensor(train_dataset.targets), return_counts=True)[1]) if args.dataset == 'pneumonia' else train_dataset.get_frequency()
            # w = get_CB_weights(class_frequency).float().cuda()
            w = torch.tensor([sum(class_frequency) / (len(class_frequency) * class_frequency[i]) for i in range(len(class_frequency))])
            print('Class Frequency and weights for train dataset: ', class_frequency, w)
            criterion = LDAMLoss(cls_num_list=class_frequency, weight= None if args.drw else w.cuda()).cuda() 
        else:
            criterion = CrossEntropyLoss().cuda()
        best_acc = 0
    denoiser, optimizer, train_loader, scheduler = accelerator.prepare(denoiser, optimizer, train_loader, scheduler)
    wandb.login()
    run = wandb.init(
    # Set the project where this run will be logged
    project="denoiser_"+args.dataset+'_'+str(args.dataset_fraction)+"_"+str(args.noise_sd),
    # Track hyperparameters and run metadata
    config={
        "learning_rate": args.lr,
        "epochs": args.epochs,
    },
    )

    for epoch in range(starting_epoch, args.epochs):
        before = time.time()
        if args.objective == 'denoising':
            train_loss = train(train_loader, denoiser, criterion, optimizer, epoch, args.noise_sd)
            test_loss = test(test_loader, denoiser, criterion, args.noise_sd, args.print_freq, args.outdir)
            test_acc = 'NA'
        elif args.objective in ['classification', 'stability', 'focal', 'ldam']:
            if args.mixup:
                train_loss = train_mixup(train_loader, denoiser, criterion, optimizer, epoch, args.noise_sd, clf, mode=args.mixup_mode, mixup_lam=args.mixup_lam)
            else:
                train_loss = train(train_loader, denoiser, criterion, optimizer, epoch, args.noise_sd, clf)
            
            test_loss, test_acc = test_with_classifier(test_loader, denoiser, criterion, args.noise_sd, args.print_freq, clf)

        after = time.time()
        wandb.log({"train_loss": train_loss, "test_loss": test_loss, "test_acc": test_acc})
        log(logfilename, "{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}".format(
            epoch, after - before,
            args.lr, train_loss, test_loss, test_acc))

        scheduler.step(epoch)
        args.lr = scheduler.get_lr()[0]

        # Switch from Adam to SGD
        if epoch == args.start_sgd_epoch and args.optimizer == 'AdamThenSGD ': # Do adam for few steps thaen continue SGD
            print("-->[Switching from Adam to SGD.]")
            args.lr = args.start_sgd_lr
            optimizer = SGD(denoiser.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)

        torch.save({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': denoiser.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(args.outdir, 'checkpoint.pth.tar'))

        if args.objective == 'denoising' and test_loss < best_loss:
            best_loss = test_loss
        elif args.objective in ['classification', 'stability', 'focal', 'ldam'] and test_acc > best_acc:
            best_acc = test_acc

def train(loader: DataLoader, denoiser: torch.nn.Module, criterion, optimizer: Optimizer, epoch: int, noise_sd: float, classifier: torch.nn.Module=None):
    """
    Function for training denoiser for one epoch
        :param loader:DataLoader: training dataloader
        :param denoiser:torch.nn.Module: the denoiser being trained
        :param criterion: loss function
        :param optimizer:Optimizer: optimizer used during trainined
        :param epoch:int: the current epoch (for logging)
        :param noise_sd:float: the std-dev of the Guassian noise perturbation of the input
        :param classifier:torch.nn.Module=None: a ``freezed'' classifier attached to the denoiser 
                                                (required classifciation/stability objectives), None for denoising objective 
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    # switch to train mode
    denoiser.train()
    if classifier:
        classifier.eval()

    # Define the resize transform
    resize_transform = transforms.Resize((224, 224))

    for i, (inputs, targets) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs = inputs.cuda()
        targets = targets.cuda()

        # augment inputs with noise
        noise = torch.randn_like(inputs, device='cuda') * noise_sd

        # compute output
        inputs = inputs + noise
        inputs = resize_transform(inputs)
        
        outputs = denoiser(inputs)
        if classifier:
            outputs = classifier(outputs)
            outputs = outputs.logits
        
        if isinstance(criterion, MSELoss):
            loss = criterion(outputs, inputs)
        else:
            if args.objective in ['classification', 'stability', 'focal', 'ldam'] and args.ssl_like == 1:
                with torch.no_grad():
                    targets = classifier(inputs)
                    targets = targets.argmax(1).detach().clone()
            loss = criterion(outputs, targets)

        # record loss
        losses.update(loss.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        # loss.backward()
        accelerator.backward(loss)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                epoch, i, len(loader), batch_time=batch_time,
                data_time=data_time, loss=losses))

    return losses.avg

def train_mixup(loader: DataLoader, denoiser: torch.nn.Module, criterion, optimizer: Optimizer, epoch: int, noise_sd: float, classifier: torch.nn.Module=None, mode='instance', mixup_lam=0.1):
    """
    Function for training denoiser for one epoch
        :param loader:DataLoader: training dataloader
        :param denoiser:torch.nn.Module: the denoiser being trained
        :param criterion: loss function
        :param optimizer:Optimizer: optimizer used during trainined
        :param epoch:int: the current epoch (for logging)
        :param noise_sd:float: the std-dev of the Guassian noise perturbation of the input
        :param classifier:torch.nn.Module=None: a ``freezed'' classifier attached to the denoiser 
                                                (required classifciation/stability objectives), None for denoising objective 
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    loader = get_combo_loader(loader, mode=mode)

    # switch to train mode
    denoiser.train()
    if classifier:
        classifier.eval()

    for i, batch in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        lam = np.random.beta(a=mixup_lam, b=1)

        inputs, targets = batch[0][0], batch[0][1]
        balanced_inputs, balanced_targets = batch[1][0], batch[1][1]

        inputs, targets = inputs.cuda(), targets.squeeze().cuda()
        balanced_inputs, balanced_targets = balanced_inputs.cuda(), balanced_targets.squeeze().cuda()

        inputs_mixup = (1 - lam) * inputs + lam * balanced_inputs
        targets_mixup = (1 - lam) * F.one_hot(targets, 23) + lam * F.one_hot(balanced_targets, 23)

        # augment inputs with noise
        noise = torch.randn_like(inputs_mixup, device='cuda') * noise_sd

        # compute output
        outputs_mixup = denoiser(inputs_mixup + noise)
        if classifier:
            outputs_mixup = classifier(outputs_mixup)
        
        if isinstance(criterion, MSELoss):
            loss = criterion(outputs_mixup, inputs_mixup)
        else:
            if args.objective in ['classification', 'stability', 'focal', 'ldam'] and args.ssl_like == 1:
                with torch.no_grad():
                    targets_mixup = classifier(inputs_mixup)
                    targets_mixup = targets_mixup.argmax(1).detach().clone()
                
            loss = criterion(outputs_mixup, targets_mixup)

        if args.both:
            inputs = torch.cat((inputs, balanced_inputs), 0)
            targets = torch.cat((targets, balanced_targets), 0)

            noise = torch.randn_like(inputs, device='cuda') * noise_sd

            # compute output
            outputs = denoiser(inputs + noise)
            if classifier:
                outputs = classifier(outputs)
            
            if isinstance(criterion, MSELoss):
                loss = criterion(outputs, inputs)
            else:
                if args.objective in ['classification', 'stability', 'focal', 'ldam'] and args.ssl_like == 1:
                    with torch.no_grad():
                        targets = classifier(inputs)
                        targets = targets.argmax(1).detach().clone()
                    
                loss += criterion(outputs, targets)


        # record loss
        losses.update(loss.item(), inputs.size(0))

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
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                epoch, i, len(loader), batch_time=batch_time,
                data_time=data_time, loss=losses))

    return losses.avg


if __name__ == "__main__":
    main()
