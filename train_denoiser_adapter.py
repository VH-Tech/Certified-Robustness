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
import torchvision.transforms as transforms


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
from adapters import ParBnConfig, SeqBnConfig, SeqBnInvConfig, PrefixTuningConfig, CompacterConfig, LoRAConfig, IA3Config

accelerator = Accelerator()
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset', type=str, default="hyper",
                    choices=DATASETS)
parser.add_argument('--arch', type=str, default="vit",
                    choices=CLASSIFIERS_ARCHITECTURES)
parser.add_argument('--outdir', type=str, default="/storage/vatsal/models/cifar10",
                    help='folder to save model and training log)')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch', default=256, type=int, metavar='N',
                    help='batchsize (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate', dest='lr')

parser.add_argument('--scheduler', default='cosine', type=str, choices=['cosine', 'step'])

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
parser.add_argument('--data_dir', type=str, default='/storage/vatsal/datasets/hyper',
                    help='Path to data directory')
parser.add_argument("--adapter_config", type=str, help="Adapter config", default="compacter")
parser.add_argument("--dataset_fraction", type=float, default=1.0, help="Fraction of dataset to use")
parser.add_argument('--mixup_lam', type=float, default=0.1, 
                    help='mixup lambda')
parser.add_argument('--mixup_mode', type=str, default='class', 
                    help='sampling mode (instance, class, sqrt, prog)')
parser.add_argument('--mixup', type=int, default=0, 
                    help='do mixup')
parser.add_argument('--ssl_like', type=int, default=0, 
                    help='do ssl like criterion')
parser.add_argument('--do_norm', type=int, default=1, 
                    help='do ssl like criterion')


parser.add_argument('--do_fourier', type=int, default=0, 
                    help='do fourier transform before vit layer')
parser.add_argument('--gap', type=int, default=1)
parser.add_argument('--fourier_location', type=str, default='attention', 
                    help='location', choices=['attention', 'adapter'])
parser.add_argument('--invert_domain', type=int, default=0, 
                    help='invert domain after output')
parser.add_argument('--train_range', type=int, default=0, 
                    help='invert domain after output')

args = parser.parse_args()

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

global VIT
VIT = True

# Step 1: Define the Fourier Transform Layer
class FourierTransformLayer(nn.Module):
    def __init__(self):
        super(FourierTransformLayer, self).__init__()

    def forward(self, hidden_states):
        return torch.fft.fft(torch.fft.fft(hidden_states.float(), dim=-1), dim=-2).real
class InverseFourierTransformLayer(nn.Module):
    def __init__(self):
        super(InverseFourierTransformLayer, self).__init__()

    def forward(self, hidden_states, *args, **kwargs):
        # print(hidden_states)
        return (torch.fft.ifft(torch.fft.ifft(hidden_states[0], dim=-1), dim=-2).real, )
# Step 2: Create the Wrapper Class
class ViTLayerWithFourierTransform(nn.Module):
    def __init__(self, original_layer, do_invert=False):
        super(ViTLayerWithFourierTransform, self).__init__()
        self.fourier_transform = FourierTransformLayer()
        self.original_layer = original_layer
        self.inverse_fourier_transform = InverseFourierTransformLayer()
        self.do_invert = do_invert

    def forward(self, hidden_states, *args, **kwargs):
        # Apply the Fourier Transform before the original layer
        hidden_states = self.fourier_transform(hidden_states)
        # Apply the original layer
        hidden_states = self.original_layer(hidden_states)
        if self.do_invert:
            hidden_states = self.inverse_fourier_transform(hidden_states)
        return hidden_states

class AdapterWithFourierTransform(nn.Module):
    def __init__(self, original_layer, do_invert=False):
        super(AdapterWithFourierTransform, self).__init__()
        self.fourier_transform = FourierTransformLayer()
        self.original_layer = original_layer
        self.do_invert = do_invert
        self.inverse_fourier_transform = InverseFourierTransformLayer()

    def bottleneck_layer_forward(self, hidden_states, *args, **kwargs):
        # Apply the Fourier Transform before the original layer
        hidden_states = self.fourier_transform(hidden_states)
        # Apply the original layer
        hidden_states = self.original_layer(hidden_states, *args, **kwargs)
        if self.do_invert:
            hidden_states = self.inverse_fourier_transform(hidden_states)
        return hidden_states
    
    def forward(self, hidden_states, *args, **kwargs):
        # Apply the Fourier Transform before the original layer
        # hidden_states = self.fourier_transform(hidden_states)
        # # Apply the original layer
        # hidden_states = self.original_layer(hidden_states, *args, **kwargs)
        # if self.do_invert:
        #     hidden_states = self.inverse_fourier_transform(hidden_states)
        return hidden_states
    
def add_fourier_transform_to_adapters(model, gap=1):
    for i in range(len(model.vit.encoder.layer)):
        if (i) % gap == 0:
            original_layer = model.vit.encoder.layer[i].output.output_adapters
            model.vit.encoder.layer[i].output.output_adapters = AdapterWithFourierTransform(original_layer, do_invert=args.invert_domain)

            original_layer2 = model.vit.encoder.layer[0].attention_adapters
            model.vit.encoder.layer[i].attention_adapters = AdapterWithFourierTransform(original_layer2, do_invert=args.invert_domain)

    return model

# Step 3: Replace the Existing Layers in the Model
def add_fourier_transform_to_vit(model, gap=1):
    for i in range(len(model.vit.encoder.layer)):
        if (i) % gap == 0:
            original_layer = model.vit.encoder.layer[i]
            model.vit.encoder.layer[i] = ViTLayerWithFourierTransform(original_layer, args.invert_domain)

    return model


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

    if args.train_range:
        folder += "_range"

    folder += "_"+str(args.dataset_fraction)
    if not os.path.exists(args.outdir+folder+"/"+str(args.noise_sd)):
        os.makedirs(args.outdir+folder+"/"+str(args.noise_sd), exist_ok=True)

    if args.do_norm == 0:
        args.do_norm = False

    else:
        args.do_norm = True

    print("Normalization : ", args.do_norm)

    if "vit" in args.arch or "swin" in args.arch:
        global VIT
        VIT = True
        train_dataset = get_dataset(args.dataset, 'train', args.data_dir, model_input=32, do_norm=args.do_norm)
        test_dataset = get_dataset(args.dataset, 'test', args.data_dir, model_input=32, do_norm=args.do_norm)

    else:
        train_dataset = get_dataset(args.dataset, 'train', args.data_dir, do_norm=args.do_norm)
        test_dataset = get_dataset(args.dataset, 'test', args.data_dir, do_norm=args.do_norm)

    # Define the desired subset size
    subset_size = int(len(train_dataset) * args.dataset_fraction)

    # Create a subset of the CIFAR10 dataset
    subset_indices = torch.randperm(len(train_dataset))[:subset_size]
    subset_train_dataset = Subset(train_dataset, subset_indices)

    pin_memory = (args.dataset == "imagenet")
    
    print("creating train dataloader with dataset of size : ",len(subset_train_dataset))
    train_loader = DataLoader(subset_train_dataset, shuffle=True, batch_size=args.batch,
                              num_workers=args.workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch,
                             num_workers=args.workers, pin_memory=pin_memory)

    model = get_architecture(args.arch, args.dataset)

    if "vit" in args.arch:
        # global VIT
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

    if os.path.isfile(model_path) and args.dataset not in ["cifar10", "imagenet"]:
        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        # starting_epoch = checkpoint['epoch']
        # Create a new state dict with modified keys
        new_state_dict = {}
        for k, v in checkpoint["state_dict"].items():
            if k.startswith('1.module.'):
                new_key = k.replace('1.module.', '1.')
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict)

        print("=> loaded checkpoint '{}' (epoch {})"
                        .format(model_path, checkpoint['epoch']))
        
        #add adapters
        # print(model)
        normalize_layer, model = model
        adapters.init(model)

    if args.dataset in ["cifar10", "imagenet"]:
        normalize_layer, model = model
        adapters.init(model)

    if args.resume : 
        # load adapter from path
        model.load_adapter(adapter_path)
        checkpoint_adapter = torch.load(os.path.join(adapter_path, 'checkpoint.pth.tar'),
                            map_location=lambda storage, loc: storage)
        # starting_epoch = checkpoint_adapter['epoch']
        best = checkpoint_adapter['test_acc']

    else:
        model.add_adapter("denoising-adapter-"+str(int(args.noise_sd*100)), config=config)
        init_logfile(logfilename, "epoch\ttime\tlr\ttrainloss\ttestloss\ttrainAcc\ttestAcc")
        best = 0.0 

    #set active adapter
    model.set_active_adapters("denoising-adapter-"+str(int(args.noise_sd*100)))
    model.train_adapter("denoising-adapter-"+str(int(args.noise_sd*100)))

    if args.dataset not in ['cifar10', 'imagenet']:
        model = torch.nn.Sequential(normalize_layer, model) 

    if args.do_fourier:
        if args.fourier_location == 'attention':
            model = add_fourier_transform_to_vit(model, gap=args.gap)
        else:
            model = add_fourier_transform_to_adapters(model, gap=1)


    model.to('cuda')
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.scheduler == 'step':
        scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr/1000)

    if args.resume:
        optimizer.load_state_dict(checkpoint_adapter['optimizer'])
        #change optimizers lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

    project_name = args.arch+"_"+args.dataset+"_"+args.adapter_config+"_"+str(args.dataset_fraction)+"_"+str(args.noise_sd)
    if args.do_fourier:
        project_name += "_fourier_"+args.fourier_location
        if args.invert_domain:
            project_name += "_invert"

    if args.train_range:
        project_name += "_range"

    wandb.init(
    # set the wandb project where this run will be logged
    project=project_name,

    # track hyperparameters and run metadata
    config={
    "learning_rate": args.lr,
    "architecture": args.arch,
    "dataset": args.dataset,
    "epochs": args.epochs,
    }
    )


    model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)
    print("Training " +  str((count_parameters_trainable(model)/(count_parameters_total(model)))*100)  +"% of the parameters")
    print("starting training")
    for epoch in range(starting_epoch, args.epochs):
        before = time.time()

        if args.mixup:
            train_loss = train_mixup(train_loader, criterion, optimizer, epoch, args.noise_sd, model, mixup_lam=args.mixup_lam)
            train_acc = 0

        elif args.train_range:
            train_loss, train_acc = train_range(train_loader, model, criterion, optimizer, epoch, args.noise_sd)
            test_loss, test_acc = test_range(test_loader, model, criterion, args.noise_sd)
        else:
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
            if args.dataset not in ['cifar10']:
                normalize_layer, model = model

            # Save adapter
            model.save_adapter( args.outdir+folder+"/"+str(args.noise_sd), "denoising-adapter-"+str(int(args.noise_sd*100)))

            if args.dataset not in ['cifar10']:
                model = torch.nn.Sequential(normalize_layer, model)
            # torch.save({
            #     'epoch': epoch + 1,
            #     'arch': args.arch,
            #     'optimizer': optimizer.state_dict(),
            #     'state_dict': model.state_dict(),
            #     'train_acc' : train_acc,
            #     'test_acc' : test_acc,
            #     'train_loss' : train_loss,
            #     'test_loss' : test_loss,
            # }, os.path.join(adapter_path, 'checkpoint.pth.tar'))

        wandb.log({"train_loss": train_loss, "test_loss": test_loss, "train_acc": train_acc, "test_acc": test_acc, "best" : best, "lr" : scheduler.get_lr()[0]})
        
def train_mixup(loader: DataLoader, criterion, optimizer: Optimizer, epoch: int, noise_sd: float, classifier: torch.nn.Module=None, mode='instance', mixup_lam=0.1):
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

        outputs_mixup = classifier(inputs_mixup + noise)
        outputs_mixup = outputs_mixup.logits
        
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
            outputs = classifier(inputs + noise)
            outputs = outputs.logits

            if isinstance(criterion, MSELoss):
                loss = criterion(outputs, inputs)
            else:
                if args.objective in ['classification', 'stability', 'focal', 'ldam'] and args.ssl_like == 1:
                    with torch.no_grad():
                        targets = classifier(inputs).logits
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
    # Define the resize transform
    resize_transform = transforms.Resize((224, 224))

    for i, (inputs, targets) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs = inputs.cuda()
        targets = targets.cuda()

        # augment inputs with noise
        if noise_sd == -1:
            #choose randomly a value between 0 and 1
            noise_sd = random.random()

        inputs = inputs + (torch.randn_like(inputs, device='cuda') * noise_sd) 
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
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))

    return (losses.avg, top1.avg)

def transform_noise(noise, a=0.5, b=0.75):
    """
    Transform the values in the noise matrix from (-1, 1) to (-b, -a) U (a, b).

    Parameters:
    - noise: A NumPy array with values in the range (-1, 1).
    - a: The lower bound of the positive target range.
    - b: The upper bound of the positive target range.

    Returns:
    - transformed_noise: A NumPy array with transformed values.
    """
     # Create a copy to avoid modifying the original tensor
    result = noise.clone()
    
    # Transform negative values
    negative_mask = noise < 0
    result[negative_mask] = -b + (noise[negative_mask] + 1) * (b - a)
    
    # Transform positive values
    positive_mask = noise > 0
    result[positive_mask] = a + noise[positive_mask] * (b - a)
    
    # Values exactly equal to 0 remain unchanged
    return result

def train_range(loader: DataLoader, model: torch.nn.Module, criterion, optimizer: Optimizer, epoch: int, noise_sd: float):
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
        max = noise_sd
        min = noise_sd - 0.25
        noise = transform_noise(torch.randn_like(inputs, device='cuda'), min, max)
        inputs = inputs + noise

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

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))

    return (losses.avg, top1.avg)

def test_range(loader: DataLoader, model: torch.nn.Module, criterion, noise_sd: float):
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

            # augment inputs with noise
            max = noise_sd
            min = noise_sd - 0.25
            noise = transform_noise(torch.randn_like(inputs, device='cuda'), min, max)
            inputs = inputs + noise


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

    # Define the resize transform
    resize_transform = transforms.Resize((224, 224))

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            # measure data loading time
            data_time.update(time.time() - end)

            inputs = inputs.cuda()
            targets = targets.cuda()

            # augment inputs with noise
            if noise_sd == -1:
                #choose randomly a value between 0 and 1
                noise_sd = random.random()
                
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
