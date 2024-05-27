# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from architectures import get_architecture, IMAGENET_CLASSIFIERS
from datasets import get_dataset, DATASETS
from torch.nn import MSELoss, CrossEntropyLoss
from torch.optim import SGD, Optimizer, Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from train_utils import AverageMeter, accuracy, init_logfile, log

from sklearn.metrics import confusion_matrix, balanced_accuracy_score, accuracy_score

import argparse
import datetime
import numpy as np
import os
import time
import torch
import collections

toPilImage = ToPILImage()

def main(args):
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    results = {}
    args.denoiser = os.path.join(os.getenv('PT_DATA_DIR', './'), args.denoiser)
    checkpoint = torch.load(args.denoiser)
    denoiser = get_architecture(checkpoint['arch'] ,args.dataset)
    denoiser.load_state_dict(checkpoint['state_dict'])
    denoiser.cuda().eval()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    args.outdir = os.path.join(os.getenv('PT_OUTPUT_DIR', './'), args.outdir)
    
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    test_dataset = get_dataset(args.dataset, 'test')
    pin_memory = (args.dataset == "imagenet")

    if args.test_subset:
        subset_len = int(len(test_dataset)/100)
        test_dataset, _ = torch.utils.data.random_split(test_dataset, [subset_len, len(test_dataset) - subset_len])

    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch,
                             num_workers=args.workers, pin_memory=pin_memory)

    if args.clf != '':
        if args.clf in IMAGENET_CLASSIFIERS:
            assert args.dataset == 'imagenet'
            # loading pretrained imagenet architectures
            clf = get_architecture(args.clf, args.dataset, pytorch_pretrained=True)
        else:
            args.clf = os.path.join(os.getenv('PT_DATA_DIR', './'), args.clf)
            checkpoint = torch.load(args.clf)
            clf = get_architecture(checkpoint['arch'], args.dataset)
            clf.load_state_dict(checkpoint['state_dict'])

        clf.cuda().eval()

        classification_criterion = CrossEntropyLoss(size_average=None, reduce=None, reduction = 'mean').cuda()

        print(f'WITH DENOISER {args.denoiser}')
        test_with_classifier(test_loader, denoiser, classification_criterion, args.noise_sd, args.print_freq, clf)
        print('WITHOUT DENOISER')
        test_with_classifier(test_loader, None, classification_criterion, args.noise_sd, args.print_freq, clf)

    return results


def test_with_classifier(loader: DataLoader, denoiser: torch.nn.Module, criterion, noise_sd: float, print_freq: int, classifier: torch.nn.Module):
    """
    A function to test the classification performance of a denoiser when attached to a given classifier
        :param loader:DataLoader: test dataloader
        :param denoiser:torch.nn.Module: the denoiser 
        :param criterion: the loss function (e.g. CE)
        :param noise_sd:float: the std-dev of the Guassian noise perturbation of the input
        :param print_freq:int: the frequency of logging
        :param classifier:torch.nn.Module: the classifier to which the denoiser is attached
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    # switch to eval mode
    classifier.eval()
    if denoiser:
        denoiser.eval()

    pred = []
    true = []
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            # measure data loading time
            data_time.update(time.time() - end)

            inputs = inputs.cuda()
            targets = targets.cuda()

            # augment inputs with noise
            inputs = inputs + torch.randn_like(inputs, device='cuda') * noise_sd

            if denoiser is not None:
                inputs = denoiser(inputs)

            # compute output
            outputs = classifier(inputs)

            _, predictions = torch.max(outputs, dim = 1)
            pred += predictions.detach().cpu().tolist()
            true += targets.detach().cpu().tolist()

    print(f"Balanced Accuracy for dataset is: {balanced_accuracy_score(true, pred)}, and Normal Accuracy is {accuracy_score(true, pred)}")
    # matrix = confusion_matrix(true, pred)
    # matrix = matrix.diagonal()/matrix.sum(axis=1)
    # frequency = dict(collections.Counter(true))
    # for i in range(len(matrix)):
    #     print(f"For class {i} with {frequency[i]} samples, the accuracy is {matrix[i]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--dataset', type=str, choices=DATASETS, required=True)
    parser.add_argument('--denoiser', type=str, default='',
                        help='Path to a denoiser ', required=True)
    parser.add_argument('--clf', type=str, default='',
                        help='Pretrained classificaiton model.', required=True)
    parser.add_argument('--outdir', type=str, default='resnet18_breakhis/test_denoise/', help='folder to save model and training log)')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--batch', default=16, type=int, metavar='N',
                        help='batchsize (default: 256)')
    parser.add_argument('--gpu', default=None, type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--noise_sd', default=0.0, type=float,
                        help="standard deviation of noise distribution for data augmentation")
    parser.add_argument('--test-subset', action='store_true',
                        help='evaluate only a predifined subset ~(1%) of the test set')
    args = parser.parse_args()
    
    main(args)
