import torch
import torch.nn as nn

from datasets import get_dataset
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset
from train_utils import AverageMeter, accuracy
from utils import count_parameters_trainable
import numpy as np
import os
import time
import random

sys_random = random.SystemRandom()
choices = np.array([0.25, 0.5, 0.75, 1.0])


def train(loader: DataLoader, model: torch.nn.Module, criterion, optimizer, epoch: int, noise_sd: float):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()  
    pop = 0
    model.train()

    for i, (inputs, targets) in enumerate(loader):
        data_time.update(time.time() - end)

        inputs = inputs.cuda()
        targets = targets.cuda()
        noise_sd = np.random.choice(choices)
        print(noise_sd)
        min_val = noise_sd - 0.25
        inputs = inputs + (torch.randn_like(inputs, device='cuda') * 0.25) + min_val 

        outputs = model(inputs)
        
        noise_class = torch.full((inputs.size(0),), choices.tolist().index(noise_sd), device='cuda').long()        
        loss = criterion(outputs, noise_class)

        acc1 = accuracy(outputs, noise_class, topk=(1,))[0]
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1.item(), inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        # accelerator.backward(loss)
        optimizer.step()

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

def test(loader: DataLoader, model: torch.nn.Module, criterion):
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        for i, (inputs, _) in enumerate(loader):
            inputs = inputs.cuda()
            batch_size = inputs.size(0)

            noise_sd = np.random.choice(choices)
            print(noise_sd)
            min_val = noise_sd - 0.25
            noisy_inputs = inputs + (torch.randn_like(inputs, device='cuda') * 0.25) + min_val

            outputs = model(noisy_inputs)
            
            noise_class = torch.full((batch_size,), choices.tolist().index(noise_sd), device='cuda').long()
            loss = criterion(outputs, noise_class)

            acc1 = accuracy(outputs, noise_class, topk=(1,))[0]
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))

            if i % 10 == 0:
                print('Test: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(loader), loss=losses, top1=top1))

    return losses.avg, top1.avg

class NoiseEstimator(nn.Module):
    def __init__(self):
        super(NoiseEstimator, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 56 * 56, 64)
        self.fc2 = nn.Linear(64, 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = NoiseEstimator()

train_dataset = get_dataset('cifar10', 'train', '/scratch/ravihm.scee.iitmandi/dataset/cifar10/', model_input=224)
test_dataset = get_dataset('cifar10', 'test', '/scratch/ravihm.scee.iitmandi/dataset/cifar10/', model_input=224)

subset_size = int(len(train_dataset) * 0.1)
subset_indices = torch.randperm(len(train_dataset))[:subset_size]
subset_train_dataset = Subset(train_dataset, subset_indices)

train_loader = DataLoader(subset_train_dataset, shuffle=True, batch_size=64)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=64)

criterion = CrossEntropyLoss().cuda()

model.to('cuda')
optimizer = SGD(model.parameters(), lr=1e-1, momentum=0.9, weight_decay=5e-4)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

print("Training " + str(count_parameters_trainable(model)) + "% of the parameters")
print("Starting training")

best = 0
for epoch in range(0, 90):
    before = time.time()
    train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, -1)
    test_loss, test_acc = test(test_loader, model, criterion)
    after = time.time()
    scheduler.step()
    if test_acc > best:
        print(f'New Best Found: {test_acc}%')
        best = test_acc
        os.makedirs("/scratch/ravihm.scee.iitmandi/models/cifar10/noise-detector", exist_ok=True)
        torch.save({
            'epoch': epoch + 1,
            'arch': "conv_net",
            'optimizer': optimizer.state_dict(),
            'state_dict': model.state_dict(),
            'train_acc': train_acc,
            'test_acc': test_acc,
            'train_loss': train_loss,
            'test_loss': test_loss,
        }, os.path.join("/scratch/ravihm.scee.iitmandi/models/cifar10/noise-detector", 'checkpoint.pth.tar'))
