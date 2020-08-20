import os
import cv2
import math
import shutil
import random
import argparse
import numpy as np
import multiprocessing
from PIL import Image

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets

from network import ResNet

p = argparse.ArgumentParser()
p.add_argument('--batch_size', '-b', type=int, default=128)
p.add_argument('--epoch', '-e', type=int, default=300)
p.add_argument('--gamma', type=float, default=0.1)
p.add_argument('--lr', type=float, default=0.1)
p.add_argument('--momentum', type=float, default=0.9)
p.add_argument('--weight_decay', type=float, default=5e-4)
p.add_argument('--workers', type=int, default=multiprocessing.cpu_count())
p.add_argument('--gpu', nargs='*', type=int, required=True)
p.add_argument('--scheduler', nargs='*', type=int, required=True, default=[150, 225])
p.add_argument('--model_dir', type=str, default='result')
p.add_argument('--model_name', type=str, default='model')
p.add_argument('--tb_dir', type=str, default='logs')
p.add_argument('--mixup', type=str, default='original')
p.add_argument('--data', type=str, default='cifar10')
p.add_argument('--pretrained_path', type=str, default='')
args = p.parse_args()

if not os.path.exists(args.model_dir):
    os.makedirs(args.model_dir)
tb = SummaryWriter(log_dir=args.tb_dir)
state = {k: v for k, v in args._get_kwargs()}

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.scheduler:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']
            

if args.data == 'svhn':
    print('Dataset: SVHN')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1505, 0.1517, 0.1616))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1505, 0.1517, 0.1616))
    ])
    train_data = datasets.SVHN(root='/root/work/Datasets/data', split='train', transform=transform_train)
    test_data = datasets.SVHN(root='/root/work/Datasets/data', split='test', transform=transform_test)

elif args.data == 'cifar10':
    print('Dataset: CIFAR10')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train_data = datasets.CIFAR10(root='/root/work/Datasets/data', train=True, transform=transform_train)
    test_data = datasets.CIFAR10(root='/root/work/Dataset/data', train=False, download=True, transform=transform_test)
    
train_items = DataLoader(dataset=train_data,
                         batch_size=args.batch_size,
                         shuffle=True,
                         num_workers=args.workers)

test_itmes = DataLoader(dataset=test_data,
                         batch_size=args.batch_size,
                         shuffle=True,
                         num_workers=args.workers)
            
iters = 0
test_iter = 0
torch.cuda.set_device(args.gpu[0])
model = nn.DataParallel(ResNet(depth=19), device_ids=args.gpu).cuda()
    
print(model)
opt = optim.SGD(model.parameters(), lr=args.lr, 
                momentum=args.momentum, weight_decay=args.weight_decay)

criterion = nn.CrossEntropyLoss().cuda()
    
batch_ind = 0
for epoch in range(1, args.epoch+1):
    adjust_learning_rate(opt, epoch)
    for index, samples in enumerate(train_items):
        inputs, targets = samples[0].cuda(), samples[1].cuda(async=True)
            
        gt = Variable(torch.arange(10).view(1, 10).expand(args.batch_size, 10).cuda())
        #targets = targets.view(inputs.size(0), -1).argmax(dim=1)
        
        per_out, attn_out, _ = model(inputs)
        per_loss = criterion(per_out, targets)
        attn_loss = criterion(attn_out, targets)
        loss = per_loss + attn_loss
            
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        iters += 1
        if index % 100 == 0:
            per_correct_rate = 0
            attn_correct_rate = 0
            total = targets.size(0)
            per_pred = torch.max(per_out.data, 1)[1]
            attn_pred = torch.max(attn_out.data, 1)[1]
            per_correct_rate += per_pred.eq(targets.data).cpu().sum().float()
            attn_correct_rate += attn_pred.eq(targets.data).cpu().sum().float()
            print('Epoch %d (%d iters) | loss (per): %f | acc: %f |'\
                % (epoch, index, per_loss.item(), per_correct_rate / total))
            tb.add_scalars('train_loss',
                          {'perception': per_loss.item(),
                           'attention': attn_loss.item()}, 
                          global_step=iters)
            tb.add_scalars('train_acc', 
                          {'percention': per_correct_rate / total,
                           'attention': attn_correct_rate / total}, 
                          global_step=iters)
    torch.save(model.state_dict(), os.path.join(args.model_dir, args.model_name))
