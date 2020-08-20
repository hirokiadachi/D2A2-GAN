import os 
import sys
import shutil
import argparse
import numpy as np
import multiprocessing

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

import utils
from ABN.network import ResNet
from nets.Generator import Generator
from nets.Discriminator import Discriminator

def args():
    p = argparse.ArgumentParser()
    p.add_argument('--latent_dim', type=int, default=100,
                   help='Latent variable dimension.')
    p.add_argument('-b', '--batch_size', type=int, default=100,
                   help='Training batch size.')
    p.add_argument('-t', '--training_data_name', required=True,
                   help='Select training data (CIFAR-10 or SVHN).')
    p.add_argument('--log_dir', type=str, default='logs',
                   help='Tensorboard directory name.')
    p.add_argument('--result_dir', type=str, default='result',
                   help='Network models saving directory name.')
    p.add_argument('--lr', type=float, default=0.0002,
                   help='Optimizer learning rate.')
    p.add_argument('--beta1', type=float, default=0.5)
    p.add_argument('--beta2', type=float, default=0.999)
    p.add_argument('--gpu', nargs='*', type=int, required=True,
                   help='GPU number.')
    p.add_argument('--workers', type=int, default=multiprocessing.cpu_count(),
                   help='Number of CPU worker.')
    p.add_argument('-e', '--epoch', type=int, default=100,
                   help='Number of trainig epoch.')
    p.add_argument('-n', '--num_gen', type=int, default=10,
                   help='Number of generation images by Generater when test time.')
    p.add_argument('--gen', type=str, default='gen',)
    p.add_argument('--dis', type=str, default='dis')
    p.add_argument('--critic', type=int, default=1)
    p.add_argument('--lr_sgd', type=float, default=0.1)
    p.add_argument('--weight_decay', type=float, default=5e-4)
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--gamma', type=float, default=0.1)
    return p.parse_args()

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
args = args()
if os.path.exists(args.log_dir):
    shutil.rmtree(args.log_dir)
    
tb = SummaryWriter(log_dir=args.log_dir)
transform_trian = transforms.Compose([
        transforms.ToTensor()
    ])

if not os.path.exists(args.result_dir):
    os.makedirs(args.result_dir)

# Choose dataset using training
if args.training_data_name == 'cifar10':
    training_data = datasets.CIFAR10(root='/root/work/Datasets/data', download=True, transform=transform_trian)
    txt_name = 'cifar10_class_names.txt'
    with open(txt_name, 'r') as f:
        class_names = f.read().split()
elif args.training_data_name == 'svhn':
    training_data = datasets.SVHN(root='/root/work/Datasets/data', download=True, transform=transform_trian)
    class_names = np.arange(10)
else:
    assert 0, '%s is not supprted.' % args.training_data_name
    
training_item = DataLoader(dataset=training_data, batch_size=args.batch_size,
                           drop_last=True, shuffle=True, num_workers=args.workers)

torch.cuda.set_device(args.gpu[0])
if len(args.gpu) > 1:
    G = nn.DataParallel(
        Generator(args.latent_dim, class_num=len(class_names)), device_ids=args.gpu).cuda()
    D = nn.DataParallel(
        Discriminator(in_ch=3, class_num=len(class_names)), device_ids=args.gpu).cuda()
    ABN = nn.DataParallel(
        ResNet(depth=18), device_ids=args.gpu).cuda()
else:
    G = Generator(args.latent_dim, class_num=len(class_names)).cuda()
    D = Discriminator(in_ch=3, class_num=len(class_names)).cuda()
    ABN = ResNet(depth=18).cuda()
    
G.apply(weight_init)
D.apply(weight_init)

opt_G = optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
opt_D = optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
opt_abn = optim.SGD(ABN.parameters(), lr=args.lr_sgd, momentum=args.momentum, weight_decay=args.weight_decay)

criterion_gan = nn.BCEWithLogitsLoss()
criterion_classification = nn.CrossEntropyLoss()

scheduler = [int(args.epoch*0.5), int(args.epoch*0.75)]
state = {k: v for k, v in args._get_kwargs()}
def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in scheduler:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

def training_abn(epoch, model, opt_abn, train_data, iteration, args):
    model.train()
    adjust_learning_rate(opt_abn, epoch)
    for index, samples in enumerate(train_data):
        inputs, targets = samples[0].cuda(), samples[1].cuda()
        
        per_out, attn_out, _ = model(inputs)
        per_loss = criterion_classification(per_out, targets)
        attn_loss = criterion_classification(attn_out, targets)
        loss = per_loss + attn_loss
            
        opt_abn.zero_grad()
        loss.backward()
        opt_abn.step()
    
        iteration += 1
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
            tb.add_scalars('train_loss/abn',
                       {'perception': per_loss.item(),
                        'attention': attn_loss.item()}, 
                        global_step=iteration)
            tb.add_scalars('train_acc/abn', 
                       {'percention': per_correct_rate / total,
                        'attention': attn_correct_rate / total}, 
                        global_step=iteration)
        
    return iteration

def training_core(epoch, gen, dis, abn, opt_g, opt_d, train_data, iteration, args):
    Tensor = torch.cuda.FloatTensor
    gen.train()
    dis.train()
    abn.eval()
    
    for index, samples in enumerate(train_data):
        inputs, targets = samples[0].cuda(), samples[1].cuda()
        alpha = 1e-4
        b = inputs.size(0)
        flag_real = torch.autograd.Variable(Tensor(b, 1).fill_(1.0), requires_grad=False)
        flag_fake = torch.autograd.Variable(Tensor(b, 1).fill_(0.0), requires_grad=False)
        latent_var = torch.randn(b, args.latent_dim+len(class_names), 1, 1).cuda()
        onehot_targets = torch.eye(len(class_names))[targets.view(-1)].view(b, -1, 1, 1)
        latent_var[:, :10] = onehot_targets
        
        for _ in range(args.critic):
            dis.zero_grad()
            fake_img = gen(latent_var)
            with torch.no_grad():
                attn_map_real_gt = abn(inputs)[2]
                attn_map_fake_gt = abn(fake_img)[2]
                
            dis_real, pred_real, attn_map_real = dis(inputs)
            dis_fake, pred_fake, attn_map_fake = dis(fake_img.detach())
            
            dis_real_loss = criterion_gan(dis_real.view(-1), flag_real.view(-1))
            dis_fake_loss = criterion_gan(dis_fake.view(-1), flag_fake.view(-1))
            dis_loss = dis_real_loss + dis_fake_loss
            
            attn_real_loss = torch.sum(torch.abs(attn_map_real_gt - attn_map_real))
            attn_fake_loss = torch.sum(torch.abs(attn_map_fake_gt - attn_map_fake))
            attn_loss = attn_real_loss + attn_fake_loss
            
            pred_real_loss = criterion_classification(pred_real.view(b, -1),
                                                      targets.view(-1))
            pred_fake_loss = criterion_classification(pred_fake.view(b, -1),
                                                      targets.view(-1))
            dis_pred_loss = pred_real_loss + pred_fake_loss
            
            loss_dis = dis_loss + (alpha * attn_loss) + dis_pred_loss
            loss_dis.backward()
            opt_d.step()
            
        gen.zero_grad()
        fake_img = gen(latent_var)
        with torch.no_grad():
            attn_map_gen_gt = abn(fake_img)[2]
        gen_fake, pred_gen, attn_map_gen = dis(fake_img)
        gen_loss = criterion_gan(gen_fake.view(-1), flag_real.view(-1))
        attn_gen_loss = torch.sum(torch.abs(attn_map_gen_gt - attn_map_gen))
        pred_gen_loss = criterion_classification(pred_gen.view(b, -1),
                                                 targets.view(-1))
        loss_gen = gen_loss + pred_gen_loss + (alpha * attn_gen_loss)
        loss_gen.backward()
        opt_g.step()
        
        iteration += 1
        
        if index % 100 == 0:
            print('[Epoch: %d (%diterations)]' % (epoch, iteration))
            tb.add_scalars('D2A2-GAN loss',
                           {'dis': loss_dis.item(),
                            'gen': loss_gen.item()},
                           iteration)
            tb.add_scalars('Adversarial loss',
                           {'dis': dis_loss.item(),
                            'gen': gen_loss.item()},
                           iteration)
            tb.add_scalars('Classification loss',
                           {'Real/dis': pred_real_loss.item(),
                            'Fake/dis': pred_fake_loss.item(),
                           'Fake/gen': pred_gen_loss.item()},
                           iteration)
            real_max = torch.max(F.softmax(pred_real.view(b, -1), dim=1).detach(), dim=1)[1]
            fake_max = torch.max(F.softmax(pred_gen.view(b, -1), dim=1).detach(), dim=1)[1]
            correct_real = real_max.eq(targets).cpu().sum().float()
            correct_fake = fake_max.eq(targets).cpu().sum().float()
            correct_rate_real = (correct_real/b) * 100
            correct_rate_fake = (correct_fake/b) * 100
            
            tb.add_scalars('Accuracy',
                           {'Real/dis': correct_rate_real,
                            'Fake/gen': correct_rate_fake},
                           iteration)
            tb.close()
            
    return iteration

if __name__ == '__main__':
    iteration_abn = 0
    iteration = 0
    for epoch in range(1, args.epoch+1):
        iteration_abn = training_abn(epoch, ABN, opt_abn, training_item, iteration_abn, args)
        torch.save(ABN.state_dict(), os.path.join(args.result_dir, 'model_abn'))
        iteration = training_core(epoch, G, D, ABN, opt_G, opt_D, training_item, iteration, args)
        utils.save_tensorboard(epoch, G, D, training_data, args.num_gen, args.latent_dim, class_names, tb)
        torch.save(G.state_dict(), os.path.join(args.result_dir, args.gen))
        torch.save(D.state_dict(), os.path.join(args.result_dir, args.dis))