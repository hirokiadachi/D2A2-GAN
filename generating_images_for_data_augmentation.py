import os
import cv2
import csv
import shutil
import argparse
import numpy as np
import multiprocessing
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.Generator import Generator
from nets.Discriminator import Discriminator

def args():
    p = argparse.ArgumentParser()
    p.add_argument('--latent_dim', type=int, default=100,
                   help='Latent variable dimension.')
    p.add_argument('-t', '--training_data_name', choices=['cifar10', 'svhn'], required=True,
                   help='Select training data (CIFAR-10 or SVHN).')
    p.add_argument('--log_dir', type=str, default='logs',
                   help='Tensorboard directory name.')
    p.add_argument('--result_dir', type=str, default='result',
                   help='Network models saving directory name.')
    p.add_argument('--workers', type=int, default=multiprocessing.cpu_count(),
                   help='Number of CPU worker.')
    p.add_argument('-n', '--num_gen', type=int, default=10,
                   help='Number of generation images by Generater when test time.')
    p.add_argument('--gen', type=str, default='gen')
    p.add_argument('--dis', type=str, default='dis')
    p.add_argument('-s', '--save_dir', default='test_samples')
    return p.parse_args()

args = args()
if args.training_data_name == 'cifar10':
    txt_name = 'cifar10_class_names.txt'
    with open(txt_name, 'r') as f:
        class_names = f.read().split()
elif args.training_data_name == 'svhn':
    class_names = np.arange(10)
    
if os.path.isdir(args.save_dir):
    shutil.rmtree(args.save_dir)
os.makedirs(args.save_dir)

img_dir = os.path.join(args.save_dir, 'Ims')
attn_dir = os.path.join(args.save_dir, 'Attn')
overlay_dir = os.path.join(args.save_dir, 'Overlay')
os.makedirs(img_dir)
os.makedirs(attn_dir)
os.makedirs(overlay_dir)

csv_file_pseudo = os.path.join(args.save_dir, 'pseudo_class_label.csv')
if os.path.exists(csv_file_pseudo):os.remove(csv_file_pseudo)
csv_file = os.path.join(args.save_dir, 'class_label.csv')
if os.path.exists(csv_file):os.remove(csv_file)

def save_items(img, attn, img_name, attn_name, overlay_name):
    img = (img*255.).squeeze().clamp(min=0., max=255.).data.cpu().numpy().transpose(1,2,0).astype(np.uint8)
    img_bgr = img[:, :, ::-1]
    img = Image.fromarray(img)
    img.save(img_name)
    
    attn = F.interpolate(attn, size=(32, 32), mode='bilinear')
    attn = (attn*256.).squeeze(0).squeeze(0).clamp(min=0., max=255.).data.cpu().numpy().astype(np.uint8)
    attn_map = cv2.applyColorMap(attn, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(attn_map, 0.3, img_bgr, 0.7, 0)
    cv2.imwrite(attn_name, attn_map)
    cv2.imwrite(overlay_name, overlay)
    
gpu = [0, 1]
G = nn.DataParallel(
    Generator(args.latent_dim, class_num=len(class_names)),
    device_ids=gpu).cuda()
D = nn.DataParallel(
    Discriminator(in_ch=3, class_num=len(class_names)),
    device_ids=gpu).cuda()

gen_model = os.path.join(args.result_dir, args.gen)
dis_model = os.path.join(args.result_dir, args.dis)
print('Generator: %s' % gen_model)
print('Discriminator: %s' % dis_model)
G.load_state_dict(torch.load(gen_model))
D.load_state_dict(torch.load(dis_model))

for i in tqdm(range(args.num_gen)):
    G.eval()
    D.eval()
    latent_val = torch.randn(1, args.latent_dim+len(class_names), 1, 1).cuda()
    cls_label = np.random.randint(len(class_names))
    onehot = torch.eye(len(class_names))[cls_label].view(1, len(class_names), 1, 1).float().cuda()
    latent_val[:, :len(class_names)] = onehot
    
    with torch.no_grad():
        _img = G(latent_val)
        _, _pred, _attn = D(_img)
    
    _pred = F.softmax(_pred)
    pseudo_label = _pred.data.cpu().view(-1).numpy()
    
    img_name = os.path.join(img_dir, 
        '{:0>6}_{}.png'.format(i, class_names[cls_label]))
    attn_name = os.path.join(attn_dir, 
        '{:0>6}_{}.png'.format(i, class_names[cls_label]))
    overlay_name = os.path.join(overlay_dir, 
        '{:0>6}_{}.png'.format(i, class_names[cls_label]))
    save_items(img = _img, attn=_attn, img_name=img_name,
        attn_name=attn_name, overlay_name=overlay_name)
    
    with open(csv_file_pseudo, 'a') as f:
        writer = csv.writer(f, lineterminator='\n')
        csvlist = []
        csvlist.append(img_name)
        csvlist.append(pseudo_label)
        writer.writerow(csvlist)
    
    with open(csv_file, 'a') as f:
        writer = csv.writer(f, lineterminator='\n')
        csvlist = []
        csvlist.append(img_name)
        csvlist.append(onehot.data.cpu().view(-1).numpy())
        writer.writerow(csvlist)
    