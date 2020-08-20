import os
import cv2
import PIL
import pickle 
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset


def save_tensorboard(epoch, G, D, training_data, num_gen, latent_dim, class_names, tb):
    G.eval()
    D.eval()
    rand_index = np.random.randint(len(training_data))
    latent_val = torch.randn(num_gen, latent_dim+10, 1, 1).cuda()
    targets = training_data[rand_index][1]
    onehot_targets = torch.eye(10)[targets]
    onehot_targets = onehot_targets.view(1, -1, 1, 1).expand(num_gen, 10, 1, 1).cuda()
    latent_val[:, :10] = onehot_targets
    real_img = training_data[rand_index][0].cuda()
    class_name = class_names[targets]
    
    with torch.no_grad():
        fake_img = G(latent_val)
        attn_map_fake = D(fake_img)[2]
        attn_map_real = D(real_img.unsqueeze(0))[2]
    
    attn_map_real = F.interpolate(attn_map_real, size=(real_img.size(1), real_img.size(2)),
                                  mode='bilinear', align_corners=True)
    attn_map_real = np.uint8((attn_map_real * 255).squeeze(0).clamp(min=0, max=255).cpu().data.numpy()).transpose(1,2,0)
    attn_map_real_jet = cv2.applyColorMap(attn_map_real, cv2.COLORMAP_JET)
    real_img_array = np.uint8((real_img * 255).clamp(min=0, max=255).cpu().data.numpy()).transpose(1,2,0)
    overlay_real = cv2.addWeighted(attn_map_real_jet, 0.3, real_img_array, 0.7, 0)[:, :, ::-1]
    
    tb.add_image('Real_result class_%s/Overlay' % class_name,
                 overlay_real, global_step=epoch, dataformats='HWC')
    tb.add_image('Real_result class_%s/Attention map' % class_name,
                 attn_map_real_jet[:, :, ::-1], global_step=epoch, dataformats='HWC')
    tb.add_image('Real_result class_%s/Image' % class_name,
                 real_img_array, global_step=epoch, dataformats='HWC')
        
    for index, (v_im, attn) in enumerate(zip(fake_img, attn_map_fake)):
        v_im = np.uint8((v_im * 255).clamp(min=0, max=255).cpu().data.numpy()).transpose(1,2,0)
        
        _attn_map = F.interpolate(attn.unsqueeze(0), size=(real_img.size(1), real_img.size(2)),
                                  mode='bilinear', align_corners=True)
        _attn_map = np.uint8((_attn_map * 255).squeeze().clamp(min=0, max=255).cpu().data.numpy())
        v_map = cv2.applyColorMap(_attn_map, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(v_map, 0.3, v_im, 0.7, 0)[:, :, ::-1]
        
        tb.add_image('Generated image/img_%d cls_%s' % (index, class_name),
                     v_im, global_step=epoch, dataformats='HWC')
        tb.add_image('Overlay_img class_%s/Attention_%d' % (class_name, index),
                     overlay, global_step=epoch, dataformats='HWC')
        
    torch.manual_seed(1729)
    latent_val = torch.randn(100, latent_dim+len(class_names), 1, 1).cuda()
    onehot_targets = torch.eye(10)[targets]
    onehot_targets = onehot_targets.view(1, -1, 1, 1).expand(100, -1, 1, 1).cuda()
    latent_val[:, :10] = onehot_targets
    with torch.no_grad():
        ims = G(latent_val)
        
    ims_cpu = ((ims + 1) / 2).clamp(min=0, max=1.).cpu().data.numpy()
    tb.add_images('Images class_%s' % class_name, ims_cpu, global_step=epoch)
        
        