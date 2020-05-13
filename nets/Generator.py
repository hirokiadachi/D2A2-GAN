import torch
import torch.nn as nn


class GBlock(nn.Module):
    def __init__(self, in_features, out_features, k=3, s=1, p=1, bias=False, layer='normal'):
        super(GBlock, self).__init__()
        self.model = nn.Sequential()
        self.model.add_module('Convolution_1',
            nn.ConvTranspose2d(in_features, out_features, kernel_size=k, stride=s, padding=p, bias=bias))
        self.model.add_module('BatchNorm_1',
            nn.BatchNorm2d(out_features))
        self.model.add_module('Activation_1',
            nn.LeakyReLU())
        self.model.add_module('Convolution_2',
            nn.Conv2d(out_features, out_features, kernel_size=3, stride=1, padding=1, bias=bias))
        if layer == 'normal':
            self.model.add_module('BatchNorm_2',
                nn.BatchNorm2d(out_features))
            self.model.add_module('Activation_2',
                nn.LeakyReLU())
        elif layer == 'last':
            self.model.add_module('Activation_2',
                nn.Tanh())
            
    def forward(self, x):
        return self.model(x)
    
class Generator(nn.Module):
    def __init__(self, latent_dim, class_num):
        super(Generator, self).__init__()
        self.l0 = GBlock(latent_dim+class_num, 256, k=2, s=1, p=0, bias=False, layer='normal')
        self.l1 = GBlock(256, 128, k=4, s=2, p=1, bias=False, layer='normal')
        self.l2 = GBlock(128, 64, k=4, s=2, p=1, bias=False, layer='normal')
        self.l3 = GBlock(64, 32, k=4, s=2, p=1, bias=False, layer='normal')        
        self.l4 = GBlock(32, 3, k=4, s=2, p=1, bias=False, layer='last')     
        
    def forward(self, z):
        h = self.l0(z)
        h = self.l1(h)
        h = self.l2(h)
        h = self.l3(h)
        out = self.l4(h)
        
        return out
