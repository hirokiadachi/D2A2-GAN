import torch
import torch.nn as nn
import torch.nn.functional as F

class DBlock(nn.Module):
    def __init__(self, in_features, out_features, k=3, s=1, p=1, bias=False, layer='normal', act='lrelu'):
        super(DBlock, self).__init__()
        self.model = nn.Sequential()
        if layer == 'first':
            self.model.add_module('Convolution',
                nn.Conv2d(in_features, out_features, kernel_size=k, stride=s, padding=p, bias=bias))
            self.model.add_module('Activation',
                nn.LeakyReLU())
            
        elif layer == 'normal':
            self.model.add_module('Convolution',
                nn.Conv2d(in_features, out_features, kernel_size=k, stride=s, padding=p, bias=bias))
            self.model.add_module('BatchNorm',
                nn.BatchNorm2d(out_features))
            if act == 'lrelu':
                self.model.add_module('Activation',
                    nn.LeakyReLU())
            elif act == 'relu':
                self.model.add_module('Activation',
                    nn.ReLU(inplace=True))
            
        
    def forward(self, x):
        return self.model(x)
    
class Discriminator(nn.Module):
    def __init__(self, in_ch, class_num):
        super(Discriminator, self).__init__()
        self.class_num = class_num
        self.l0 = DBlock(in_ch, 32, k=4, s=2, p=1, bias=False, layer='first')
        self.l1 = DBlock(32, 64, k=4, s=2, p=1, bias=False, layer='normal', act='relu')
        self.attn_conv_layers, self.attn_block1, self.attn_block2 = self.make_attention_branch(nn.Sequential(), 64, 2)
        self.adv_conv_layers, self.adv_block = self.make_adversarial_branch(nn.Sequential(), 64, 2)
        
        
    def make_adversarial_branch(self, model, in_features, n):
        for i in range(n):
            model.add_module('Convolution_%d' % i,
                nn.Conv2d(in_features, in_features*2, kernel_size=4, stride=2, padding=1, bias=False))
            model.add_module('BatchNorm_%d' % i,
                nn.BatchNorm2d(in_features*2))
            model.add_module('Activation_%d' % i,
                nn.LeakyReLU())
            in_features = in_features * 2
        
        adv_block = nn.Sequential(
            nn.Conv2d(in_features+1, 1, kernel_size=2, stride=1, padding=0))
                
        return model, adv_block
    
    def make_attention_branch(self, model, in_features, n):
        for i in range(n):
            model.add_module('Convolution_%d' % i,
                nn.Conv2d(in_features, in_features*2, kernel_size=3, stride=1, padding=1, bias=False))
            model.add_module('BatchNorm_%d' % i,
                nn.BatchNorm2d(in_features*2))
            model.add_module('Activation_%d' % i,
                nn.ReLU(inplace=True))
            in_features = in_features * 2
        
        block1 = nn.Sequential(
            nn.Conv2d(in_features, self.class_num, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.class_num),
            nn.ReLU(inplace=True))
        
        block2 = nn.Sequential(
            nn.Conv2d(self.class_num, self.class_num, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AdaptiveAvgPool2d(1))
        
        return model, block1, block2
        
    def attention_mechanisim(self, adv_in_feature, attn_out_feature):
        attn_out_feature_sum = attn_out_feature.sum(dim=1, keepdim=True)
        max_val = attn_out_feature_sum.flatten(start_dim=1).max(dim=1)[0].view(-1, 1, 1, 1)
        min_val = attn_out_feature_sum.flatten(start_dim=1).min(dim=1)[0].view(-1, 1, 1, 1)
        attn_map = (attn_out_feature_sum - min_val) / (max_val - min_val)
        
        feature_map = (adv_in_feature * attn_map.expand(adv_in_feature.shape)) + adv_in_feature
        return feature_map, attn_map
        
    def minibatch_std(self, x, eps=1e-8):
        stddev = torch.sqrt(
            torch.mean((x - torch.mean(x, dim=0, keepdim=True))**2, dim=0, keepdim=True) + eps)
        inject_shape = list(x.size())[:]
        inject_shape[1] = 1
        inject = torch.mean(stddev, dim=1, keepdim=True)
        inject = inject.expand(inject_shape)
        return torch.cat((x, inject), dim=1)
    
    def forward(self, x):
        h = self.l0(x)
        h = self.l1(h)
        
        # attention branch
        h_attn = self.attn_conv_layers(h)
        attn_features = self.attn_block1(h_attn)
        cls_out = self.attn_block2(attn_features)
        
        h, attn_map = self.attention_mechanisim(h, attn_features)
        # adversarial branch
        h_adv = self.adv_conv_layers(h)
        h_adv = self.minibatch_std(h_adv)
        h_adv = self.adv_block(h_adv)
        h_adv = h_adv.flatten(start_dim=1)
        adv_out = h_adv
        
        return adv_out, cls_out, attn_map
    