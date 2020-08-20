import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_features, out_features, stride=1):
    return nn.Conv2d(in_features, out_features, 
                     kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_features, out_features, stride=1, down=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_features, out_features, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_features)
        
        self.conv2 = conv3x3(out_features, out_features)
        self.bn2 = nn.BatchNorm2d(out_features)
        
        self.act = nn.ReLU(inplace=True)
        self.down = down
        
    def forward(self, x):
        res = x
        
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.act(h)
        
        h = self.conv2(h)
        h = self.bn2(h)
        
        if self.down is not None:
            res = self.down(x)
        h += res
        return self.act(h)
    
class ResNet(nn.Module):
    def __init__(self, depth, classes=10, multi=False):
        super(ResNet, self).__init__()
        if multi:
            out_dim = 1
        else:
            out_dim = classes
        n = (depth - 2) // 6
        block = BasicBlock
        
        self.in_ch = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self.make_layer(block, 16, n, stride=2, down_size=True)
        self.layer2 = self.make_layer(block, 32, n, stride=2, down_size=True)
        
        self.attn_layer3 = self.make_layer(block, 64, n, down_size=False)
        self.attn_conv1 = nn.Conv2d(64 * block.expansion, 10, kernel_size=1, padding=0, bias=False)
        self.attn_bn1 = nn.BatchNorm2d(10)
        self.attn_conv2  = nn.Conv2d(10, 10, kernel_size=1, padding=0, bias=False)
        self.att_conv3  = nn.Conv2d(10, 1, kernel_size=3, padding=1, bias=False)
        self.bn_att3 = nn.BatchNorm2d(1)
        self.attn_gap = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
        
        
        self.layer3 = self.make_layer(block, 64, n, stride=2, down_size=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, out_dim)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
    def make_layer(self, block, ch, blocks, stride=1, down_size=None):
        downsamle = None
        if stride != 1 or self.in_ch != ch * block.expansion:
            downsamle = nn.Sequential(
                nn.Conv2d(self.in_ch, ch * block.expansion, 
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(ch*block.expansion))
            
        layers = []
        layers.append(block(self.in_ch, ch, stride, downsamle))
        
        if down_size:
            self.in_ch = ch * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.in_ch, ch))
            return nn.Sequential(*layers)
        else:
            in_ch = ch * block.expansion
            for i in range(1, blocks):
                layers.append(block(in_ch, ch))
            return nn.Sequential(*layers)
        
    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu(h)    # 32x32

        h = self.layer1(h)  # 32x32
        h = self.layer2(h)  # 16x16
        
        hr = self.attn_layer3(h)
        hr = self.relu(self.attn_bn1(self.attn_conv1(hr)))
        attn_out = self.attn_gap(self.attn_conv2(hr)).view(-1, 10)
        #attn_map = self.sigmoid(self.bn_att3(self.att_conv3(hr)))
        hr_sum = hr.sum(dim=1, keepdim=True)
        hr_max = hr_sum.flatten(start_dim=1).max(dim=1)[0].view(-1, 1, 1, 1)
        hr_min = hr_sum.flatten(start_dim=1).min(dim=1)[0].view(-1, 1, 1, 1)
        attn_map = (hr_sum - hr_min) / (hr_max - hr_min)
        
        h = (h * attn_map) + h
        h = self.layer3(h)
        h = self.avgpool(h)
        per_out = self.fc(torch.flatten(h, start_dim=1))
        return per_out, attn_out, attn_map
    
def resnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet(**kwargs)