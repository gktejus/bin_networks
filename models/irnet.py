import torch
import torch.nn as nn
from torch.autograd import Function,Variable
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import transforms, datasets
from torchvision.datasets import CIFAR10, CIFAR100
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm.notebook import tqdm
import torch.optim as optim
import pandas as pd
import math
from collections import OrderedDict
from torch.autograd import Function


class BinaryQuantize(Function):
    @staticmethod
    def forward(ctx, input, k, t):
        ctx.save_for_backward(input, k, t)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, k, t = ctx.saved_tensors
        grad_input = k * t * (1 - torch.pow(torch.tanh(input * t), 2)) * grad_output
        return grad_input, None, None

class IRConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(IRConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.k = torch.tensor([10]).float().cuda()
        self.t = torch.tensor([0.1]).float().cuda()

    def forward(self, input):
        w = self.weight
        a = input
        bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1, 1)
        bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1, 1, 1)
        sw = torch.pow(torch.tensor([2]*bw.size(0)).cuda().float(), (torch.log(bw.abs().view(bw.size(0), -1).mean(-1)) / math.log(2)).round().float()).view(bw.size(0), 1, 1, 1).detach()
        bw = BinaryQuantize().apply(bw, self.k, self.t)
        ba = BinaryQuantize().apply(a, self.k, self.t)
        bw = bw * sw
        output = F.conv2d(ba, bw, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)
        return output

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = IRConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = IRConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
       #Double Skip 
        residual = x 
        out = self.conv1(x)
        out = self.bn1(out)
        out+= self.shortcut(residual)
        out = F.hardtanh(out)
        residual = out
        out = self.conv2(out)
        out = self.bn2(out)
        out+=residual
        out = F.hardtanh(out)
        
        
          #Single Skip
#         out = F.hardtanh(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = F.hardtanh(out)
       
        return out






class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.bn2 = nn.BatchNorm1d(512*block.expansion)
        # self.softmax = nn.LogSoftmax()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.bn2(out)
        out = self.linear(out)
        # out = self.softmax(out)
        return out 

def get_irnet(num_classes):
    return ResNet(BasicBlock, [2,2,2,2], num_classes)
