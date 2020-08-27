# Written by lvyan on 2020/8/19


import torch
import torch.nn as nn
from .DeepCL_parts import *


class CNet(nn.Module):

    def __init__(self, n_channels, n_classes):
        super(CNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, n_classes, 3, padding=1),
            nn.BatchNorm2d(n_classes),
            nn.ReLU(inplace=True),
            )
        
        self.conv6 = nn.Conv2d(n_classes, 1, 1, padding=0)
        self.maxpool = nn.MaxPool2d(2)
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        self.ca = ChannelAttention(128)
        self.sa = SpatialAttention()
              
    def forward(self, input):
        conv1 = self.conv1(input)
        pool1 = self.maxpool(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.maxpool(conv2)
        conv3 = self.conv3(pool2)

        up1 = self.deconv1(conv3)
        up1 = torch.cat([up1, conv2], dim=1)
        conv4 = self.conv4(up1)
        up2 = self.deconv2(conv4)
        up2 = torch.cat([up2, conv1], dim=1)
        conv5_1 = self.conv5(up2)
        att = self.ca(up2) * up2
        att = self.sa(att) * att
        conv5_2 = self.conv5(att)

        out1 = self.conv6(conv5_1)
        out2 = self.conv6(conv5_2)

        return out1, out2





