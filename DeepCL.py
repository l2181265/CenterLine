# Written by lvyan on 2020/8/19


import torch
import torch.nn as nn
from torchsummary import summary


def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        # out = avg_out + max_out
        # return self.sigmoid(out)
        return self.sigmoid(avg_out) #sigmoid替换了原文的softmax

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
 
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
 
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class DeepCL(nn.Module):

    def __init__(self, n_class=2):
        super().__init__()
                
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
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
            nn.Conv2d(128, n_class, 3, padding=1),
            nn.BatchNorm2d(n_class),
            nn.ReLU(inplace=True),
            )
        
        self.conv6 = nn.Conv2d(n_class, 1, 1, padding=0)
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

if __name__=="__main__":
    from torch.autograd import Variable
    x = Variable(torch.randn(2,3,64,64)).cuda()

    model = DeepCL().cuda()
    summary(model, input_size = (3,64,64))
    param = count_param(model)
    y = model(x)
    print('Output:',y)
    print('DeepCL totoal parameters: %.2fM (%d)'%(param/1e6,param))




