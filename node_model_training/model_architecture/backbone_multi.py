import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
import math
try:
    from torchinfo import summary  # noqa: F401
except ImportError:
    summary = None
import torch.nn.functional as F
from . import ECANet as att
# import GCN
# import SK
# import GCT
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, groups=groups, bias=False, dilation=dilation)


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Deconv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(Deconv2d, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, bias = False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Depthwise_conv2d(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(Depthwise_conv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, bias = False, groups = in_channels, **kwargs)
        #self.bn = nn.BatchNorm2d(in_channels, eps=0.001)
        #self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        #x = self.bn(x)
        #x = self.relu(x)
        return(x)

class Pointwise_conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super (Pointwise_conv2d, self).__init__()
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size = 1, bias = False)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return(x)




class Deconv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(Deconv2d, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, bias = False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class conv_M1(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_M1, self).__init__()
        self.branch1_1 = BasicConv2d(in_channels, 64, kernel_size = 1, stride = 1)
        self.branch1_2 = BasicConv2d(64, 64, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 4)
        self.branch1_3 = BasicConv2d(64, 64,kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 4)
        self.branch1_4 = nn.Dropout(0.25)

        self.branch2_1 = BasicConv2d(in_channels, 64, kernel_size = 1, stride = 1)
        self.branch2_2 = BasicConv2d(64, 64, kernel_size = 3, stride = 1, padding = 2, dilation = 2, groups =4)
        self.branch2_3 = BasicConv2d(64, 64, kernel_size = 3, stride = 1, padding = 2, dilation = 2, groups =4)
        self.branch2_4 = nn.Dropout(0.25)

        self.branch3_1 = BasicConv2d(in_channels, 32, kernel_size = 1, stride = 1)
        self.branch3_2 = Deconv2d(32, 32, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 4)
        self.branch3_3 = Deconv2d(32, 32, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 4)
        self.branch3_4 = nn.Dropout(0.25)


    def forward(self, x):
        #x_residual = x
        x_branch1 = self.branch1_1(x)
        x_branch1 = self.branch1_2(x_branch1)
        x_branch1 = self.branch1_3(x_branch1)
        x_branch1 = self.branch1_4(x_branch1)



        x_branch2 = self.branch2_1(x)
        x_branch2 = self.branch2_2(x_branch2)
        x_branch2 = self.branch2_3(x_branch2)
        x_branch2 = self.branch2_4(x_branch2)

        x_branch3 = self.branch3_1(x)
        x_branch3 = self.branch3_2(x_branch3)
        x_branch3 = self.branch3_3(x_branch3)
        x_branch3 = self.branch3_4(x_branch3)

        x = torch.cat([x_branch1, x_branch2, x_branch3], 1)
        


        return x

'''
class conv_M2(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_M2, self).__init__()
        self.branch1_1 = BasicConv2d(in_channels, 64, kernel_size = 1, stride = 1)
        self.branch1_2 = BasicConv2d(64, 128, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 4)
        self.branch1_3 = BasicConv2d(128, 128, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 4)
        self.branch1_4 = nn.Dropout(0.2)

        self.branch2_1 = BasicConv2d(in_channels, 64, kernel_size = 1, stride = 1)
        self.branch2_2 = BasicConv2d(64, 128, kernel_size = 3, stride = 1, padding = 2, dilation = 2, groups =4)
        self.branch2_3 = BasicConv2d(128, 128, kernel_size = 3, stride = 1, padding = 2, dilation = 2, groups =4)
        self.branch2_4 = nn.Dropout(0.2)

        self.branch3_1 = BasicConv2d(in_channels, 64, kernel_size = 1, stride = 1)
        self.branch3_2 = Deconv2d(64, 64, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 4)
        self.branch3_3 = Deconv2d(64, 64, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 4)
        self.branch3_4 = nn.Dropout(0.2)
        #self.branch4 = nn.Dropout(0.15)


    def forward(self, x):
        x_branch1 = self.branch1_1(x)
        x_branch1 = self.branch1_2(x_branch1)
        x_branch1 = self.branch1_3(x_branch1)
        x_branch1 = self.branch1_4(x_branch1)



        x_branch2 = self.branch2_1(x)
        x_branch2 = self.branch2_2(x_branch2)
        x_branch2 = self.branch2_3(x_branch2)
        x_branch2 = self.branch2_4(x_branch2)

        x_branch3 = self.branch3_1(x)
        x_branch3 = self.branch3_2(x_branch3)
        x_branch3 = self.branch3_3(x_branch3)
        x_branch3 = self.branch3_4(x_branch3)

        x = torch.cat([x_branch1, x_branch2, x_branch3], 1)
        #x = self.branch4(x)


        return x

class conv_M2_2(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_M2_2, self).__init__()
        self.branch1_1 = BasicConv2d(in_channels, 64, kernel_size = 1, stride = 1)
        self.branch1_2 = BasicConv2d(64, 128, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 4)
        self.branch1_3 = BasicConv2d(128, 160, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 4)
        self.branch1_4 = nn.Dropout(0.2)

        self.branch2_1 = BasicConv2d(in_channels, 64, kernel_size = 1, stride = 1)
        self.branch2_2 = BasicConv2d(64, 128, kernel_size = 3, stride = 1, padding = 2, dilation = 2, groups =4)
        self.branch2_3 = BasicConv2d(128, 160, kernel_size = 3, stride = 1, padding = 2, dilation = 2, groups =4)
        self.branch2_4 = nn.Dropout(0.2)

        self.branch3_1 = BasicConv2d(in_channels, 64, kernel_size = 1, stride = 1)
        self.branch3_2 = Deconv2d(64, 64, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 4)
        self.branch3_3 = Deconv2d(64, 128, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 4)
        self.branch3_4 = nn.Dropout(0.2)
        #self.branch4 = nn.Dropout(0.15)


    def forward(self, x):
        x_branch1 = self.branch1_1(x)
        x_branch1 = self.branch1_2(x_branch1)
        x_branch1 = self.branch1_3(x_branch1)
        x_branch1 = self.branch1_4(x_branch1)



        x_branch2 = self.branch2_1(x)
        x_branch2 = self.branch2_2(x_branch2)
        x_branch2 = self.branch2_3(x_branch2)
        x_branch2 = self.branch2_4(x_branch2)

        x_branch3 = self.branch3_1(x)
        x_branch3 = self.branch3_2(x_branch3)
        x_branch3 = self.branch3_3(x_branch3)
        x_branch3 = self.branch3_4(x_branch3)

        x = torch.cat([x_branch1, x_branch2, x_branch3], 1)
        #x = self.branch4(x)


        return x

'''
class conv_M2(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_M2, self).__init__()
        self.res = conv1x1(in_channels, 320)
        self.branch1_1 = BasicConv2d(in_channels, 96, kernel_size = 1, stride = 1)
        self.branch1_2_1 = Depthwise_conv2d(96, kernel_size = 3, stride = 1, padding = 1, dilation = 1)
        self.branch1_2_2 = Pointwise_conv2d(96, 128, stride = 1, padding = 1, dilation = 1)
        self.branch1_3_1 = Depthwise_conv2d(128, kernel_size = 3, stride = 1, padding = 1, dilation = 1)
        self.branch1_3_2 = Pointwise_conv2d(128, 128, stride = 1, padding = 1, dilation = 1)
        #att_list1 = [att.eca_layer(128)]
        #self.att_layer1 = nn.Sequential(*att_list1)
        self.branch1_4 = nn.Dropout(0.2)

        self.branch2_1 = BasicConv2d(in_channels, 96, kernel_size = 1, stride = 1)
        self.branch2_2_1 = Depthwise_conv2d(96,  kernel_size = 3, stride = 1, padding = 2, dilation = 2)
        self.branch2_2_2 = Pointwise_conv2d(96, 128, stride = 1, padding = 2, dilation = 2)
        self.branch2_3_1 = Depthwise_conv2d(128, kernel_size = 3, stride = 1, padding = 2, dilation = 2)
        self.branch2_3_2 = Pointwise_conv2d(128, 128, stride = 1, padding = 2, dilation = 2)
        #att_list2 = [att.eca_layer(128)]
        #self.att_layer2 = nn.Sequential(*att_list2)
        self.branch2_4 = nn.Dropout(0.2)

        self.branch3_1 = BasicConv2d(in_channels, 64, kernel_size = 1, stride = 1)
        self.branch3_2 = Deconv2d(64, 64, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 4)
        self.branch3_3 = Deconv2d(64, 64, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 4)
        #att_list3 = [att.eca_layer(64)]
        #self.att_layer3 = nn.Sequential(*att_list3)
        self.branch3_4 = nn.Dropout(0.2)
        self.dr = nn.Dropout(0.15)
        #self.branch4 = nn.Dropout(0.15)


    def forward(self, x):
        x_residual = self.res(x)
        x_branch1 = self.branch1_1(x)
        x_branch1 = self.branch1_2_1(x_branch1)
        x_branch1 = self.branch1_2_2(x_branch1)
        x_branch2 = self.branch2_1(x)
        x_branch2 = self.branch2_2_1(x_branch2)
        x_branch2 = self.branch2_2_2(x_branch2)
        x_branch2 = x_branch1 + x_branch2

        #x_branch1 = self.att_layer1(x_branch1)
        #x_branch1 = torch.mul(x_branch1_att, x_branch1)
        x_branch1 = self.branch1_3_1(x_branch1)
        x_branch1 = self.branch1_3_2(x_branch1)
        
        x_branch1 = self.branch1_4(x_branch1)
        #x_branch2 = self.att_layer2(x_branch2)
        #x_branch2 = torch.mul(x_branch2_att, x_branch2)
        x_branch2 = self.branch2_3_1(x_branch2)
        x_branch2 = self.branch2_3_2(x_branch2)
        
        x_branch2 = self.branch2_4(x_branch2)
        x_branch2 = x_branch1 + x_branch2

        x_branch3 = self.branch3_1(x)
        x_branch3 = self.branch3_2(x_branch3)
        #x_branch3 = self.att_layer3(x_branch3)
        #x_branch3 = torch.mul(x_branch3_att, x_branch3)
        x_branch3 = self.branch3_3(x_branch3)
        
        x_branch3 = self.branch3_4(x_branch3)

        #x = torch.cat([x_branch1, x_branch2, x_branch3], 1)
        x = torch.cat([x_branch1, x_branch2, x_branch3], 1)
        x = self.dr(x)
        return x + x_residual

class conv_M2_2(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_M2_2, self).__init__()
        self.res = conv1x1(in_channels, 448)
        self.branch1_1 = BasicConv2d(in_channels, 64, kernel_size = 1, stride = 1)
        self.branch1_2_1 = Depthwise_conv2d(64, kernel_size = 3, stride = 1, padding = 1, dilation = 1)
        self.branch1_2_2 = Pointwise_conv2d(64, 128, stride = 1, padding = 1, dilation = 1)
        self.branch1_3_1 = Depthwise_conv2d(128, kernel_size = 3, stride = 1, padding = 1, dilation = 1)
        self.branch1_3_2 = Pointwise_conv2d(128, 160, stride = 1, padding = 1, dilation = 1)
        self.branch1_4 = nn.Dropout(0.2)

        self.branch2_1 = BasicConv2d(in_channels, 64, kernel_size = 1, stride = 1)
        self.branch2_2_1 = Depthwise_conv2d(64, kernel_size = 3, stride = 1, padding = 2, dilation = 2)
        self.branch2_2_2 = Pointwise_conv2d(64, 128, stride = 1, padding = 2, dilation = 2)
        self.branch2_3_1 = Depthwise_conv2d(128, kernel_size = 3, stride = 1, padding = 2, dilation = 2)
        self.branch2_3_2 = Pointwise_conv2d(128, 160, stride = 1, padding = 2, dilation = 2)
        self.branch2_4 = nn.Dropout(0.2)

        self.branch3_1 = BasicConv2d(in_channels, 64, kernel_size = 1, stride = 1)
        self.branch3_2 = Deconv2d(64, 64, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 4)
        self.branch3_3 = Deconv2d(64, 128, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 4)
        self.branch3_4 = nn.Dropout(0.2)
        #self.branch4 = nn.Dropout(0.15)


    def forward(self, x):
        x_residual = self.res(x)
        #print(x_residual.size())
        x_branch1 = self.branch1_1(x)
        x_branch1 = self.branch1_2_1(x_branch1)
        x_branch1 = self.branch1_2_2(x_branch1)
        x_branch1 = self.branch1_3_1(x_branch1)
        x_branch1 = self.branch1_3_2(x_branch1)
        x_branch1 = self.branch1_4(x_branch1)



        x_branch2 = self.branch2_1(x)
        x_branch2 = self.branch2_2_1(x_branch2)
        x_branch2 = self.branch2_2_2(x_branch2)
        x_branch2 = self.branch2_3_1(x_branch2)
        x_branch2 = self.branch2_3_2(x_branch2)
        x_branch2 = self.branch2_4(x_branch2)

        x_branch3 = self.branch3_1(x)
        x_branch3 = self.branch3_2(x_branch3)
        x_branch3 = self.branch3_3(x_branch3)
        x_branch3 = self.branch3_4(x_branch3)

        #x = torch.cat([x_branch1, x_branch2, x_branch3], 1)
        x = torch.cat([x_branch1, x_branch2, x_branch3], 1)
        return x + x_residual


class conv_M3(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_M3, self).__init__()
        self.branch1_1 = BasicConv2d(in_channels, 144, kernel_size = 1, stride = 1)
        self.branch1_2 = BasicConv2d(144, 256, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 4)
        self.branch1_3 = BasicConv2d(256, 320, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 4)
        self.branch1_4 = nn.Dropout(0.2)

        self.branch2_1 = BasicConv2d(in_channels,144, kernel_size = 1, stride = 1)
        self.branch2_2 = BasicConv2d(144, 256, kernel_size = 3, stride = 1, padding = 2, dilation = 2, groups =4)
        self.branch2_3 = BasicConv2d(256, 320, kernel_size = 3, stride = 1, padding = 2, dilation = 2, groups =4)
        self.branch2_4 = nn.Dropout(0.2)

        self.branch3_1 = BasicConv2d(in_channels, 64, kernel_size = 1, stride = 1)
        self.branch3_2 = Deconv2d(64, 64, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 4)
        self.branch3_3 = Deconv2d(64, 64, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 4)
        self.branch3_4 = nn.Dropout(0.2)
        #self.branch4 = nn.Dropout(0.15)

    def forward(self, x):
        x_branch1 = self.branch1_1(x)
        x_branch1 = self.branch1_2(x_branch1)
        x_branch1 = self.branch1_3(x_branch1)
        x_branch1 = self.branch1_4(x_branch1)



        x_branch2 = self.branch2_1(x)
        x_branch2 = self.branch2_2(x_branch2)
        x_branch2 = self.branch2_3(x_branch2)
        x_branch2 = self.branch2_4(x_branch2)

        x_branch3 = self.branch3_1(x)
        x_branch3 = self.branch3_2(x_branch3)
        x_branch3 = self.branch3_3(x_branch3)
        x_branch3 = self.branch3_4(x_branch3)

        x = torch.cat([x_branch1, x_branch2, x_branch3], 1)
        #x = self.branch4(x)


        return x



class resnet18_multi(nn.Module):
    def __init__(self,use_att = False, block = BasicBlock, layers = [2, 2, 2, 2], num_classes = 2):
        self.inplanes = 64  
        self.use_att = use_att
        super(resnet18_multi, self).__init__()  
        
        #print(self.layer2.state_dict())
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,  
                               bias=False)  
        self.bn1 = nn.BatchNorm2d(64)  
        self.relu = nn.ReLU(inplace = True)  
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #self.maxpool_non_pad = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)  

        self.layer1 = self._make_layer(block, 64, layers[0])  
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  
        # self.avgpool = nn.AvgPool2d(14, stride=1) 
        # convm2_list = [conv_M2(128, 320)]
        if extracted_layer == "5_point":
            self.avgpool = nn.AvgPool2d(14, stride=1)
            convm2_list = [conv_M2(128, 320)]
        elif extracted_layer == "6_point":
            self.avgpool = nn.AvgPool2d(7, stride=1)
            convm2_list = [conv_M2(256, 320)]
        elif extracted_layer == "7_point":
            self.avgpool = nn.AvgPool2d(7, stride=1)
            convm2_list = [conv_M2(256, 320)]
        elif extracted_layer == "8_point":
            self.avgpool = nn.AvgPool2d(7, stride=1)
            convm2_list = [conv_M2(512, 320)]
        # att_list = [att.eca_layer(128)]
        # #tt_list = [GCT.GCT(128)]
        # att_list2 = [GCT.GCT(320)]


        #convm2_list_2 = [conv_M2_2(320, 448)]

        #convm3_list = [conv_M3(320, 704)]
        #convm3_list_2 = [conv_M3(704,704)]
        self.convm2_layer = nn.Sequential(*convm2_list)
        # self.att_layer = nn.Sequential(*att_list)
        # self.att_layer2 = nn.Sequential(*att_list2)

        #self.convm2_layer_2 = nn.Sequential(*convm2_list_2)

        #self.convm3_layer = nn.Sequential(*convm3_list)
        #self.convm3_layer_2 = nn.Sequential(*convm3_list_2)
        self.linear_test = BasicConv2d(320, 256, kernel_size = 1, stride =1)

        for m in self.modules():
           if isinstance(m, nn.Conv2d):  
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels  
                #m.weight.data.normal_(0, math.sqrt(2. / n))  
           elif isinstance(m, nn.BatchNorm2d):  
                m.weight.data.fill_(1)  
                m.bias.data.zero_()   #深度優先遍歷模式  

        #print(self.modules())
        #self.linear_test_2 = BasicConv2d(704, 512, kernel_size = 1, stride =1)
        '''
        for m in self.modules():
           if isinstance(m, nn.Conv2d):  
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels  
                m.weight.data.normal_(0, math.sqrt(2. / n))  
           elif isinstance(m, nn.BatchNorm2d):  
                m.weight.data.fill_(1)  
                m.bias.data.zero_()   #深度優先遍歷模式  
        '''

 


    def _make_layer(self, block, planes, blocks, stride=1):  
        downsample = None  
        if stride != 1 or self.inplanes != planes * block.expansion:  
            downsample = nn.Sequential(  
                nn.Conv2d(self.inplanes, planes * block.expansion,  
                          kernel_size=1, stride=stride, bias=False),  
                nn.BatchNorm2d(planes * block.expansion),  
            )  

        layers = []  
        layers.append(block(self.inplanes, planes, stride, downsample))  
        self.inplanes = planes * block.expansion  
        for i in range(1, blocks):  
            layers.append(block(self.inplanes, planes))  

        return nn.Sequential(*layers)  

    def forward(self, x, test_flag):
        if test_flag:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool1(x)
            x = self.layer1(x)
            x = self.layer2(x)
            if extracted_layer == '6_point':
                x = self.layer3[0](x)
            if extracted_layer == '7_point':
                x = self.layer3(x)
            if extracted_layer == '8_point':
                x = self.layer3(x)
                x = self.layer4[0](x)
            
        if self.use_att:
            res = x
            x = self.att_layer(x)
            x = x + res

        
        
            
        x = self.convm2_layer(x)

        if self.use_att:
            res2 = x
            x = self.att_layer2(x)
            x = x + res2



       
        #x = self.convm2_layer_2(x)
        # x = self.maxpool1(x)
        if extracted_layer != "8_point":
            x = self.maxpool1(x)
        
        #print(x.size())
        x = self.linear_test(x)

        

        #x = self.layer3(x)
        

        #x = self.layer4(x)
        
        #x = self.convm3_layer(x)
        #x = self.convm3_layer_2(x)
        #x = self.maxpool1(x)
        #x = self.linear_test_2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)


        return x


class ResNet18Fc(nn.Module):
    def __init__(self):
        super(ResNet18Fc, self).__init__() # 繼承父類，MRO(Method Resolution Order)
        model_resnet18 = models.resnet18(pretrained=True) # If pretrained=True, returns a model pre-trained on ImageNet

        self.conv1 = model_resnet18.conv1
        self.bn1 = model_resnet18.bn1
        self.relu = model_resnet18.relu
        self.maxpool = model_resnet18.maxpool

        self.layer1 = model_resnet18.layer1

        self.layer2 = model_resnet18.layer2

        self.layer3 = model_resnet18.layer3

        self.layer4 = model_resnet18.layer4

        self.avgpool = model_resnet18.avgpool
        self.__in_features = model_resnet18.fc.in_features # 最後一層fully connected 的 input_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        x = self.avgpool(x)

        #x = x.view(x.size(0), -1) 
        return x

    def output_num(self):
        return self.__in_features


network_dict = {
                # "alexnet": AlexNetFc,
                "resnet18_multi_new": resnet18_multi,
                "resnet18": ResNet18Fc
                # "resnet34": ResNet34Fc,
                # "resnet50": ResNet50Fc,
                # "resnet101": ResNet101Fc,
                # "resnet152": ResNet152Fc
                }

