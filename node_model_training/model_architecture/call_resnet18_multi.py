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
from . import backbone_multi as ml




def load_resnet18_multi():
    #載入pretrained model resnet18
    resnet18 = models.resnet18(pretrained = True)
    resnet18_new = ml.resnet18_multi(block = ml.BasicBlock, layers = [2, 2, 2, 2])
    #print(resnet18_new)
    
    #print(resnet18) # 3 4 6 3 分別表示layer1 2 3 4 中 bottleneck 的數量(以resnet 50為例) 
    #print(resnet18_new)

    #讀取引數
    pretrained_dict = resnet18.state_dict()
    #print(pretrained_dict)
    model_dict = resnet18_new.state_dict()
    #print(resnet18)
    #(resnet18_new)



    #將pretrained_dict 裡不屬於model_dict 的鍵剔除掉
    pretrained_dict = {k : v for k, v in pretrained_dict.items() if k in model_dict}

    #更新現有的model_dict
    model_dict.update(pretrained_dict)

    #載入真正需要的state_dict
    resnet18_new.load_state_dict(model_dict)
    #print (resnet18_new.layer3.state_dict())
    #resnet18_new.cuda()
    #summary(resnet18_new, (8, 3, 224, 224))


    #載入真正需要的state_dict
    return resnet18_new

#load_resnet18_multi()
