# coding: UTF-8


import os
import math
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models, datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from SPP_layer import SPPLayer


def calc_auto(num, channels):
    lst = [1, 2, 4, 8, 16, 32]
    return sum(map(lambda x: x ** 2, lst[:num])) * channels


# 将Resnet模型fc层前面的avgpool改为SPP

model_resnet = models.resnet18(pretrained=True)
model_resnet.avgpool = SPPLayer(3)
model_resnet.fc = nn.Linear(calc_auto(3, 512), 1000)

img_to_tensor = transforms.ToTensor()

if __name__ == '__main__':
    img = Image.open('xxx.png')
    tensor = img_to_tensor(img)
    tensor.unsqueeze_(0)

    result = model_resnet(tensor)
    # 预测结果
    print(np.argmax(result.detach().numpy()[0]))
