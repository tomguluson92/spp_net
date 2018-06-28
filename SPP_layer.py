# coding: UTF-8

'''
    @date: 2018/6/27
    @target: 目标很简单，是实现SPP Layer，以便于不同size的input最后归一化
    @author: samuel ko
    @attention: 1 此层不需要反向传播，因此不用重写backward()
                2 需要根据自己模型输入SPP层之前的channel计算一个输出特征
'''

import math
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models, datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# Modified_SPPLayer是解决到最后的时候，
# 由于pad的size会大于kernel_size/2，因此pooling会报错
# pad should be smaller than half of kernel size,
# but got padW = 1, padH = 3, kW = 1, kH = 2 at /pytorch/aten/src/THNN/generic/SpatialDilatedMaxPooling.c:35
class SPPLayer(torch.nn.Module):

    # 定义Layer需要的额外参数（除Tensor以外的）
    def __init__(self, num_levels, pool_type='max_pool'):
        super(SPPLayer, self).__init__()

        self.num_levels = num_levels
        self.pool_type = pool_type

    # forward()的参数只能是Tensor(>=0.4.0) Variable(< 0.4.0)
    def forward(self, x):
        # num:样本数量 c:通道数 h:高 w:宽
        # num: the number of samples
        # c: the number of channels
        # h: height
        # w: width
        num, c, h, w = x.size()
        print('original', x.size())
        level = 1
        #         print(x.size())
        for i in range(self.num_levels):
            level <<= 1

            '''
            The equation is explained on the following site:
            http://www.cnblogs.com/marsggbo/p/8572846.html#autoid-0-0-0
            '''
            kernel_size = (math.ceil(h / level), math.ceil(w / level))  # kernel_size = (h, w)
            padding = (
                math.floor((kernel_size[0] * level - h + 1) / 2), math.floor((kernel_size[1] * level - w + 1) / 2))

            # update input data with padding
            #  class torch.nn.ZeroPad2d(padding)[source]
            #
            #     Pads the input tensor boundaries with zero.
            #
            #     For N`d-padding, use :func:`torch.nn.functional.pad().
            #     Parameters:	padding (int, tuple) – the size of the padding. If is int, uses the same padding in all boundaries.
            # If a 4-tuple, uses (paddingLeft, paddingRight, paddingTop, paddingBottom)
            zero_pad = torch.nn.ZeroPad2d((padding[1], padding[1], padding[0], padding[0]))
            x_new = zero_pad(x)

            # update kernel and stride
            h_new, w_new = x_new.size()[2:]

            kernel_size = (math.ceil(h_new / level), math.ceil(w_new / level))
            stride = (math.floor(h_new / level), math.floor(w_new / level))

            # 选择池化方式
            if self.pool_type == 'max_pool':
                tensor = F.max_pool2d(x_new, kernel_size=kernel_size, stride=stride).view(num, -1)
            elif self.pool_type == 'avg_pool':
                tensor = F.avg_pool2d(x_new, kernel_size=kernel_size, stride=stride).view(num, -1)

            # 展开、拼接
            if (i == 0):
                x_flatten = tensor.view(num, -1)
            else:
                x_flatten = torch.cat((x_flatten, tensor.view(num, -1)), 1)

        return x_flatten
