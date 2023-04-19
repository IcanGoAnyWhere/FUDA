# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import DA_IMAGE

from torch.autograd import Function

def channel_entropy(feature):
    near_0 = 1e-10
    input_soft = torch.softmax(feature, dim=1)
    predict_map = torch.log(input_soft+near_0)
    entropy_cha = -predict_map * input_soft
    entropy_cha = torch.sum(entropy_cha, dim=1).detach()

    return entropy_cha

class GRL(Function):

    @staticmethod
    def forward(ctx, input):
        ctx.alpha = torch.tensor(0.1, requires_grad=True)
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_outputs):
        output = grad_outputs.neg() * ctx.alpha
        return output


@DA_IMAGE.register_module()
class FUA(nn.Module):
    def __init__(self):
        super(FUA, self).__init__()
        self.conv1 = nn.Conv2d(272, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0, bias=False)

        self.conv_en1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding='same', bias=False)
        self.bn_en1 = nn.BatchNorm2d(128)
        self.conv_en2 = nn.Conv2d(128, 16, kernel_size=3, stride=1, padding='same', bias=False)
        self.bn_en2 = nn.BatchNorm2d(16)

    def forward(self, feature, domain):
        feature_GRL = GRL.apply(feature)
        cha_en = channel_entropy(feature_GRL)

        # uncertainty_map = F.relu(self.bn_en1(self.conv_en1(cha_en.unsqueeze(0))))
        uncertainty_map = F.relu(self.bn_en1(self.conv_en1(feature_GRL)))
        uncertainty_map = F.relu(self.bn_en2(self.conv_en2(uncertainty_map)))

        fea_with_uncer = [feature_GRL, uncertainty_map]
        fea_with_uncer = torch.cat(fea_with_uncer, dim=1)

        x = F.relu(self.bn1(self.conv1(fea_with_uncer)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.conv3(x))

        out = F.sigmoid(x)

        domain_label = domain * torch.ones_like(out)
        near_0 = 1e-10
        loss1 = -domain_label * torch.log(out+near_0)

        loss2 = -(torch.ones_like(out)-domain_label) * torch.log(torch.ones_like(out) - out + near_0)

        loss = loss1 + loss2

        loss = torch.mean(loss)

        return loss, uncertainty_map




