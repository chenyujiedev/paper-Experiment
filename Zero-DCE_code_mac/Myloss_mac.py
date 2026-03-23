# -*- coding: utf-8 -*-
# @Time    : 2026/3/7 17:39
# @Author  : 陈宇杰（CHEN YUJIE）
# @File    : Myloss_mac.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vgg import vgg16


class L_color(nn.Module):
    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x):
        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr - mg, 2)
        Drb = torch.pow(mr - mb, 2)
        Dgb = torch.pow(mb - mg, 2)
        k = torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2), 0.5)
        return k


class L_spa(nn.Module):
    def __init__(self):
        super(L_spa, self).__init__()

        kernel_left = torch.tensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        kernel_right = torch.tensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        kernel_up = torch.tensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        kernel_down = torch.tensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        self.register_buffer("weight_left", kernel_left)
        self.register_buffer("weight_right", kernel_right)
        self.register_buffer("weight_up", kernel_up)
        self.register_buffer("weight_down", kernel_down)

        self.pool = nn.AvgPool2d(4)

    def forward(self, org, enhance):
        device = org.device

        org_mean = torch.mean(org, 1, keepdim=True)
        enhance_mean = torch.mean(enhance, 1, keepdim=True)

        org_pool = self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)

        val_1 = torch.tensor([1.0], device=device)
        val_03 = torch.tensor([0.3], device=device)
        val_0 = torch.tensor([0.0], device=device)
        val_05 = torch.tensor([0.5], device=device)

        _ = torch.max(val_1 + 10000 * torch.min(org_pool - val_03, val_0), val_05)
        _ = torch.mul(torch.sign(enhance_pool - val_05), enhance_pool - org_pool)

        D_org_left = F.conv2d(org_pool, self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool, self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool, self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool, self.weight_down, padding=1)

        D_enhance_left = F.conv2d(enhance_pool, self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool, self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool, self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool, self.weight_down, padding=1)

        D_left = torch.pow(D_org_left - D_enhance_left, 2)
        D_right = torch.pow(D_org_right - D_enhance_right, 2)
        D_up = torch.pow(D_org_up - D_enhance_up, 2)
        D_down = torch.pow(D_org_down - D_enhance_down, 2)

        E = D_left + D_right + D_up + D_down
        return E


class L_exp(nn.Module):
    def __init__(self, patch_size, mean_val):
        super(L_exp, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val

    def forward(self, x):
        device = x.device
        x = torch.mean(x, 1, keepdim=True)
        mean = self.pool(x)
        target = torch.tensor([self.mean_val], device=device)
        d = torch.mean(torch.pow(mean - target, 2))
        return d


class L_TV(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(L_TV, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size


class Sa_Loss(nn.Module):
    def __init__(self):
        super(Sa_Loss, self).__init__()

    def forward(self, x):
        r, g, b = torch.split(x, 1, dim=1)
        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Dr = r - mr
        Dg = g - mg
        Db = b - mb
        k = torch.pow(torch.pow(Dr, 2) + torch.pow(Db, 2) + torch.pow(Dg, 2), 0.5)
        k = torch.mean(k)
        return k


class perception_loss(nn.Module):
    def __init__(self):
        super(perception_loss, self).__init__()
        features = vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h = self.to_relu_2_2(h)
        h = self.to_relu_3_3(h)
        h = self.to_relu_4_3(h)
        return h


class DarkRegionExposureLoss(nn.Module):
    """
    暗区域加权曝光损失
    参数：
        patch_size: 局部平均池化大小
        target_mean: 暗区增强后的目标曝光值
        tau: 暗区阈值
        gamma: 暗区强调系数
    """
    def __init__(self, patch_size=16, target_mean=0.6, tau=0.4, gamma=2.0):
        super(DarkRegionExposureLoss, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.target_mean = target_mean
        self.tau = tau
        self.gamma = gamma

    def forward(self, input_img, enhanced_img):
        input_lum = (
            0.299 * input_img[:, 0:1, :, :] +
            0.587 * input_img[:, 1:2, :, :] +
            0.114 * input_img[:, 2:3, :, :]
        )

        enhanced_lum = (
            0.299 * enhanced_img[:, 0:1, :, :] +
            0.587 * enhanced_img[:, 1:2, :, :] +
            0.114 * enhanced_img[:, 2:3, :, :]
        )

        dark_mask = torch.clamp((self.tau - input_lum) / self.tau, min=0.0, max=1.0)
        dark_mask = torch.pow(dark_mask, self.gamma)

        mean_enhanced = self.pool(enhanced_lum)
        mean_dark = self.pool(dark_mask)

        target = torch.tensor([self.target_mean], device=input_img.device)

        loss = torch.mean(mean_dark * torch.pow(mean_enhanced - target, 2))
        return loss