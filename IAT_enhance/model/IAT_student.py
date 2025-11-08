import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse
from torchvision.models import vgg16
from collections import defaultdict

from data_loaders.lol import lowlight_loader
from model.IAT_main import IAT
from IQA_pytorch import SSIM
from utils import PSNR, validation, LossNetwork
from torch.utils.tensorboard import SummaryWriter



class CBlock_bn(nn.Module):
    """LayerNorm → BatchNorm (빠름!)"""
    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim)
    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return out + identity


# ============================================================================
# Student Model (Fast Inference, High Quality)
# ============================================================================

class IAT_Student_BN(nn.Module):
    """
    1) Multi-level Distillation
    2) CBlock_bn -> layer norm to batch norm for sppen
    3) shared conv
    """
    def __init__(self, in_dim=3, dim=16):
        super().__init__()
        
        # Shared initial conv
        self.shared_conv1 = nn.Conv2d(in_dim, dim, 3, padding=1)
        self.shared_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        # Local branches (CBlock_bn 2개 - 빠름!)
        blocks1 = [CBlock_bn(dim, drop_path=0.05), 
                  CBlock_bn(dim, drop_path=0.1)]
        blocks2 = [CBlock_bn(dim, drop_path=0.05),
                  CBlock_bn(dim, drop_path=0.1)]
        
        self.mul_blocks = nn.Sequential(*blocks1)
        self.add_blocks = nn.Sequential(*blocks2)
        self.mul_end = nn.Sequential(nn.Conv2d(dim, 3, 3, 1, 1), nn.ReLU())
        self.add_end = nn.Sequential(nn.Conv2d(dim, 3, 3, 1, 1), nn.Tanh())
        
        # Global branch
        self.global_refine = nn.Sequential(
            nn.AdaptiveAvgPool2d((64, 64)),
            nn.Conv2d(dim*3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        self.gamma_head = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())
        self.color_head = nn.Linear(32, 9)
        
        self.register_buffer('identity', torch.eye(3))
    
    def apply_color(self, image, ccm):
        b, c, h, w = image.shape
        image = image.view(b, c, -1)
        image = torch.bmm(ccm, image)
        image = image.view(b, c, h, w)
        return torch.clamp(image, 1e-8, 1.0)
    
    def forward(self, img_low, return_intermediates=False):
        # Shared feature
        shared_feature = self.shared_relu(self.shared_conv1(img_low))
        
        # Local processing
        mul_feat = self.mul_blocks(shared_feature) + shared_feature
        add_feat = self.add_blocks(shared_feature) + shared_feature
        
        mul = self.mul_end(mul_feat)
        add = self.add_end(add_feat)
        img_local = img_low.mul(mul).add(add)
        
        # Global
        combined_feat = torch.cat([shared_feature, mul_feat, add_feat], dim=1)
        global_feature = self.global_refine(combined_feat)
        
        gamma = self.gamma_head(global_feature)
        color = self.color_head(global_feature).view(-1, 3, 3)
        color = color + self.identity.unsqueeze(0)
        
        # Final
        img_high = self.apply_color(img_local, color)
        img_high = img_high ** gamma.view(-1, 1, 1, 1)
        
        if return_intermediates:
            return {
                'mul': mul,
                'add': add,
                'img_local': img_local,
                'gamma': gamma,
                'color': color,
                'img_high': img_high,
                'shared_feature': shared_feature,
                'mul_feat': mul_feat,
                'add_feat': add_feat,
                'combined_feat': combined_feat,
                'global_feature': global_feature
            }
        
        return mul, add, img_high
