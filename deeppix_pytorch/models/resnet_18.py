#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 12:03:40 2021

@author: vijay
"""

import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F

class Resnet_18(nn.Module):
    """
    The class defining Deep Pixel-wise Binary Supervision for Face Presentation Attack
    """

    def __init__(self, pretrained=True):
        super(Resnet_18, self).__init__()
        resnet = models.resnet18(pretrained=pretrained)
        features = list(resnet.children())
        self.backbone = nn.Sequential(*features[0:9])
        
        self.classi = nn.Sequential(
                      nn.Linear(512, 256),  
                      nn.ReLU(), 
                      nn.Dropout(0.3),
                      nn.Linear(256, 2))
       
    def forward(self, x):
        bb = self.backbone(x)
        #print(bb.shape)
        op = self.classi(bb.view(-1, 512))
        
        return op
