# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 20:47:39 2022

@author: Lawrence
"""

import torch
import torchvision
from torch import nn, Tensor
import torchvision.transforms.functional as F


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, masks):
        for t in self.transforms:
            image, masks = t(image, masks)
        return image, masks 
    
class ToTensors(nn.Module):
    """
    Converts the 
    def forward




