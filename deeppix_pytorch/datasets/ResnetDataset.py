#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 14:09:49 2021

@author: vijay
"""

import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms


class ResnetDataset(Dataset):
    """ A data loader for Pixel Wise Deep Supervision PAD where samples are organized in this way

    Args:
        root_dir (string): Root directory path
        csv_file (string): csv file to dataset annotation
        map_size (int): size of pixel-wise binary supervision map. The paper uses map_size=14
        transform: A function/transform that takes in a sample and returns a transformed version
        smoothing (bool): Use label smoothing
    """

    def __init__(self, root_dir, csv_file, map_size, transform=None, smoothing=True):
        super().__init__()
        self.root_dir = root_dir
        self.data = pd.read_csv(os.path.join('', csv_file))
        self.map_size = map_size
        self.transform = transform
        
       


    def __getitem__(self, index):
        """ Get image, output map and label for a given index
        Args:
            index (int): index of image
        Returns:
            img (PIL Image): 
            mask: output map (14x14)
            label: 1 (genuine), 0 (fake) 
        """
        img_name = self.data.iloc[index, 0]
        img_name = os.path.join(self.root_dir, img_name)
        img = Image.open(img_name)

        label = self.data.iloc[index, 1]
        
        if self.transform:
            img = self.transform(img)

        return img, label


    def __len__(self):
        return len(self.data)
