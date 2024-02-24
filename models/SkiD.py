# create an encoder-decoder pytorch model code for the given architecture with skip connections
''' 
# Encoder
input(256, 256)
l1: Conv2D(3x3, 32 Filters) + Batch Norm + Relu
l2: Conv2D(3x3, 64 Filters) + Batch Norm + Relu
l3: Conv2D(3x3, 32 Filters) + Batch Norm + Relu + l1
l4: MaxPooling2D(2x2)

l5: Conv2D(3x3, 64 Filters) + Batch Norm + Relu
l6: Conv2D(3x3, 32 Filters) + Batch Norm + Relu + l4
l7: MaxPooling2D(2x2)

l8: Conv2D(3x3, 64 Filters) + Batch Norm + Relu
l9: Conv2D(3x3, 32 Filters) + Batch Norm + Relu + l7
l10: MaxPooling2D(2x2)

# Decoder
l11: Conv2D(3x3, 32 Filters) + Batch Norm + Relu
l12: UpSampling2D(2x2) + concact(l9)

l13: Conv2D(3x3, 64 Filters) + Batch Norm + Relu
l14: Conv2D(3x3, 32 Filters) + Batch Norm + Relu + l12
l15: UpSampling2D(2x2) + concat(l6)

l16: Conv2D(3x3, 64 Filters) + Batch Norm + Relu
l17: Conv2D(3x3, 32 Filters) + Batch Norm + Relu + l15
l18: UpSampling2D(2x2) + concat(l3)

l19: Conv2D(3x3, 64 Filters) + Batch Norm + Relu
l20: Conv2D(3x3, 32 Filters) + Batch Norm + Relu + l18
l21: Conv2D(7x7, 1 Filter) + input(256, 256)

output(256, 256)
'''
import os
import numpy as np
import time
import random
import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF

class AutoencoderWithoutSkip(nn.Module):
    def __init__(self):
        super(AutoencoderWithoutSkip, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 1, 7, padding=3),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x