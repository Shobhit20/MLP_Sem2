# creat a pytroch encoder-decoder model for the given architecture
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

class SkidNet(nn.Module):
    def __init__(self):
        super(SkidNet, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        # ------------------------------- Encoder ------------------------------- #
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), 
            nn.BatchNorm2d(32), 
            nn.Conv2d(32, 64, 3, padding=1), 
            nn.BatchNorm2d(64), 
            nn.Conv2d(64, 32, 3, padding=1), 
            nn.BatchNorm2d(32), 
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), 
            nn.BatchNorm2d(64), 
            nn.Conv2d(64, 32, 3, padding=1), 
            nn.BatchNorm2d(32), 
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, 3, padding=1), 
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2)
        )
        self.mediator = nn.Conv2d(32, 32, 3, padding=1)

        # ------------------------------- Decoder ------------------------------- #
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            # major block 1
            # minor block 1
            nn.Conv2d(64, 64, 3, padding=1), # input filters change on the basis of addition/concat
            nn.BatchNorm2d(64),
            # minor block 2
            nn.Conv2d(64, 32, 3, padding=1), 
            nn.BatchNorm2d(32),
            nn.Upsample(scale_factor=2),
            
            # major block 2
            # minor block 1
            nn.Conv2d(64, 64, 3, padding=1), # input filters change on the basis of addition/concat
            nn.BatchNorm2d(64),
            # minor block 2
            nn.Conv2d(64, 32, 3, padding=1), 
            nn.BatchNorm2d(32), 
            nn.Upsample(scale_factor=2),

            # major block 3
            # minor block 1
            nn.Conv2d(64, 64, 3, padding=1), # input filters change on the basis of addition/concat
            nn.BatchNorm2d(64),
            # minor block 2
            nn.Conv2d(64, 32, 3, padding=1), 
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 1, 7, padding=3)
        )
    
    def forward(self, x):
        input = x.clone()

        # ------------------------------- Encoder ------------------------------- #
        e0 = self.encoder[0](x)
        e1 = self.relu(self.encoder[1](e0))
        e2 = self.encoder[2](e1)
        e3 = self.relu(self.encoder[3](e2))
        e4 = self.encoder[4](e3)
        e5 = self.relu(self.encoder[5](e4)) + e1
        e6 = self.encoder[6](e5) # Maxpooling
        e7 = self.encoder[7](e6)
        e8 = self.relu(self.encoder[8](e7))
        e9 = self.encoder[9](e8)
        e10 = self.relu(self.encoder[10](e9)) + e6
        e11 = self.encoder[11](e10) # Maxpooling
        e12 = self.encoder[12](e11)
        e13 = self.relu(self.encoder[13](e12))
        e14 = self.encoder[14](e13)
        e15 = self.relu(self.encoder[15](e14)) + e11
        e16 = self.encoder[16](e15) # Maxpooling
        

        med_l1 = self.relu(self.mediator(e16))
        # ------------------------------- Decoder ------------------------------- #
        # deconv major block 1
        d0 = self.decoder[0](med_l1)
        concat1 = torch.cat((d0, e15), dim=1)
        d1 = self.decoder[1](concat1) 
        d2 = self.relu(self.decoder[2](d1))
        d3 = self.decoder[3](d2) 
        d4 = self.relu(self.decoder[4](d3)) + d0
        d5 = self.decoder[5](d4)
        concat2 = torch.cat((d5, e10), dim=1)

        # deconv major block 2
        d6 = self.decoder[6](concat2) 
        d7 = self.relu(self.decoder[7](d6))
        d8 = self.decoder[8](d7) 
        d9 = self.relu(self.decoder[9](d8)) + d5
        d10 = self.decoder[10](d9)
        concat3 = torch.cat((d10, e5), dim=1)

        # deconv major block 3
        d11 = self.decoder[11](concat3)
        d12 = self.relu(self.decoder[12](d11))
        d13 = self.decoder[13](d12)
        d14 = self.relu(self.decoder[14](d13)) + d10

        d15 = self.decoder[15](d14) + input
        return d15