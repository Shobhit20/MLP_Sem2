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
import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        # ------------------------------- Encoder ------------------------------- #
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
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
        
        # ------------------------------- Decoder ------------------------------- #
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Upsample(scale_factor=2, align_corners=True),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Upsample(scale_factor=2, align_corners=True),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Upsample(scale_factor=2, align_corners=True),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 1, 7, padding=1)
        )
    
    def forward(self, x):
        input = x.clone()

        # ------------------------------- Encoder ------------------------------- #
        e0 = self.encoder[0](x)
        e1 = self.encoder[1](e0)
        e2 = self.encoder[2](e1) + e0
        e3 = self.encoder[3](e2) # Maxpooling

        e4 = self.encoder[4](e3)
        e5 = self.encoder[5](e4) + e3
        e6 = self.encoder[6](e5) # Maxpooling

        e7 = self.encoder[7](e6)
        e8 = self.encoder[8](e7) + e6
        e9 = self.encoder[9](e8) # MaxPooling
        
        # ------------------------------- Decoder ------------------------------- #
        d0 = self.decoder[0](e9)
        d1 = self.decoder[1](d0) # Upsampling
        d1_skip = torch.cat((e8, d1), dim=1) # Concatenating from Encoder

        d2 = self.decoder[2](d1_skip)
        d3 = self.decoder[3](d2) + d1_skip
        d4 = self.decoder[4](d3) # Upsampling
        d4_skip = torch.cat((e5, d4), dim=1) # Concatenating from Encoder

        d5 = self.decoder[5](d4_skip)
        d6 = self.decoder[6](d5) + d4_skip
        d7 = self.decoder[7](d6) # Upsampling
        d7_skip = torch.cat((e2, d7), dim=1) # Concatenating from Encoder

        d8 = self.decoder[8](d7_skip)
        d9 = self.decoder[9](d8) + d7_skip
        d10 = self.decoder[10](d9) + input
        
        return d10

model = Autoencoder()

# Print the model architecture
print(model)

dummy_input = torch.randn(1, 3, 256, 256)  # Batch size 1, 3 channels, 256x256 size

# Pass the input through the model to get the output
output = model(dummy_input)