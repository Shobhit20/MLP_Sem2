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
    
    def evaluate_model(self, model, dataloader, device='cpu'):
        model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for data in dataloader:
                images, _ = data
                images = images.to(device)

                # Forward pass
                outputs = model(images)

                # Calculate reconstruction loss (MSE)
                loss = nn.functional.mse_loss(outputs, images)

                total_loss += loss.item()
                num_batches += 1

        average_loss = total_loss / num_batches
        return average_loss
	
    def generate_images(self, model, dataloader, n, device='cpu'):
        model.eval()
        original_images = []
        generated_images = []
        random_indices = random.sample(range(dataloader.batch_size), n)

        with torch.no_grad():
            for i, data in enumerate(dataloader):
                if i in random_indices:
                    img, _ = data
                    img = img.to(device)
                    output = model(img)
                    original_images.append(img[0])
                    generated_images.append(output[0])
        print(f'Sample Images Selected {random_indices}')

        fig, axes = plt.subplots(2, n, figsize=(3 * n, 8))
        for i in range(n):

            # For Grayscale Images (Single Channel)
            input_image = original_images[i].cpu().squeeze().numpy()
            output_image = generated_images[i].cpu().squeeze().numpy()

            # axes[0, i].imshow(input_image) # For Colour Images
            axes[0, i].imshow(input_image, cmap = 'gray')
            axes[0, i].set_title('Input Image')
            axes[0, i].axis('off')

            # axes[1, i].imshow(input_image) # For Colour Images
            axes[1, i].imshow(output_image, cmap = 'gray')
            axes[1, i].set_title('Output Image')
            axes[1, i].axis('off')

        plt.show()
        return generated_images
	
    def create_output(self, model, input_image, device='cpu'):
        model.eval()
        with torch.no_grad():
            input_image = input_image.to(device)
            output = model(input_image.unsqueeze(1))

        # input_image_pil = TF.to_pil_image(input_image.cpu().squeeze())
        # output_image_pil = TF.to_pil_image(output.cpu().squeeze())

        input_image = input_image.cpu().squeeze().numpy()
        output_image = output.cpu().squeeze().numpy()

        # Display input and output images side by side
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        axes[0].imshow(input_image)
        axes[0].set_title('Input Image'); axes[0].axis('off')

        axes[1].imshow(output_image)
        axes[1].set_title('Output Image'); axes[1].axis('off')
        plt.show()

        return 0
    
# Random Testing
# model = AutoencoderWithoutSkip()
# dummy_input = torch.randn(1, 1, 256, 256)  # Batch size 1, 3 channels, 256x256 size

# # Pass the input through the model to get the output
# output = model(dummy_input)

# # Check shape of output
# print(output.shape)
