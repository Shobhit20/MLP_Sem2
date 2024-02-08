import os
import numpy as np
import matplotlib.pyplot as plt
import time
import random
import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF

import cv2
from sklearn.model_selection import train_test_split

# ----------------------- Basic Autoencoder Model ----------------------- #
# Defining the autoencoder architecture
class Autoencoder(nn.Module):
	def __init__(self):
		super(Autoencoder, self).__init__()
		self.encoder = nn.Sequential(
			nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2)
		)
		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
			nn.ReLU(),
			nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
			nn.Sigmoid()
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
		original_images =[]
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
			# input_image = dataloader.dataset[random_indices[i]][0].cpu().permute(1, 2, 0)
			input_image = original_images[i].cpu().squeeze().permute(1, 2, 0)
			output_image = generated_images[i].cpu().squeeze().permute(1, 2, 0)

			axes[0, i].imshow(input_image)
			axes[0, i].set_title('Input Image')
			axes[0, i].axis('off')

			axes[1, i].imshow(output_image)
			axes[1, i].set_title('Output Image')
			axes[1, i].axis('off')

		plt.show()
		return generated_images
	
	def create_output(self, model, input_image, device='cpu'):
		model.eval()
		with torch.no_grad():
			input_image = input_image.to(device)
			output = model(input_image)
		
		input_image = input_image.cpu().permute(1, 2, 0)
		output_image = output.cpu().squeeze().permute(1, 2, 0)

		# Display input and output images side by side
		fig, axes = plt.subplots(1, 2, figsize=(8, 4))

		axes[0].imshow(input_image)
		axes[0].set_title('Input Image'); axes[0].axis('off')

		axes[1].imshow(output_image)
		axes[1].set_title('Output Image'); axes[1].axis('off')
		plt.show()

		return 0