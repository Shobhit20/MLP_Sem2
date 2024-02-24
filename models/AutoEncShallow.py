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
class Autoencoder(nn.Module):
	'''Class defining the basic encoder-decoder model with shallow encoder and decoder'''

	def __init__(self):
		'''Initialises the model separate encoder and decoder block'''
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
		'''
		Computes a forward iteration in the model

		Args:
			x: the input image to be fed through the model
		
		Returns:
			x: the output image after being fed through the model
		'''
		x = self.encoder(x)
		x = self.decoder(x)
		return x