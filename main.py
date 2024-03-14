import time
import random
import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from utility.utils import *
from utility.noise_functions import *

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from torchvision import utils

from models.AutoEncShallow import *
from models.SkiD import *
from models.SkiDwithSkip import *
from models.SkiDwithSkipUnet import *
from models.SuperMRI import *

# Initialize the autoencoder
model = SkidNet()
print("The number of parameters in the model: ", sum(p.numel() for p in model.parameters()))

data_dir = 'data/'
batch_size = 32
train_loader, test_loader = loadData(data_dir, batch_size, test_size=0.2, color='gray', noise=True)
print('Data Loading Complete!')
# showImages(train_loader, 5)

# Move the model to GPU
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Device: {device}\n')
model.to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss() # nn.functional.binary_cross_entropy_with_logits
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the autoencoder
check_loss = 999
to_train = 0
num_epochs = 10
if to_train:
	for epoch in range(num_epochs):
		start = time.time()
		for i, mod in enumerate(tqdm.tqdm(train_loader, total = len(train_loader))):
			modif, actual = mod
			optimizer.zero_grad()

			output = model(modif)
			loss = criterion(output, actual)

			loss.backward()
			optimizer.step()

		if loss.item() < check_loss:
			check_loss = loss.item()
			print(f'Saving New Best Model')
			torch.save(model.state_dict(), 'final/SkidNet_2.pth')

		print(f'Time taken for epoch: {time.time() - start}')
		print(f'Epoch [{epoch + 1}/{num_epochs}]  |  Loss: {loss.item()}\n')

	# torch.save(model.state_dict(), 'saved_models/testing.pth')

# Load the model and test the autoencoder on test set
model = SkidNet()
model.load_state_dict(torch.load('final/SkidNet_2.pth'))
model.to(device)
print('Model Loaded\n')

# Evaluate the model
print(f'Evaluating the Model:')
test_loss = evaluate_model(model, test_loader, device)
print(f'Test loss: {test_loss:.4f}\n')

# PSNR of Model
print(f'Calculating PSNR of Model:')
psnr = PSNR(model, test_loader, device)
print(f'PSNR on Test: {psnr:.4f}\n')

# SSIM of Model
print(f'Calculating SSIM of Model:')
psnr = SSIM(model, test_loader, device)
print(f'SSIM on Test: {psnr:.4f}\n')

# Generate output for random images
n = 5
print(f'Generating output for random images:')
output_images = generate_images(model, test_loader, n, device, path=None)
print('Images Generated')

# Create specific output images
# input_image = next(iter(test_loader))[0][7]
# create_output(model, input_image, device)