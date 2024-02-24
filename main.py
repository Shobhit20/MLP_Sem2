import time
import random
import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF

from utils import *
from AutoEncShallow import *
from SkiD import *
from SkiDwithSkip import *
from SkiDwithSkipUnet import *

# Initialize the autoencoder
model = SkidNet()

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
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training the autoencoder
check_loss = 999
to_train = 1
num_epochs = 50
if to_train:
	start = time.time()
	for epoch in range(num_epochs):
		for i, data in enumerate(tqdm.tqdm(train_loader)):
			img, _ = data
			optimizer.zero_grad()
			output = model(img)
			loss = criterion(output, img)
			loss.backward()
			optimizer.step()

		if loss.item() < check_loss:
			check_loss = loss.item()
			print(f'Saving New Best Model')
			torch.save(model.state_dict(), 'models/best_SkidNet_50.pth')

		print(f'Time for one epoch: {time.time() - start}')
		print(f'Epoch [{epoch + 1}/{num_epochs}]  |  Loss: {loss.item()}\n')

	torch.save(model.state_dict(), 'models/SkidNet_50.pth')

# Load the model and test the autoencoder on test set
model = SkidNet()
model.load_state_dict(torch.load('models/best_SkidNet_50.pth'))
model.to(device)
print('Model Loaded')

# Evaluate the model
test_loss = model.evaluate_model(model, test_loader, device)
print(f'Test loss: {test_loss:.4f}')

# Generate output for random images
n = 5
output_images = model.generate_images(model, test_loader, n, device)
print('Images Generated')

# Create specific output images
# input_image = next(iter(test_loader))[0][7]
# model.create_output(model, input_image, device)