import torch
import torch.nn as nn
import torch.optim as optim
from utils import *
import time
import random
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from models import *

# Define the autoencoder architecture
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
		images = []
		random_indices = random.sample(range(dataloader.batch_size), n)

		with torch.no_grad():
			for i, data in enumerate(dataloader):
				if i in random_indices:
					img, _ = data
					img = img.to(device)
					output = model(img)
					images.append(output[0].cpu())
		print(f'Sample Images Selected {random_indices}')

		fig, axes = plt.subplots(2, n, figsize=(3 * n, 8))
		for i in range(n):
			input_image = dataloader.dataset[random_indices[i]][0].cpu().permute(1, 2, 0)
			output_image = images[i].squeeze().permute(1, 2, 0)

			axes[0, i].imshow(input_image)
			axes[0, i].set_title('Input Image')
			axes[0, i].axis('off')

			axes[1, i].imshow(output_image)
			axes[1, i].set_title('Output Image')
			axes[1, i].axis('off')

		plt.show()
		return images
	
	def create_output(self, model, input_image, device='cpu'):
		model.eval()
		with torch.no_grad():
			input_image = input_image.to(device)
			output = model(input_image)

		input_image_pil = TF.to_pil_image(input_image.cpu().squeeze())
		output_image_pil = TF.to_pil_image(output.cpu().squeeze())

		# Display input and output images side by side
		fig, axes = plt.subplots(1, 2, figsize=(8, 4))

		axes[0].imshow(input_image_pil)
		axes[0].set_title('Input Image'); axes[0].axis('off')

		axes[1].imshow(output_image_pil)
		axes[1].set_title('Output Image'); axes[1].axis('off')
		plt.show()

		return 0


# Initialize the autoencoder
model = AutoencoderWithoutSkip()

data_dir = 'data/'
batch_size = 32
train_loader, test_loader = loadData(data_dir, batch_size)
print('Data Loading Complete!')

# Move the model to GPU
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Device: {device}')
model.to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the autoencoder
to_train = 1
i = 0
if to_train:
	num_epochs = 10
	start = time.time()
	for epoch in range(num_epochs):
		for data in train_loader:
			img, _ = data
			optimizer.zero_grad()
			output = model(img)
			loss = criterion(output, img)
			loss.backward()
			optimizer.step()
			i = i + 1

		print(f'Time for one epoch: {time.time() - start}')
		if epoch % 5== 0:
			print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

	# Save the model
	torch.save(model.state_dict(), 'conv_autoencoder_without.pth')

# Load the model and test the autoencoder on test set
# model = Autoencoder()
# model.load_state_dict(torch.load('conv_autoencoder.pth'))
# model.to(device)
# print('Data Loaded')

# Evaluate the model
# test_loss = model.evaluate_model(model, test_loader, device)
# print(f'Test loss: {test_loss:.4f}')

# Generate output images
n = 5
# showImages(test_loader, n)
output_images = model.generate_images(model, test_loader, n, device)
print('Images Generated')

# Create output images
# input_image = next(iter(test_loader))[0][31]
# model.create_output(model, input_image, device)