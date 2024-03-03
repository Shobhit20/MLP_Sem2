import torch
import torchvision.transforms.functional as TF
import random
from PIL import Image
import cv2
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F
from matplotlib import pyplot as plt
import torch.nn as nn


def gaussian_mask(size, std):
    """Generate a 2D Gaussian mask centered at the image's center."""
    grid = torch.arange(size, dtype=torch.float32)
    grid -= (size - 1) / 2  # Center the grid
    xx, yy = torch.meshgrid(grid, grid)
    gaussian = torch.exp(-(xx ** 2 + yy ** 2) / (2 * std ** 2))
    return gaussian

def generate_binary_mask_from_gaussian(gaussian_mask):
    """Generate a binary mask from a Gaussian mask based on random number comparison."""
    random_numbers = torch.rand(gaussian_mask.size())
    binary_mask = random_numbers < gaussian_mask
    return binary_mask

def add_salt_and_pepper_noise(image, salt_prob=0.05, pepper_prob=0.05):
    """Add salt and pepper noise to the input image."""
    size = image.size()
    gaussian = gaussian_mask(size[-1], std=size[-1]//4)
    mask = generate_binary_mask_from_gaussian(gaussian)
    salt_mask = torch.rand(image.size())  < salt_prob
    pepper_mask = torch.rand(image.size()) < pepper_prob
    salt_mask = salt_mask & mask
    pepper_mask = pepper_mask & mask
    # Salt noise
    image[salt_mask] = 1.0
    # Pepper noise
    image[pepper_mask] = 0.0
    return image


def add_speckle_noise(image, mean=0, std=0.1):
    """
    Add speckle noise to the input image.

    Parameters:
        image (ndarray): Input image array.
        mean (float): Mean of the noise distribution.
        std (float): Standard deviation of the noise distribution.

    Returns:
        ndarray: Image array with speckle noise added.
    """
    # Generate speckle noise with the same shape as the input image
    noise = np.random.normal(mean, std, size=np.array(image).shape)
    # Add the noise to the image
    noisy_image = image.numpy() + image.numpy() * noise
    # Clip the pixel values to ensure they are within the valid range [0, 1]
    noisy_image = np.clip(noisy_image, 0, 255)
    noisy_image = torch.from_numpy(noisy_image.astype(np.float32))
    return noisy_image


def add_poisson_noise(image,noise_factor):
    """
    Add Poisson noise to the input tensor.

    Args:
    - tensor (torch.Tensor): Input tensor
    - noise_factor (float): Factor to control the amount of noise

    Returns:
    - torch.Tensor: Noisy tensor
    """

    # Generate a mask with Poisson noise
    size = image.size()
    noise_mask = torch.empty_like(image).uniform_(0, 1)
    gaussian = gaussian_mask(size[-1], size[-1]//4)
    noise_mask = torch.poisson(noise_mask * gaussian * noise_factor)
    # Add the noise mask to the input tensor
    noisy_tensor = image.numpy() + noise_mask.numpy()
    np_img = np.clip(noisy_tensor, 0, 1)
    return torch.from_numpy(np_img)


def gaussian_blur(image, kernel_size, sigma):
    """
    Apply Gaussian blurring to a PIL image.

    Args:
    - image (PIL.Image): Input PIL image
    - kernel_size (int): Size of the Gaussian kernel (both width and height)
    - sigma (float): Standard deviation of the Gaussian distribution

    Returns:
    - PIL.Image: Blurred image
    """

    # Convert PIL image to a PyTorch tensor
    image_tensor = image.unsqueeze(0)
    # Generate Gaussian kernel
    kernel = transforms.GaussianBlur(kernel_size, sigma)
    # Apply Gaussian blur
    blurred_tensor = kernel(image_tensor)
    return blurred_tensor.squeeze(0)


# # Example usage:
# image_path = "data/00009229_016.png"  
# input_image = Image.open(image_path).convert('L')  # Load an example input image
# # Example usage:
# size = 1024  # Size of the mask
# std = 200 # Standard deviation of the Gaussian distribution
# print("input mae")

# # input_image.show()

# # noisy_image = add_speckle_noise(input_image)
# # noisy_image = add_poisson_noise(input_image, 0.8)

# # Apply Gaussian blur with varying standard deviation to the input image
# blurred_image = gaussian_blur(input_image, kernel_size=15, sigma=5)

# plt.imshow(blurred_image, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
# plt.show()
# gaussian = gaussian_mask(size, std)
# mask = generate_binary_mask_from_gaussian(gaussian)


# noisy_image = add_salt_and_pepper_noise(input_image, mask, salt_prob=0.1, pepper_prob=0.1)
# noisy_image.show()  # Display the noisy image
