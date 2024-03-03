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
    '''
    Generate a 2D Gaussian mask centered at the image's center

    Args:
        size: Size of the mask
        std: Standard deviation of the Gaussian distribution

    Returns:
        gaussian_mask: Gaussian mask of the specified size
    '''
    grid = torch.arange(size, dtype=torch.float32)
    grid -= (size - 1) / 2  # Center the grid
    xx, yy = torch.meshgrid(grid, grid)
    gaussian = torch.exp(-(xx ** 2 + yy ** 2) / (2 * std ** 2))
    return gaussian

def generate_binary_mask_from_gaussian(gaussian_mask):
    '''
    Generate a binary mask from a Gaussian mask based on random number comparison

    Args:
        gaussian_mask: Gaussian mask to generate the binary mask
    
    Returns:
        binary_mask: Binary mask generated from the Gaussian mask
    '''
    random_numbers = torch.rand(gaussian_mask.size())
    binary_mask = random_numbers < gaussian_mask
    return binary_mask

def add_salt_and_pepper_noise(image, salt_prob=0.05, pepper_prob=0.05):
    '''
    Adds Salt and Pepper Noise to the given image

    Args:
        image: the image to add noise to
        salt_prob: the probability of adding salt noise
        pepper_prob: the probability of adding pepper noise
    
    Returns:
        noisy_image: the noisy image
    '''
    # Creating a Gaussian Mask
    size = image.size()
    gaussian = gaussian_mask(size[-1], std=size[-1]//4)
    mask = generate_binary_mask_from_gaussian(gaussian)
    
    # Add salt noise to the image
    salt_mask = torch.rand(image.size())  < salt_prob
    salt_mask = salt_mask & mask
    image[salt_mask] = 1.0
    
    # Add pepper noise to the image
    pepper_mask = torch.rand(image.size()) < pepper_prob
    pepper_mask = pepper_mask & mask
    image[pepper_mask] = 0.0

    return image


def add_speckle_noise(image, mean=0, std=0.1):
    '''
    Add speckle noise to the input image.

    Args:
        image: Input image array.
        mean: Mean of the noise distribution.
        std: Standard deviation of the noise distribution.

    Returns:
        noise_image: Image array with speckle noise added.
    '''
    # Generate speckle noise with the same shape as the input image
    noise = np.random.normal(mean, std, size=np.array(image).shape)

    # Add the noise to the image
    noisy_image = image.numpy() + image.numpy() * noise

    # Clip the pixel values to ensure they are within the valid range [0, 1]
    noisy_image = np.clip(noisy_image, 0, 255)
    noisy_image = torch.from_numpy(noisy_image.astype(np.float32))

    return noisy_image


def add_poisson_noise(image, noise_factor):
    '''
    Adds Poisson Noise to the given tensor image

    Args:
        image: the tensor image to add noise to
        noise_factor: the intensity of the noise
    
    Returns:
        noisy_image: the noisy tensor image
    '''
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
    '''
    Apply Gaussian blurring to a PIL image.

    Args:
        image (PIL.Image): Input PIL image
        kernel_size (int): Size of the Gaussian kernel (both width and height)
        sigma (float): Standard deviation of the Gaussian distribution

    Returns:
        PIL.Image: Blurred image
    '''
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
