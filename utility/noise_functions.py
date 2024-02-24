import os
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.utils as vutils
from sklearn.model_selection import train_test_split

# ----------------------- Numpy Image Manipulation ---------------------- #
def addGaussianNoise(image, mean=0, std=25):
    '''
    Adds Gaussian Noise to the given image

    Args:
        image: the image to add noise to
        mean: the mean of the Gaussian distribution
        std: the standard deviation of the Gaussian distribution
    
    Returns:
        image_noisy: the noisy image
    '''
    noise = np.random.normal(mean, std, image.shape)
    image_noisy = image + noise
    image_noisy = np.clip(image_noisy, 0, 255)
    return image_noisy

def addSaltPepperNoise(image, salt_prob = 0.08, pepper_prob = 0.08):
    '''
    Adds Salt and Pepper Noise to the given image

    Args:
        image: the image to add noise to
        salt_prob: the probability of adding salt noise
        pepper_prob: the probability of adding pepper noise
    
    Returns:
        noisy_image: the noisy image
    '''
    noisy_image = np.copy(image)

    # Add salt noise
    salt_mask = np.random.rand(*image.shape[:2]) < salt_prob
    noisy_image[salt_mask] = 255

    # Add pepper noise
    pepper_mask = np.random.rand(*image.shape[:2]) < pepper_prob
    noisy_image[pepper_mask] = 0

    return noisy_image

def addPoissonNoise(image, intensity = 0.1):
    '''
    Adds Poisson Noise to the given image

    Args:
        image: the image to add noise to
        intensity: the intensity of the noise
    
    Returns:
        noisy_image: the noisy image
    '''
    poisson_noise = intensity * np.random.poisson(image / intensity)
    noisy_image = np.clip(image + poisson_noise, 0, 255).astype(np.uint8)
    return noisy_image

def addSpeckleNoise(image, scale = 0.4):
    ''' 
    Adds Speckle Noise to the given image

    Args:
        image: the image to add noise to
        scale: the scale of the noise

    Returns:
        noisy_image: the noisy image
    '''
    # Generate speckle noise
    row, col, ch = image.shape
    speckle_noise = scale * np.random.randn(row, col, ch)

    # Add noise to the image
    noisy_image = image + image * speckle_noise

    # Clip values to ensure they are within the valid range [0, 255]
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    return noisy_image

# ---------------------- Tensor Image Manipulation ---------------------- #
def addGaussianNoiseTensor(image, mean = 0, std = 25):
    '''
    Adds Gaussian Noise to the given tensor image

    Args:
        image: the tensor image to add noise to
        mean: the mean of the Gaussian distribution
        std: the standard deviation of the Gaussian distribution
    
    Returns:
        image_noisy: the noisy tensor image
    '''
    noise = torch.randn_like(image) * std + mean
    image_noisy = image + noise
    image_noisy = torch.clamp(image_noisy, 0, 255)
    return image_noisy

def addSaltPepperNoiseTensor(image, salt_prob=0.08, pepper_prob=0.08):
    '''
    Adds Salt and Pepper Noise to the given tensor image

    Args:
        image: the tensor image to add noise to
        salt_prob: the probability of adding salt noise
        pepper_prob: the probability of adding pepper noise
    
    Returns:
        noisy_image: the noisy tensor image
    '''
    noisy_image = image.clone()

    # Add salt noise
    salt_mask = torch.rand_like(image) < salt_prob
    noisy_image[salt_mask] = 1.0

    # Add pepper noise
    pepper_mask = torch.rand_like(image) < pepper_prob
    noisy_image[pepper_mask] = 0.0

    return noisy_image

def addPoissonNoiseTensor(image, intensity=0.1):
    '''
    Adds Poisson Noise to the given tensor image

    Args:
        image: the tensor image to add noise to
        intensity: the intensity of the noise
    
    Returns:
        noisy_image: the noisy tensor image
    '''
    poisson_noise = intensity * torch.rand_like(image)
    noisy_image = torch.clamp(image + poisson_noise, 0, 255)
    return noisy_image

def addSpeckleNoiseTensor(image, scale=0.4):
    ''' 
    Adds Speckle Noise to the given tensor image

    Args:
        image: the tensor image to add noise to
        scale: the scale of the noise

    Returns:
        noisy_image: the noisy tensor image
    '''
    speckle_noise = scale * torch.randn_like(image)
    noisy_image = image + image * speckle_noise

    # Clip values to ensure they are within the valid range [0, 255]
    noisy_image = torch.clamp(noisy_image, 0, 255)

    return noisy_image

# -------------------------- Make Sample Image -------------------------- #
def makeNoisyImages(path):
    '''
    Adds different types of noise to the given image and saves the noisy images

    Args:
        path: the path to the image to add noise to

    Returns:
        None
    '''
    # Read an image from data folder
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))

    # Add Gaussian Noise to the image using addGaussianNoise function
    gaussian_img = addGaussianNoise(img, mean = 1)
    cv2.imwrite('gaussian.png', gaussian_img)

    # Add Salt and Pepper noise to the image using addSaltPepperNoise function
    sap_img = addSaltPepperNoise(img)
    cv2.imwrite('sap.png', sap_img)

    # Add Poisson Noise to the image using addPoissonNoise function
    poisson_img = addPoissonNoise(img)
    cv2.imwrite('poisson.png', poisson_img)