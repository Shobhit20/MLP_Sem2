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

class AutoencoderDataset(Dataset):
    '''Class defining the dataset for the autoencoder'''

    def __init__(self, data, device='cpu', color='gray', transform=None, transform_noise=None):
        '''
        Constructor for the AutoencoderDataset class

        Args:
            data: the data to be used for the dataset
            device: the device to load the data on
            color: define the color type of the images
            transform: the transformations to be applied to the images
        '''
        self.data = data
        self.transform = transform
        self.device = device
        self.color = color
        self.transform_noise = transform_noise

    def __len__(self):
        '''Returns the length of the dataset'''
        return len(self.data)

    def __getitem__(self, index):
        '''
        Returns the item at the given index

        Args:
            index: the index of the item to be returned

        Returns:
            x, x: the image at the given index as the input and the target
        '''
        x = Image.open(self.data[index])

        if self.color == 'color':
            if x.mode != 'RGB':
                x = x.convert('RGB')
        elif self.color == 'gray':
            x = x.convert('L')
        else:
            raise ValueError('Invalid color type. Please use either "color" or "gray"')

        if self.transform:
            x = self.transform(x)

        if self.transform_noise:
            new_x = self.transform_noise(x)
            return new_x.to(self.device), x.to(self.device)

        return x.to(self.device), x.to(self.device)

def loadData(data_dir, batch_size, test_size=0.2, color='gray', noise=False):
    '''
    Loads the data from the given directory and returns the train and test loaders

    Args:
        data_dir: the directory containing the data
        batch_size: the batch size for the data loaders
        test_size: the proportion of the data to be assigned to the test set
        color: the color type of the images
    
    Returns:
        train_loader: the data loader for the training set
        test_loader: the data loader for the test set
    '''
    gaussian_noise = transforms.Lambda(lambda x: addGaussianNoiseTensor(x, mean = 0.1, std = 0.05))
    sap_noise = transforms.Lambda(lambda x: addSaltPepperNoiseTensor(x, salt_prob = 0.06, pepper_prob = 0.06))
    poisson_noise = transforms.Lambda(lambda x: addPoissonNoiseTensor(x, intensity=0.05))
    speckle_noise = transforms.Lambda(lambda x: addSpeckleNoiseTensor(x, scale=0.4))

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.456], std=[0.229])
    ])
    
    if noise:
        transform_noise = transforms.Compose([
            transforms.RandomApply([gaussian_noise], p = 0.4),
            transforms.RandomApply([sap_noise], p = 0.4),
            transforms.RandomApply([poisson_noise], p = 0.4)
        ])
    else: transform_noise = None

    data = []
    for image_name in os.listdir(data_dir):
        image_path = os.path.join(data_dir, image_name)
        data.append(image_path)

    data_train, data_test = train_test_split(data, test_size=test_size, random_state=42)

    device = getDevice()
    train_dataset = AutoencoderDataset(data_train, device=device, color=color, transform=transform, transform_noise=transform_noise)
    test_dataset = AutoencoderDataset(data_test, device=device, color=color, transform=transform, transform_noise=transform_noise)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=0)

    return train_loader, test_loader

def showImages(dataloader, num_images=5):
    '''
    Displays a grid of sample images from the given data loader

    Args:
        dataloader: the data loader to display the images from
        num_images: the number of images to display

    Returns:
        None
    '''
    data_iter = iter(dataloader)
    images, labels = next(data_iter)

    plt.figure(figsize=(15, 4))
    for i in range(num_images):
        plt.subplot(2, num_images, i + 1)
        plt.axis("off")
        plt.title("Input {}".format(i + 1))
        plt.imshow(np.transpose(vutils.make_grid(images[i], padding=5, normalize=True).cpu(), (1, 2, 0)))

        plt.subplot(2, num_images, i + 1 + num_images)
        plt.axis("off")
        plt.title("Output {}".format(i + 1))
        plt.imshow(np.transpose(vutils.make_grid(labels[i], padding=5, normalize=True).cpu(), (1, 2, 0)))

    plt.show()

def getDevice():
    '''Returns the device to be used in case of availability of the GPU'''
    return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# ----------------------------------------------------------------------- #
#                         Numpy Image Manipulation                        #
# ----------------------------------------------------------------------- #
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

# ----------------------------------------------------------------------- #
#                        Tensor Image Manipulation                        #
# ----------------------------------------------------------------------- #
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