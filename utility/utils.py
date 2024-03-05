import os
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt
import numpy as np
import tqdm

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.utils as vutils
from sklearn.model_selection import train_test_split
from utility.noise import gaussian_blur, add_poisson_noise, add_salt_and_pepper_noise, add_speckle_noise
from utility.noise_functions import *

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
    # gaussian_noise = transforms.Lambda(lambda x: addGaussianNoiseTensor(x, mean = 0.1, std = 0.05))
    # sap_noise = transforms.Lambda(lambda x: addSaltPepperNoiseTensor(x, salt_prob = 0.015, pepper_prob = 0.015))
    # poisson_noise = transforms.Lambda(lambda x: addPoissonNoiseTensor(x, intensity=0.05))
    # speckle_noise = transforms.Lambda(lambda x: addSpeckleNoiseTensor(x, scale=0.4))
    gaussian_noise = transforms.Lambda(lambda x: gaussian_blur(x, kernel_size=15, sigma=1))
    sap_noise = transforms.Lambda(lambda x: add_salt_and_pepper_noise(x, salt_prob=0.05, pepper_prob=0.05))
    poisson_noise = transforms.Lambda(lambda x: add_poisson_noise(x, 0.1))
    speckle_noise = transforms.Lambda(lambda x: add_speckle_noise(x))


    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.456], std=[0.229])
    ])
    
    if noise:
        transform_noise = transforms.Compose([
            transforms.RandomApply([gaussian_noise], p = 0.4),
            transforms.RandomApply([sap_noise], p = 0.4),
            transforms.RandomApply([poisson_noise], p = 0.4),
            transforms.RandomApply([speckle_noise], p = 0.4)
        ])
    else: transform_noise = None

    data = []
    for image_name in os.listdir(data_dir):
        image_path = os.path.join(data_dir, image_name)
        data.append(image_path)

    data_train, data_test = train_test_split(data, test_size=test_size)

    device = getDevice()

    # ---------------------- Artificially Noised Images --------------------- #
    train_dataset = AutoencoderDataset(data_train, device=device, color=color, transform=transform, transform_noise=transform_noise)
    test_dataset = AutoencoderDataset(data_test, device=device, color=color, transform=transform, transform_noise=transform_noise)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=0)

    # ----------------------- Original Image for PSNR ----------------------- #
    train = AutoencoderDataset(data_train, device=device, color=color, transform=transform)
    test = AutoencoderDataset(data_test, device=device, color=color, transform=transform)

    train_original = DataLoader(train, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=0)
    test_original = DataLoader(test, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=0)

    return train_loader, test_loader, train_original, test_original

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

def evaluate_model(model, original, dataloader, device='cpu'):
    '''
    Evaluates the given model on the given dataloader and returns the average loss

    Args:
        model: the model to be evaluated
        dataloader: the dataloader to provide the dataset
        device: the device to run the model on

    Returns:
        average_loss: the average loss of the model on the dataset
    '''
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for i, (real, mod) in enumerate(tqdm.tqdm(zip(original, dataloader), total=len(original))):
            actual, _ = real
            actual = actual.to(device)

            modif, _ = mod
            modif = modif.to(device)

            # Forward pass
            outputs = model(modif)

            # Calculate reconstruction loss (MSE)
            loss = nn.functional.binary_cross_entropy_with_logits(outputs, actual)

            total_loss += loss.item()
            num_batches += 1

    average_loss = total_loss / num_batches
    return average_loss

def PSNR(model, original, dataloader, device='cpu'):
    '''
    Evaluates the given model on the given dataloader and returns the average PSNR

    Args:
        model: the model to generate images
        dataloader: the dataloader to provide the dataset
        device: the device to run the model on

    Returns:
        average_psnr: the average PSNR of the model on the dataset
    '''
    model.eval()
    total_psnr = 0.0
    num_batches = 0

    with torch.no_grad():
        for i, (real, mod) in enumerate(tqdm.tqdm(zip(original, dataloader), total=len(original))):
            actual, _ = real
            actual = (actual * 255).to(torch.uint8).to(device)
            
            images, _ = mod
            images = images.to(device)

            # Forward pass
            outputs = model(images)
            outputs = (outputs * 255).to(torch.uint8).to(device)

            if actual.dim() == 3:
                highest = torch.max(actual, dim = (1, 2)).item()
            else: highest = torch.max(actual).item()

            mse = nn.functional.mse_loss(outputs, actual)
            psnr = 10 * torch.log10((highest ** 2) / mse)
            total_psnr += psnr.item()
            num_batches += 1

    average_psnr = total_psnr / num_batches
    return average_psnr

def generate_images(model, dataloader, n, device='cpu', path=None):
    '''
    Picks n random images from a dataset and generates output images from a given model

    Args:
        model: the model to generate the images
        dataloader: the dataloader for the dataset
        n: the number of images to generate
        device: the device to run the model on

    Returns:
        generated_images: the output images generated by the model
    '''
    model.eval()
    original_images =[]
    generated_images = []
    random.seed(2024)
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
        input_image = original_images[i].cpu().squeeze()
        output_image = generated_images[i].cpu().squeeze()

        if len(input_image.shape) == 2:
            axes[0, i].imshow(input_image, cmap='gray')
            axes[1, i].imshow(output_image, cmap='gray')
        else:
            axes[0, i].imshow(input_image.permute(1, 2, 0))
            axes[1, i].imshow(output_image.permute(1, 2, 0))

        axes[0, i].set_title('Input Image'); axes[0, i].axis('off')
        axes[1, i].set_title('Output Image'); axes[1, i].axis('off')
    
    if path:
        plt.savefig(f'generated_images/{path}.png')

    plt.show()
    return generated_images

def create_output(model, input_image, device='cpu', path=None):
    '''
    Generates an output image for a given input image using a given model

    Args:
        model: the model to generate the output image
        input_image: the input image to generate the output from
        device: the device to run the model on

    Returns:
        0
    '''
    model.eval()
    with torch.no_grad():
        input_image = input_image.to(device)
        output = model(input_image.unsqueeze(0))
    
    input_image = input_image.cpu().squeeze()
    output_image = output.cpu().squeeze()

    # Display input and output images side by side
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    if len(input_image.shape) == 2:
        axes[0].imshow(input_image, cmap='gray')
        axes[1].imshow(output_image, cmap='gray')
    else:
        axes[0].imshow(input_image.permute(1, 2, 0))
        axes[1].imshow(output_image.permute(1, 2, 0))

    axes[0].set_title('Input Image'); axes[0].axis('off')
    axes[1].set_title('Output Image'); axes[1].axis('off')

    if path:
        plt.savefig(f'generated_images/{path}.png')

    plt.show()

    return 0