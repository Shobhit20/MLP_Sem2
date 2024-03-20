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

data_dir = 'data/'
batch_size = 32
train_loader, test_loader = loadData(data_dir, batch_size, test_size=0.2, color='gray', noise=True)
print('Data Loading Complete!')

# ------------------------ Move the model to GPU ------------------------ #
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Device: {device}\n')

# --------- Load the model and test the autoencoder on test set --------- #
model = SkidNet()
model.load_state_dict(torch.load('saved_models/SkidNet_3.pth'))
model.to(device)
print('Model Loaded\n')

# ------------------------- Save Modified Images ------------------------ #
model.eval()
counter = 0
for i, mod in enumerate(tqdm.tqdm(train_loader, total = len(train_loader))):
    modif, actual = mod
    output = model(modif)

    for j, k in zip(range(actual.size(0)), range(output.size(0))):
        actual_image = actual[j]
        actual_image = (actual_image - actual_image.min()) / (actual_image.max() - actual_image.min())

        generated_image = output[k]
        generated_image = (generated_image - generated_image.min()) / (generated_image.max() - generated_image.min())

        utils.save_image(actual_image, f"intermediate_data/Skid_MSE_og2/{counter}.png")
        utils.save_image(generated_image, f"intermediate_data/Skid_MSE2/{counter}.png")
        counter += 1