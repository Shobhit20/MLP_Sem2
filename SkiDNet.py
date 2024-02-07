import numpy
import pandas
import os

import torch
import torch.nn as nn
from utils import *
from models import *

# Load data in code from loadData function
data_dir = 'data/'
batch_size = 32
train_loader, test_loader = loadData(data_dir, batch_size)

# train the dataset on the model
model = Autoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)

n_epochs = 20
for epoch in range(1, n_epochs+1):
    train_loss = 0.0
    for data in train_loader:
        images, _ = data
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*images.size(0)
    train_loss = train_loss/len(train_loader)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))

# Save the model
# model_dir = 'models/'
# os.makedirs(model_dir, exist_ok=True)
# model_path = model_dir + 'autoencoder.pth'
# torch.save(model.state_dict(), model_path)