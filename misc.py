import os
import numpy as np
import matplotlib.pyplot as plt

import cv2
from sklearn.model_selection import train_test_split

def loadData(data_dir, test_size=0.2):
    data = []
    for image_name in os.listdir(data_dir):
        image_path = os.path.join(data_dir, image_name)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (256, 256))
        image = image / 255.0
        data.append(image)

    data = np.array(data)
    np.random.shuffle(data)
    return train_test_split(data, test_size=test_size, random_state=42)

def showImages(images, num_images=5):
    for i in range(num_images):
        plt.subplot(1, num_images, i+1)
        plt.imshow(images[i])
        plt.axis('off')
    plt.show()