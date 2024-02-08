import numpy as np
import cv2

def add_speckle_noise(image, scale=0.5):
    """
    Add speckle noise to an image.

    Parameters:
    - image: Input image (numpy array).
    - scale: Scaling factor for the noise.

    Returns:
    - Noisy image (numpy array).
    """
    # Generate speckle noise
    row, col, ch = image.shape
    speckle_noise = scale * np.random.randn(row, col, ch)

    # Add noise to the image
    noisy_image = image + image * speckle_noise

    # Clip values to ensure they are within the valid range [0, 255]
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    return noisy_image

# Example usage:
# Load an image using OpenCV
image_path = 'data/00006632_004.png'
original_image = cv2.imread(image_path)

# Add speckle noise
noisy_image = add_speckle_noise(original_image, scale=0.2)

# Display the original and noisy images (you may need to install a display library like matplotlib)
import matplotlib.pyplot as plt

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB))
plt.title('Noisy Image with Speckle Noise')

plt.show()
