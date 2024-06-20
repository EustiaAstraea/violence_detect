import cv2
import os
import numpy as np

# Read each image in the folder
folder_path = r"..\train"
output_folder = r"..\gauss_noise"

for filename in os.listdir(folder_path):
    # Read the image
    img = cv2.imread(os.path.join(folder_path, filename))
    # Set the mean and standard deviation for the Gaussian distribution
    mean = 0
    sigma = 25
    # Generate noise following a Gaussian distribution based on mean and standard deviation
    gauss = np.random.normal(mean, sigma, img.shape)
    # Add Gaussian noise to the image
    noisy_img = img + gauss
    # Clip pixel values of the image after adding noise
    noisy_img = np.clip(noisy_img, a_min=0, a_max=255)
    # Save the image with formatted filename
    output_path = os.path.join(output_folder, "gauss_noisy_" + filename)
    cv2.imwrite(output_path, noisy_img)
