import os
import cv2
import numpy as np

folder_path = r"D:\AI\violence_224\train"
output_folder = r"D:\AI\violence_224\salt-pepper_noise"

# Traverse image files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        # Read the image
        img = cv2.imread(os.path.join(folder_path, filename))
        
        # Set the ratio of salt to pepper noise
        s_vs_p = 0.5
        
        # Set the amount of noise pixels
        amount = 0.04
        
        # Create a copy of the image
        noisy_img = np.copy(img)
        
        # Add salt noise
        num_salt = np.ceil(amount * img.size * s_vs_p)
        coords = [np.random.randint(0, i-1, int(num_salt)) for i in img.shape[:2]]
        noisy_img[coords[0], coords[1], :] = [255, 255, 255]
        
        # Add pepper noise
        num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i-1, int(num_pepper)) for i in img.shape[:2]]
        noisy_img[coords[0], coords[1], :] = [0, 0, 0]
        
        # Save the noisy image
        cv2.imwrite(os.path.join(output_folder, "salt-pepper_noise_" + filename), noisy_img)
