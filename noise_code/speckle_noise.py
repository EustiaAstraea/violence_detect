import os
import cv2
import numpy as np

folder_path = r"D:\AI\violence_224\train"

# Traverse image files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        # Read the image
        img = cv2.imread(os.path.join(folder_path, filename))
        
        # Get the height, width, and channels of the image
        img_height, img_width, img_channels = img.shape
        
        # Generate noise following a distribution
        gauss = np.random.randn(img_height, img_width, img_channels)
        
        # Add speckle noise to the image
        noisy_img = img + img * gauss
        
        # Normalize the pixel values of the image
        noisy_img = np.clip(noisy_img, a_min=0, a_max=255)
        
        # Save the image
        cv2.imwrite(os.path.join(r"D:\AI\violence_224\noise", "noisy_" + filename), noisy_img)
