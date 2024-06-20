import os
import cv2
import numpy as np

folder_path = r"D:\AI\violence_224\train"
output_folder = r"D:\AI\violence_224\poisson_noise"

# Traverse image files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        # Read the image
        img = cv2.imread(os.path.join(folder_path, filename))
        
        # Compute the range of pixel distribution
        vals = len(np.unique(img))
        vals = 2 ** np.ceil(np.log2(vals))
        
        # Add Poisson noise to the image
        noisy_img = np.random.poisson(img * vals) / float(vals)
        
        # Save the noisy image
        cv2.imwrite(os.path.join(output_folder, "poisson_noise_" + filename), noisy_img)
