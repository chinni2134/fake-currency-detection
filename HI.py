import os
import shutil
import random

# Define the source folder and destination paths
source_folder = 'Dataset2(Final)/10_front view'  # Correct source path
real_folder = 'Organized_Dataset/real'
fake_folder = 'Organized_Dataset/fake'

# Create destination folders if they don't exist
os.makedirs(real_folder, exist_ok=True)
os.makedirs(fake_folder, exist_ok=True)

# Get the list of images
image_files = os.listdir(source_folder)

# Shuffle the list to ensure a random split
random.shuffle(image_files)

# Half the dataset will be real, half fake
split_point = len(image_files) // 2
real_images = image_files[:split_point]
fake_images = image_files[split_point:]

# Move the images to their respective folders
for img in real_images:
    shutil.move(os.path.join(source_folder, img), os.path.join(real_folder, img))

for img in fake_images:
    shutil.move(os.path.join(source_folder, img), os.path.join(fake_folder, img))

print("Dataset has been split into 'real' and 'fake' folders.")
