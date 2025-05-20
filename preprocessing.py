import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# --- Configuration ---
dataset_folder = 'Dataset2(Final)'
valid_image_extensions = ['.jpg', '.jpeg', '.png']

# --- Helper Function ---
def is_valid_image(filename):
    if filename.startswith('._'):  # Skip hidden macOS files
        return False
    ext = os.path.splitext(filename)[1].lower()
    return ext in valid_image_extensions

# --- Main Function ---
def load_and_process_dataset(folder_path_to_load):
    data = []
    labels = []
    total_processed_images = 0

    if not os.path.isdir(folder_path_to_load):
        print(f"❌ Error: Dataset directory not found at '{folder_path_to_load}'")
        return np.array([]), np.array([]), np.array([]), np.array([])

    for foldername in os.listdir(folder_path_to_load):
        folder_path = os.path.join(folder_path_to_load, foldername)
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                if is_valid_image(filename):
                    file_path = os.path.join(folder_path, filename)
                    try:
                        image = Image.open(file_path).convert('RGB')
                        image = image.resize((224, 224))
                        image_array = np.array(image) / 255.0  # Normalize
                        data.append(image_array)
                        labels.append(foldername)
                        total_processed_images += 1
                    except Exception as e:
                        print(f"❌ Error loading image {file_path}: {e}")

    print(f"✅ Finished loading. Total images: {total_processed_images}, Classes: {len(set(labels))}")

    # Convert to numpy arrays
    data = np.array(data)
    labels = np.array(labels)

    # Encode labels to integers
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    one_hot_labels = to_categorical(encoded_labels)

    # Split into train/test sets
    x_train, x_test, y_train, y_test = train_test_split(
        data, one_hot_labels, test_size=0.2, stratify=one_hot_labels, random_state=42
    )

    print(f"📦 Train samples: {len(x_train)}, Test samples: {len(x_test)}")
    return x_train, x_test, y_train, y_test

# --- Run if this file is executed directly ---
if __name__ == '__main__':
    x_train, x_test, y_train, y_test = load_and_process_dataset(dataset_folder)
