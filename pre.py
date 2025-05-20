import os
from PIL import Image # For image loading and resizing
import numpy as np
from sklearn.model_selection import train_test_split # If you decide to split here
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical # For one-hot encoding
import cv2 # For image operations, can be useful for samples
import time # For unique filenames and simulated delays
import traceback # For detailed error logging
import random # For sample selection

# Configuration (can be passed as arguments or defined if static for this module)
VALID_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.gif'] # Common image extensions
TARGET_IMAGE_SIZE = (224, 224) # As per your original preprocessing.py

# This directory should be created by app.py if it doesn't exist
# This is where the final .npy files will be saved, relative to app.py/pre.py
PROCESSED_DATA_DIR = 'processed_data'

def is_valid_image_file(filename):
    """Checks if the filename has a valid image extension and isn't a hidden file."""
    if filename.startswith('._'):  # Skip hidden macOS resource fork files
        return False
    ext = os.path.splitext(filename)[1].lower()
    return ext in VALID_IMAGE_EXTENSIONS

def process_dataset_step_by_step(dataset_path, status_callback=None, image_output_dir_static_abs=None, image_output_dir_static_rel=None):
    """
    Processes the dataset from dataset_path, provides status updates via status_callback,
    saves sample images, and saves the final processed X and Y data.

    Args:
        dataset_path (str): Absolute path to the root of the dataset to process.
        status_callback (function, optional): Function to call with log messages. Defaults to print.
        image_output_dir_static_abs (str, optional): Absolute path to the directory within 'static' 
                                                     where sample images should be SAVED.
        image_output_dir_static_rel (str, optional): Path relative to the 'static' folder, used for 
                                                     constructing web-accessible URLs for sample images.

    Returns:
        tuple: (final_X_path, final_Y_path, sample_image_web_paths_dict)
               - final_X_path (str): Path to the saved X_processed.npy file, or None on failure.
               - final_Y_path (str): Path to the saved Y_processed.npy file, or None on failure.
               - sample_image_web_paths_dict (dict): Dictionary of sample images, or error info.
    """
    if status_callback is None:
        status_callback = print

    status_callback(f"Preprocessing initiated for dataset: {os.path.basename(dataset_path)}")
    time.sleep(0.1) # Small delay for UI

    all_loaded_data_pil = [] # Store PIL images initially for easier sample saving
    all_loaded_labels_str = []
    sample_image_web_paths_dict = {}
    total_images_attempted = 0
    total_images_successfully_loaded = 0
    sample_save_count_per_class = 1 # Number of sample images to save per class

    # --- Ensure Output Directories Exist ---
    # Sample image output directory (absolute path for saving)
    # This should ideally be created by app.py, but check here too.
    if image_output_dir_static_abs and not os.path.exists(image_output_dir_static_abs):
        try:
            os.makedirs(image_output_dir_static_abs)
            status_callback(f"INFO: Created sample image output directory: {image_output_dir_static_abs}")
        except Exception as e_dir_img:
            status_callback(f"WARNING: Could not create sample image directory {image_output_dir_static_abs}: {e_dir_img}. Sample images may not be saved.")
            image_output_dir_static_abs = None # Disable sample image saving

    # Processed .npy output directory (relative to current script)
    if not os.path.exists(PROCESSED_DATA_DIR):
        try:
            os.makedirs(PROCESSED_DATA_DIR)
            status_callback(f"INFO: Created directory for processed .npy files: {PROCESSED_DATA_DIR}")
        except Exception as e_dir_npy:
            status_callback(f"CRITICAL ERROR: Could not create output directory {PROCESSED_DATA_DIR} for .npy files: {e_dir_npy}")
            return None, None, {"Error": "Failed to create .npy output directory."}

    if not os.path.isdir(dataset_path):
        status_callback(f"❌ CRITICAL ERROR: Dataset directory not found at '{dataset_path}'")
        return None, None, {"Error": "Dataset directory not found."}

    try:
        # --- 1. Load Images and Labels ---
        status_callback("Step 1: Scanning dataset and loading images...")
        class_folders = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
        
        if not class_folders:
            status_callback("❌ CRITICAL ERROR: No class subdirectories found in the dataset path!")
            return None, None, {"Error": "No class subdirectories found."}
        
        status_callback(f"INFO: Found {len(class_folders)} potential class folders: {', '.join(class_folders)}")

        for class_idx, foldername in enumerate(class_folders):
            folder_path_abs = os.path.join(dataset_path, foldername)
            status_callback(f"Processing class '{foldername}' ({class_idx + 1}/{len(class_folders)})...")
            images_in_class_loaded_count = 0

            filenames_in_folder = os.listdir(folder_path_abs)
            if not filenames_in_folder:
                status_callback(f"  INFO: Class folder '{foldername}' is empty. Skipping.")
                continue

            for filename in filenames_in_folder:
                total_images_attempted += 1
                if is_valid_image_file(filename):
                    file_path_abs = os.path.join(folder_path_abs, filename)
                    try:
                        # Load with PIL, convert to RGB, resize
                        image_pil = Image.open(file_path_abs).convert('RGB')
                        image_pil_resized = image_pil.resize(TARGET_IMAGE_SIZE, Image.LANCZOS) # Use high-quality downsampling

                        all_loaded_data_pil.append(image_pil_resized) # Store PIL image
                        all_loaded_labels_str.append(foldername)
                        total_images_successfully_loaded += 1
                        images_in_class_loaded_count += 1

                        # Save sample image (first one from each class)
                        if images_in_class_loaded_count <= sample_save_count_per_class and \
                           image_output_dir_static_abs and image_output_dir_static_rel:
                            
                            sample_img_filename = f"{foldername.replace(' ', '_')}_sample_{images_in_class_loaded_count}_{int(time.time())}.jpg"
                            absolute_sample_save_path = os.path.join(image_output_dir_static_abs, sample_img_filename)
                            
                            try:
                                image_pil_resized.save(absolute_sample_save_path, "JPEG", quality=85)
                                web_path = os.path.join(image_output_dir_static_rel, sample_img_filename).replace("\\", "/")
                                sample_image_web_paths_dict[f"{foldername} Sample {images_in_class_loaded_count}"] = web_path
                                status_callback(f"  INFO: Saved sample: {web_path}")
                            except Exception as e_save_sample:
                                status_callback(f"  WARNING: Could not save sample image {sample_img_filename}: {e_save_sample}")
                        
                        if total_images_successfully_loaded % 50 == 0: # Update every 50 images
                             status_callback(f"  INFO: Loaded {total_images_successfully_loaded} images so far ({filename})...")

                    except Exception as e_load_img:
                        status_callback(f"  ❌ ERROR loading/processing image {file_path_abs}: {e_load_img}")
                # else: (Optional: log skipped non-image files)
                #    status_callback(f"  Skipping non-image or hidden file: {filename}")
            status_callback(f"Finished class '{foldername}'. Loaded {images_in_class_loaded_count} valid images.")
            time.sleep(0.05) # Brief pause for UI responsiveness

        status_callback(f"✅ Step 1 Complete: Image loading finished. Attempted: {total_images_attempted}, Successfully loaded: {total_images_successfully_loaded}.")

        if not all_loaded_data_pil:
            status_callback("❌ CRITICAL ERROR: No images were successfully loaded. Aborting.")
            return None, None, sample_image_web_paths_dict

        # --- 2. Convert to NumPy and Normalize ---
        status_callback("Step 2: Converting images to NumPy arrays and normalizing...")
        # Convert PIL images to NumPy arrays
        data_np_list = [np.array(img_pil) for img_pil in all_loaded_data_pil]
        del all_loaded_data_pil # Free memory
        
        data_np = np.array(data_np_list, dtype=np.float32)
        labels_np_str = np.array(all_loaded_labels_str)
        status_callback(f"  INFO: Data array shape: {data_np.shape}, Labels array shape: {labels_np_str.shape}")

        status_callback("  Normalizing image data (dividing by 255.0)...")
        data_np = data_np / 255.0
        status_callback("✅ Step 2 Complete: Normalization finished.")
        time.sleep(0.1)

        # --- 3. Encode Labels ---
        status_callback("Step 3: Encoding labels...")
        encoder = LabelEncoder()
        integer_encoded_labels = encoder.fit_transform(labels_np_str)
        # Log the class mapping
        class_mapping_log = {i: cls_name for i, cls_name in enumerate(encoder.classes_)}
        status_callback(f"  INFO: LabelEncoder class mapping: {class_mapping_log}")
        
        status_callback("  One-hot encoding labels...")
        one_hot_encoded_labels = to_categorical(integer_encoded_labels, num_classes=len(encoder.classes_))
        status_callback(f"✅ Step 3 Complete: Label encoding finished. Shape of one-hot labels (Y): {one_hot_encoded_labels.shape}")
        time.sleep(0.1)

        # --- 4. Save Processed Data (Full Dataset) ---
        # The train_test_split from your original code can be done later, before actual training.
        # Here, we save the entire processed dataset.
        X_to_save = data_np
        Y_to_save = one_hot_encoded_labels

        status_callback("Step 4: Saving final processed X and Y arrays as .npy files...")
        timestamp = int(time.time())
        # Save .npy files in the PROCESSED_DATA_DIR defined at the top of this file
        final_X_filename = f"X_processed_full_{TARGET_IMAGE_SIZE[0]}x{TARGET_IMAGE_SIZE[1]}_{timestamp}.npy"
        final_Y_filename = f"Y_processed_full_{len(encoder.classes_)}classes_{timestamp}.npy"
        
        final_X_path = os.path.join(PROCESSED_DATA_DIR, final_X_filename)
        final_Y_path = os.path.join(PROCESSED_DATA_DIR, final_Y_filename)

        try:
            np.save(final_X_path, X_to_save)
            status_callback(f"  ✅ Successfully saved processed X data to: {final_X_path}")
            np.save(final_Y_path, Y_to_save)
            status_callback(f"  ✅ Successfully saved processed Y data to: {final_Y_path}")
        except Exception as e_save_npy_final:
            status_callback(f"❌ CRITICAL ERROR saving final .npy files: {e_save_npy_final}")
            traceback.print_exc() # Log to server console
            return None, None, sample_image_web_paths_dict # Return samples but indicate NPY saving failed
            
        status_callback("✅ Step 4 Complete: Processed data saved.")
        status_callback("✅✅✅ All preprocessing steps finished successfully! ✅✅✅")
        return final_X_path, final_Y_path, sample_image_web_paths_dict

    except Exception as e_main_processing:
        error_msg = f"❌❌❌ AN UNEXPECTED FATAL ERROR occurred during preprocessing: {e_main_processing}"
        status_callback(error_msg)
        # Log full traceback to server console for debugging
        print(f"--- TRACEBACK FOR PREPROCESSING ERROR IN pre.py (Task: {os.path.basename(dataset_path)}) ---")
        traceback.print_exc()
        print("------------------------------------------------------------------------------------")
        # Return None for paths, but include any samples collected and an error indicator
        sample_image_web_paths_dict["Critical Error"] = "Processing failed. Check server logs for details."
        return None, None, sample_image_web_paths_dict

# --- Standalone Test Block (optional, for testing pre.py directly) ---
if __name__ == '__main__':
    print("--- Running pre.py Standalone Test ---")
    
    # Define a base directory for the test (where pre.py is located)
    test_base_dir = os.path.abspath(os.path.dirname(__file__))
    
    # Dataset folder for testing (assumed to be next to pre.py)
    test_dataset_folder_name = 'Dataset2(Final)' # Your dataset folder name
    test_dataset_path_abs = os.path.join(test_base_dir, test_dataset_folder_name)

    # Simulate static folder structure for test output
    test_static_folder_abs = os.path.join(test_base_dir, "test_outputs", "static")
    test_image_output_rel = os.path.join("images", "preprocessing_live_test") # Relative path for web URLs
    test_image_output_abs = os.path.join(test_static_folder_abs, test_image_output_rel) # Absolute path for saving

    # Ensure output directories for the test exist
    if not os.path.exists(test_image_output_abs):
        os.makedirs(test_image_output_abs)
        print(f"Test INFO: Created directory for sample images: {test_image_output_abs}")
    
    npy_output_dir_test = os.path.join(test_base_dir, "test_outputs", PROCESSED_DATA_DIR)
    if not os.path.exists(npy_output_dir_test): # Adjust PROCESSED_DATA_DIR for test if needed
        os.makedirs(npy_output_dir_test)
        print(f"Test INFO: Created directory for .npy files: {npy_output_dir_test}")
    
    # Temporarily override PROCESSED_DATA_DIR for standalone test if you want output elsewhere
    # global PROCESSED_DATA_DIR # If you modify it
    # PROCESSED_DATA_DIR = npy_output_dir_test 
    # Or, better, pass it as an argument if you often test standalone with different output dirs

    if not os.path.exists(test_dataset_path_abs) or not os.path.isdir(test_dataset_path_abs):
        print(f"❌ Test ERROR: Dataset folder '{test_dataset_folder_name}' not found at '{test_dataset_path_abs}'.")
        print("Please ensure it's in the same directory as pre.py for this standalone test.")
    else:
        print(f"Test INFO: Using dataset: {test_dataset_path_abs}")
        print(f"Test INFO: Sample images will be saved to (absolute): {test_image_output_abs}")
        print(f"Test INFO: Web paths for samples will use base: {test_image_output_rel}")
        print(f"Test INFO: .npy files will be saved to default PROCESSED_DATA_DIR: {os.path.join(test_base_dir, PROCESSED_DATA_DIR)}")
        print("-" * 30)

        x_file_path, y_file_path, samples_dict = process_dataset_step_by_step(
            dataset_path=test_dataset_path_abs,
            # status_callback=print, # Default is print, explicit for clarity
            image_output_dir_static_abs=test_image_output_abs,
            image_output_dir_static_rel=test_image_output_rel
        )

        print("-" * 30)
        if x_file_path and y_file_path:
            print(f"✅ Test Finished Successfully.")
            print(f"  Processed X data saved to: {x_file_path}")
            print(f"  Processed Y data saved to: {y_file_path}")
            print(f"  Sample images generated ({len(samples_dict)}):")
            if samples_dict:
                for title, path in samples_dict.items():
                    print(f"    - {title}: {path} (web path)")
            else:
                print("    No sample images were generated or reported.")
        else:
            print(f"❌ Test Finished With Errors During Preprocessing.")
            if samples_dict and "Error" in samples_dict:
                 print(f"   Error details: {samples_dict['Error']}")
            elif samples_dict and "Critical Error" in samples_dict:
                 print(f"   Error details: {samples_dict['Critical Error']}")


    print("--- End of pre.py Standalone Test ---")
