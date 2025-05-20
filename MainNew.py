# -*- coding: utf-8 -*-
from tkinter import *
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
# Use tensorflow.keras consistently
from tensorflow.keras.models import model_from_json, Sequential
from tensorflow.keras.layers import Input, MaxPooling2D, Dense, Dropout, Activation, Flatten, Conv2D # Use Conv2D
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
import pickle
import os
from sklearn.model_selection import train_test_split
# Import threading if you want to implement non-blocking operations later
# import threading
# from queue import Queue # Useful for thread communication

main = Tk()
main.title("Fake Currency Detection System")
main.geometry("1200x800")
main.resizable(True, True)

# --- Globals ---
# Note: Heavy use of globals can make complex apps harder to manage.
# Consider using a class structure for larger applications.
filename = None # Initialize filename for dataset path
X, Y = None, None # Initialize raw data holders
X_train, X_test, y_train, y_test = None, None, None, None # Initialize train/test splits
model = None # Initialize model holder
classes = [] # List of class names found
display_classes = {} # Dictionary mapping raw class names to display names

# --- Functions ---

def uploadDataset():
    """Opens a dialog to select the main dataset folder."""
    global filename
    # Ask user to select a directory containing class subfolders
    # Uses '.' (current directory) as the starting point
    new_filename = filedialog.askdirectory(initialdir=".", title="Select Currency Dataset Folder")
    if new_filename: # Check if a directory was actually selected
        filename = new_filename
        pathlabel.config(text=filename) # Update the label in the GUI
        text.delete('1.0', END)
        text.insert(END, f"Selected dataset folder: {filename}\n\n")
    else:
        # User cancelled the dialog
        pathlabel.config(text="No Dataset Selected")
        text.delete('1.0', END)
        text.insert(END, "Dataset selection cancelled.\n")

def preprocess():
    """
    Loads images, preprocesses them (resize, normalize), splits into train/test sets.
    Handles loading/saving preprocessed data to .npy files.
    WARNING: This function can take time and WILL freeze the GUI.
    """
    global X_train, X_test, y_train, y_test
    global filename
    global X, Y
    global classes, display_classes # Ensure classes are updated globally
    text.delete('1.0', END) # Clear the output text area

    if filename is None:
        text.insert(END, "Error: No dataset directory selected.\n")
        text.insert(END, "Please use 'Upload Currency Dataset' first.\n")
        return

    path = filename # Use the globally selected dataset path

    text.insert(END, f"Starting preprocessing using dataset path: {path}\n")

    # --- Directory and File Paths ---
    if not os.path.exists(path):
        text.insert(END, f"Error: Dataset directory '{path}' does not exist.\n")
        text.insert(END, "Please select a valid dataset directory.\n")
        return

    model_dir = 'model' # Directory to save processed data and model files
    if not os.path.exists(model_dir):
        text.insert(END, f"Creating '{model_dir}' directory for saved data/model...\n")
        try:
            os.makedirs(model_dir)
        except OSError as e:
            text.insert(END, f"Error creating directory {model_dir}: {e}\n")
            return

    # Consistent .npy extension for processed data
    x_file = os.path.join(model_dir, 'X.npy')
    y_file = os.path.join(model_dir, 'Y.npy')
    data_loaded_successfully = False

    # --- Attempt to Load Pre-Saved Data ---
    if os.path.exists(x_file) and os.path.exists(y_file):
        text.insert(END, f"\nAttempting to load pre-saved data from {x_file} and {y_file}...\n")
        try:
            # Reset classes before attempting load, in case of failure
            classes = []
            display_classes = {}

            X = np.load(x_file, allow_pickle=True) # Allow pickle just in case, though not expected for numpy arrays
            Y = np.load(y_file, allow_pickle=True) # Y contains class indices

            # Basic Validation checks on loaded data
            if X.size == 0 or Y.size == 0:
                text.insert(END, "Warning: Loaded X or Y arrays are empty. Regenerating dataset...\n")
                X, Y = [], [] # Reset for regeneration
            elif X.shape[0] != Y.shape[0]:
                text.insert(END, f"Warning: Mismatch in loaded X ({X.shape[0]}) and Y ({Y.shape[0]}) sizes. Regenerating dataset...\n")
                X, Y = [], []
            elif len(X.shape) != 4 or X.shape[1:4] != (64, 64, 3):
                text.insert(END, f"Warning: Invalid loaded X shape {X.shape}. Expected (N, 64, 64, 3). Regenerating dataset...\n")
                X, Y = [], []
            elif len(np.unique(Y)) < 2: # Need at least two classes
                text.insert(END, f"Warning: Less than 2 unique classes in loaded Y ({len(np.unique(Y))}). Regenerating dataset...\n")
                X, Y = [], []
            else:
                # Data shapes seem valid, now repopulate class info
                text.insert(END, f"Loaded data array shapes seem valid. X: {X.shape}, Y: {Y.shape}\n")

                # --- >> NEW: Repopulate class info from loaded data << ---
                try:
                    unique_indices = sorted(np.unique(Y))
                    num_loaded_classes = len(unique_indices)
                    text.insert(END, f"Found {num_loaded_classes} unique indices in loaded Y: {unique_indices}\n")

                    # Attempt to get class names from the original dataset directory structure
                    temp_classes = []
                    if os.path.exists(path): # Check if original dataset path still exists
                        text.insert(END, f"Checking original dataset path '{path}' for class subdirectories...\n")
                        for item in os.listdir(path):
                            item_path = os.path.join(path, item)
                            if os.path.isdir(item_path):
                                temp_classes.append(item)
                        temp_classes = sorted(temp_classes)
                    else:
                        text.insert(END, f"Warning: Original dataset path '{path}' not found. Cannot verify class names.\n")

                    # Use directory names if they exist and match the count, otherwise default
                    if len(temp_classes) == num_loaded_classes:
                         classes = temp_classes # Use names found in directory
                         text.insert(END, f"Successfully matched {num_loaded_classes} directory names to loaded data: {classes}\n")
                    else:
                         # Fallback if directory names don't match loaded data indices/count
                         if temp_classes:
                             text.insert(END, f"Warning: Found {len(temp_classes)} directory names ({temp_classes}), but expected {num_loaded_classes} classes based on loaded Y data.\n")
                         classes = [f"Class_{i}" for i in unique_indices]
                         text.insert(END, f"Using default class names: {classes}\n")

                    # Regenerate display_classes based on the now populated 'classes' list
                    display_classes = {label: label.replace('_', ' ').title() + (" (Original)" if "fake" not in label.lower() and "invalid" not in label.lower() else " (Fake)") for label in classes}
                    text.insert(END, f"Populated class information from loaded data successfully. Display names: {display_classes}\n")
                    data_loaded_successfully = True # Flag that data AND class info were loaded

                except Exception as e_class_load:
                    text.insert(END, f"Warning: Could not repopulate class info from loaded data: {e_class_load}. Regenerating dataset...\n")
                    X, Y = [], [] # Force regeneration if class info can't be derived
                    data_loaded_successfully = False # Ensure we don't skip regeneration
                # --- >> End of New Class Repopulation Logic << ---

        except Exception as e:
            text.insert(END, f"Error loading pre-saved data: {str(e)}. Regenerating dataset...\n")
            X, Y = [], [] # Reset on any loading error
            classes = []  # Ensure classes are reset on load failure
            display_classes = {}
            data_loaded_successfully = False
    else:
        text.insert(END, "\nNo pre-saved dataset found. Generating new dataset from images...\n")
        X, Y = [], [] # Ensure lists are empty if starting fresh
        classes = []
        display_classes = {}

    # --- Regenerate Data if Not Loaded Successfully ---
    if not data_loaded_successfully:
        # Reset classes here too, in case loading failed after partially setting them
        classes = []
        display_classes = {}
        local_X = [] # Use local lists to build the data within this block
        local_Y = []
        found_labels = []

        # Find class labels (subdirectories in the dataset path)
        try:
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path):
                    found_labels.append(item)
        except OSError as e:
             text.insert(END, f"Error reading dataset directory {path}: {e}\n")
             return

        if not found_labels:
            text.insert(END, "Error: No valid class subdirectories found in the dataset folder.\n")
            text.insert(END, "Ensure the folder selected has subdirectories named after classes (e.g., 'real', 'fake').\n")
            return

        classes = sorted(found_labels) # Sort class names for consistent index assignment
        display_classes = {label: label.replace('_', ' ').title() + (" (Original)" if "fake" not in label.lower() and "invalid" not in label.lower() else " (Fake)") for label in classes} # Create user-friendly names
        text.insert(END, f"Found {len(classes)} classes by scanning directory: {', '.join(classes)}\n")
        text.insert(END, f"Display names mapping: {display_classes}\n")

        # Load images from each class subdirectory
        total_images_processed = 0
        for class_index, label in enumerate(classes):
            class_path = os.path.join(path, label)
            text.insert(END, f"Processing class '{label}' (index {class_index}) from: {class_path}\n")
            image_count_for_class = 0
            try:
                for file_name in os.listdir(class_path):
                    file_lower = file_name.lower()
                    # Check for common image file extensions
                    if file_lower.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                        img_path = os.path.join(class_path, file_name)
                        try:
                            img = cv2.imread(img_path)
                            if img is None:
                                text.insert(END, f"  Warning: Failed to load image {img_path} (might be corrupt or not an image).\n")
                                continue # Skip this file

                            # Resize image to the required input size for the CNN (64x64)
                            img_resized = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
                            # Append the processed image and its class index
                            local_X.append(img_resized)
                            local_Y.append(class_index)
                            image_count_for_class += 1

                        except Exception as e:
                            text.insert(END, f"  Error processing file {img_path}: {str(e)}\n")
            except OSError as e:
                 text.insert(END, f"Error reading class directory {class_path}: {e}\n")
                 continue # Skip this class if directory reading fails

            text.insert(END, f"  Processed {image_count_for_class} images for class '{label}'.\n")
            total_images_processed += image_count_for_class

        if not local_X or not local_Y:
            text.insert(END, "\nError: No valid images were loaded from the dataset.\n")
            text.insert(END, "Ensure the class subdirectories contain valid image files.\n")
            return

        # Convert lists to NumPy arrays
        X = np.asarray(local_X)
        Y = np.asarray(local_Y) # Y now contains numerical class indices (0, 1, ...)
        text.insert(END, f"\nGenerated dataset: {X.shape[0]} images total.\n")
        text.insert(END, f"  Raw X shape: {X.shape}, Raw Y shape: {Y.shape}\n")
        text.insert(END, f"  Unique class indices found in Y: {len(np.unique(Y))} -> {np.unique(Y)}\n")

        # --- Save the Newly Generated Data ---
        try:
            np.save(x_file, X)
            np.save(y_file, Y) # Save indices in Y
            text.insert(END, f"Saved newly generated dataset to {x_file} and {y_file}.\n")
        except Exception as e:
            text.insert(END, f"Warning: Error saving generated dataset: {str(e)}\n")
            text.insert(END, "Preprocessing will continue with data in memory.\n")

    # --- Final Preprocessing Steps (Normalization, Shuffling, Splitting) ---
    try:
        # Re-check data integrity after loading or generation
        if X is None or Y is None or X.shape[0] == 0 or Y.shape[0] == 0:
             raise ValueError("X or Y data is missing or empty after loading/generation phase.")
        if X.shape[0] != Y.shape[0]:
             raise ValueError(f"Inconsistent number of samples between X ({X.shape[0]}) and Y ({Y.shape[0]}).")
        if not classes: # Check if classes list is populated
             raise ValueError("Class information (labels) is missing after loading/generation phase.")

        text.insert(END, f"\nStarting final preprocessing steps on {X.shape[0]} samples...\n")

        # Check for sufficient data before splitting
        if X.shape[0] < 2:
            raise ValueError("Too few images loaded (< 2) to perform train/test split.")

        num_unique_classes = len(np.unique(Y)) # Y still holds indices here
        if num_unique_classes < 2:
            raise ValueError(f"Only {num_unique_classes} unique class(es) found. Need at least 2 distinct classes for training.")

        # Ensure number of classes matches the populated list length
        num_classes_list = len(classes)
        if num_classes_list != num_unique_classes:
             raise ValueError(f"Mismatch between number of unique indices in Y ({num_unique_classes}) and length of populated classes list ({num_classes_list}).")

        # Normalize pixel values from [0, 255] to [0.0, 1.0]
        X = X.astype('float32') / 255.0
        text.insert(END, "Normalized image pixel values (X) to [0.0, 1.0].\n")

        # Shuffle data - important for train/test split and training batches
        # Create an array of indices, shuffle them, and apply to both X and Y
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        Y = Y[indices] # Y still contains original class indices (0, 1, ...)
        text.insert(END, "Shuffled X and Y dataset.\n")

        # Convert Y (class indices) to one-hot encoded format for the model
        num_classes = len(classes) # Use length of the populated classes list
        Y_categorical = to_categorical(Y, num_classes=num_classes)
        text.insert(END, f"Converted Y from indices to one-hot categorical format. Y_categorical shape: {Y_categorical.shape}\n")

        # Perform train/test split
        # Use stratify=Y (the original indices) to ensure classes are proportionally represented in both sets
        test_split_size = 0.2 # 20% for testing
        X_train, X_test, y_train, y_test = train_test_split(X, Y_categorical,
                                                            test_size=test_split_size,
                                                            random_state=42, # For reproducibility
                                                            stratify=Y) # Stratify based on original indices
        text.insert(END, f"Performed train/test split ({int((1-test_split_size)*100)}%/{int(test_split_size*100)}%):\n")
        text.insert(END, f"  X_train shape: {X_train.shape}, X_test shape: {X_test.shape}\n")
        text.insert(END, f"  y_train shape: {y_train.shape} (one-hot), y_test shape: {y_test.shape} (one-hot)\n")

        text.insert(END, f"\nPreprocessing Summary:\n")
        text.insert(END, f"  Total images processed: {X.shape[0]}\n")
        text.insert(END, f"  Training images: {X_train.shape[0]}\n")
        text.insert(END, f"  Testing images: {X_test.shape[0]}\n")
        text.insert(END, f"  Number of classes: {num_classes}\n")
        text.insert(END, "Preprocessing and splitting complete.\n")

        # Optional: Display a sample processed image
        if X_train.shape[0] > 0:
            # Display first training image (after normalization, convert back for display)
            sample_display = (X_train[0] * 255).astype(np.uint8)
            sample_display_resized = cv2.resize(sample_display, (200, 200)) # Resize for display window
            cv2.imshow("Sample Processed Training Image", sample_display_resized)
            cv2.waitKey(1) # Show briefly (required for imshow to update)
            # Consider adding cv2.destroyWindow("Sample Processed Training Image") later if needed
        else:
             text.insert(END, "Warning: No training images available to display sample.\n")

    except ValueError as ve:
        text.insert(END, f"\nValueError during final preprocessing steps: {str(ve)}\n")
        # Reset potentially inconsistent global state if error occurs mid-process
        X_train, X_test, y_train, y_test = None, None, None, None
        classes = []
        display_classes = {}
        return
    except IndexError as ie:
        text.insert(END, f"\nIndexError during final preprocessing steps: {str(ie)}\n")
        X_train, X_test, y_train, y_test = None, None, None, None
        classes = []
        display_classes = {}
        return
    except Exception as e:
        text.insert(END, f"\nUnexpected error during final preprocessing steps: {str(e)}\n")
        X_train, X_test, y_train, y_test = None, None, None, None
        classes = []
        display_classes = {}
        return

def trainCNN():
    """
    Trains the Convolutional Neural Network model.
    Loads existing model if available and compatible, otherwise trains a new one.
    Saves the trained model and history. Evaluates on the test set.
    WARNING: This function can take a long time and WILL freeze the GUI.
    """
    global model, X_train, X_test, y_train, y_test
    global classes # Need class information for model output layer size
    text.delete('1.0', END)

    # --- Prerequisite Checks ---
    if y_train is None or y_test is None or X_train is None or X_test is None:
        text.insert(END, "Error: Dataset has not been successfully preprocessed and split.\n")
        text.insert(END, "Please run 'Preprocess Dataset' first.\n")
        return

    if not classes:
         text.insert(END, "Error: Class information not available (list is empty).\n")
         text.insert(END, "Preprocessing might have failed or needs to be run first.\n")
         return

    num_classes = len(classes)
    if y_train.shape[1] != num_classes:
        text.insert(END, f"Error: Mismatch between number of classes derived from preprocessing ({num_classes}) ")
        text.insert(END, f"and the shape of the one-hot encoded training labels ({y_train.shape[1]}).\n")
        text.insert(END, "Please check dataset structure and re-run preprocessing.\n")
        return

    # --- Model Paths and Directory ---
    model_dir = 'model'
    if not os.path.exists(model_dir):
        try:
            os.makedirs(model_dir)
            text.insert(END, f"Created '{model_dir}' directory.\n")
        except OSError as e:
            text.insert(END, f"Error creating directory {model_dir}: {e}\n")
            return

    model_json_path = os.path.join(model_dir, 'model.json')
    model_weights_path = os.path.join(model_dir, 'model_weights.h5')
    history_path = os.path.join(model_dir, 'cnn_history.pkl') # To save training history

    # --- CNN Model Definition ---
    def build_model(num_classes_local, input_shape=(64, 64, 3)):
        """Defines the CNN architecture."""
        cnn_model = Sequential(name="FakeCurrencyCNN")
        # Input Layer: Specify the shape of input images
        cnn_model.add(Input(shape=input_shape, name="input_image"))

        # Convolutional Block 1
        cnn_model.add(Conv2D(32, (3, 3), padding='same', activation='relu', name="conv2d_1"))
        cnn_model.add(MaxPooling2D((2, 2), name="maxpool_1"))

        # Convolutional Block 2
        cnn_model.add(Conv2D(64, (3, 3), padding='same', activation='relu', name="conv2d_2"))
        cnn_model.add(MaxPooling2D((2, 2), name="maxpool_2"))

        # Convolutional Block 3
        cnn_model.add(Conv2D(128, (3, 3), padding='same', activation='relu', name="conv2d_3"))
        cnn_model.add(MaxPooling2D((2, 2), name="maxpool_3"))

        # Flatten the output for the Dense layers
        cnn_model.add(Flatten(name="flatten"))

        # Fully Connected (Dense) Layers
        cnn_model.add(Dense(256, activation='relu', name="dense_1"))
        cnn_model.add(Dropout(0.5, name="dropout")) # Dropout for regularization

        # Output Layer: num_classes neurons, softmax activation for multi-class probability
        cnn_model.add(Dense(num_classes_local, activation='softmax', name="output_softmax"))

        # Print model summary to the text widget
        text.insert(END, "\n--- Model Architecture ---\n")
        cnn_model.summary(print_fn=lambda x: text.insert(END, x + '\n'))
        text.insert(END, "-------------------------\n")
        return cnn_model

    # --- Load or Train Model ---
    model_loaded_from_file = False
    try:
        # Check if both model structure and weights files exist
        if os.path.exists(model_json_path) and os.path.exists(model_weights_path):
            text.insert(END, f"\nFound existing model files:\n  {model_json_path}\n  {model_weights_path}\nAttempting to load...\n")
            with open(model_json_path, "r") as json_file:
                loaded_model_json = json_file.read()

            # Load model structure from JSON
            loaded_model = model_from_json(loaded_model_json)

            # CRITICAL CHECK: Ensure loaded model's output matches current dataset's classes
            if loaded_model.output_shape[-1] == num_classes:
                model = loaded_model # Assign to global model variable
                model.load_weights(model_weights_path) # Load weights
                # Compile the model after loading weights - needed for evaluation/prediction
                model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                text.insert(END, "Loaded existing model structure and weights successfully.\n")
                text.insert(END, "Model is ready for evaluation or prediction.\n")
                model_loaded_from_file = True
            else:
                # Mismatch in output layer size - need to retrain
                text.insert(END, f"Warning: Existing model output shape ({loaded_model.output_shape[-1]}) ")
                text.insert(END, f"does not match required classes ({num_classes}).\n")
                text.insert(END, "Deleting old model files and training a new model...\n")
                # Attempt to remove old files before retraining
                if os.path.exists(model_json_path): os.remove(model_json_path)
                if os.path.exists(model_weights_path): os.remove(model_weights_path)
                if os.path.exists(history_path): os.remove(history_path)
                model = None # Reset global model variable
        else:
             text.insert(END, "\nModel files not found. Will build and train a new model.\n")

    except Exception as e:
        text.insert(END, f"\nError loading existing model: {str(e)}\n")
        text.insert(END, "Deleting potentially corrupt model files and training a new model...\n")
        # Attempt to remove potentially corrupt files
        if os.path.exists(model_json_path): os.remove(model_json_path)
        if os.path.exists(model_weights_path): os.remove(model_weights_path)
        if os.path.exists(history_path): os.remove(history_path)
        model = None # Reset global model variable

    # --- Train a New Model if Not Loaded ---
    if not model_loaded_from_file:
        text.insert(END, "\n--- Building and Training New CNN Model ---\n")
        try:
            # Build the model structure
            model = build_model(num_classes)
            # Compile the model: specify optimizer, loss function, and metrics
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            text.insert(END, "Model built and compiled successfully.\n")

            text.insert(END, "Starting model training...\n")
            text.insert(END, "(This may take time and the GUI will be unresponsive)\n")
            text.update_idletasks() # Force GUI update before starting fit

            # Train the model using the training data
            # validation_split: Use part of the training data for validation during training
            # epochs: Number of times to iterate over the entire training dataset
            # batch_size: Number of samples per gradient update
            # verbose=2: Print one line per epoch
            hist = model.fit(X_train, y_train,
                             batch_size=32,
                             epochs=20,     # Adjust epochs as needed (more epochs might improve accuracy but risk overfitting)
                             validation_split=0.1, # Use 10% of training data for validation
                             shuffle=True, # Shuffle training data before each epoch
                             verbose=2)

            text.insert(END, "\n--- Training Finished ---\n")

            # --- Save the Newly Trained Model and History ---
            text.insert(END, "Saving model structure, weights, and training history...\n")
            try:
                model.save_weights(model_weights_path) # Save weights
                model_json = model.to_json() # Save structure
                with open(model_json_path, "w") as json_file:
                    json_file.write(model_json)
                # Save history object (contains loss/accuracy per epoch)
                with open(history_path, "wb") as f:
                    pickle.dump(hist.history, f)
                text.insert(END, "Model and history saved successfully.\n")
            except Exception as e:
                 text.insert(END, f"Error saving trained model/history: {str(e)}\n")
                 # Model might still be usable in memory for evaluation

        except Exception as e:
             text.insert(END, f"\nAn error occurred during model building or training: {str(e)}\n")
             model = None # Ensure model is None if training failed
             return # Stop if training failed

    # --- Evaluation Phase ---
    if model is None:
         text.insert(END, "\nError: Model is not available (None). Cannot evaluate.\n")
         return

    text.insert(END, "\n--- Evaluating Model Performance on Test Data ---\n")
    try:
        # Evaluate the model on the unseen test data
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        text.insert(END, f'CNN Model Test Loss     : {loss:.4f}\n')
        text.insert(END, f'CNN Model Test Accuracy : {accuracy*100:.2f}%\n')

        # Get predictions for the test set
        predict_probabilities = model.predict(X_test)
        # Convert probabilities to class indices (highest probability)
        predict_classes_indices = np.argmax(predict_probabilities, axis=1)
        # Convert true one-hot test labels to class indices
        y_true_classes_indices = np.argmax(y_test, axis=1)

        # Calculate Precision, Recall, F1-Score
        # 'weighted' averages account for class imbalance. 'macro' treats all classes equally.
        avg_method = 'weighted'
        p = precision_score(y_true_classes_indices, predict_classes_indices, average=avg_method, zero_division=0) * 100
        r = recall_score(y_true_classes_indices, predict_classes_indices, average=avg_method, zero_division=0) * 100
        f1 = f1_score(y_true_classes_indices, predict_classes_indices, average=avg_method, zero_division=0) * 100

        text.insert(END, f'\nMetrics (average=\'{avg_method}\'):\n')
        text.insert(END, f'  Precision : {p:.2f}%\n')
        text.insert(END, f'  Recall    : {r:.2f}%\n')
        text.insert(END, f'  F1 Score  : {f1:.2f}%\n')
        text.update_idletasks() # Update GUI text area

        # --- Confusion Matrix ---
        text.insert(END, "\nGenerating Confusion Matrix plot...\n")
        conf_matrix = confusion_matrix(y_true_classes_indices, predict_classes_indices)

        # Get display names for the tick labels
        tick_labels = ["Unknown"] * num_classes # Initialize with placeholders
        if display_classes and len(classes) == num_classes:
            try:
                 # Ensure classes indices match range
                 tick_labels = [display_classes.get(classes[i], f"Class {i}") for i in range(num_classes)]
            except IndexError:
                 text.insert(END, "Warning: Index error getting tick labels. Using default.\n")
                 tick_labels = [f"Class {i}" for i in range(num_classes)]
        else:
            text.insert(END, "Warning: display_classes or classes mismatch. Using default tick labels.\n")
            tick_labels = [f"Class {i}" for i in range(num_classes)]


        plt.figure(figsize=(max(6, num_classes * 0.8), max(5, num_classes * 0.6))) # Dynamic size
        sns.heatmap(conf_matrix, xticklabels=tick_labels, yticklabels=tick_labels,
                    annot=True, cmap="viridis", fmt="d") # "d" format for integer counts
        plt.title("CNN Model Confusion Matrix (Test Set)")
        plt.ylabel('True Class')
        plt.xlabel('Predicted Class')
        plt.xticks(rotation=45, ha='right') # Rotate x-axis labels if they are long
        plt.yticks(rotation=0)
        plt.tight_layout() # Adjust plot to prevent labels overlapping
        plt.show() # Display the plot in a separate window

    except Exception as e:
        text.insert(END, f"\nAn error occurred during model evaluation or plotting: {str(e)}\n")


def graph():
    """Loads and plots the training history (accuracy and loss)."""
    history_path = os.path.join('model', 'cnn_history.pkl')
    text.delete('1.0', END)
    text.insert(END, f"Attempting to load training history from: {history_path}\n")

    try:
        if not os.path.exists(history_path):
             raise FileNotFoundError(f"History file '{history_path}' not found. Please train the CNN model first.")

        with open(history_path, 'rb') as f:
            history_data = pickle.load(f)

        # Check if essential keys exist in the loaded history dictionary
        acc = history_data.get('accuracy')
        loss = history_data.get('loss')
        val_acc = history_data.get('val_accuracy') # Might be None if validation_split wasn't used
        val_loss = history_data.get('val_loss')     # Might be None

        if acc is None or loss is None:
            text.insert(END, "Error: History file does not contain required 'accuracy' or 'loss' keys.\n")
            return

        text.insert(END, "History loaded successfully. Generating plots...\n")
        epochs_range = range(1, len(acc) + 1)

        plt.figure(figsize=(14, 6)) # Adjust figure size

        # --- Accuracy Plot ---
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, 'o-', color='royalblue', label='Training Accuracy')
        if val_acc:
            plt.plot(epochs_range, val_acc, 'o-', color='forestgreen', label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(loc='best') # Adjust legend location automatically
        plt.grid(True, linestyle='--', alpha=0.7)

        # --- Loss Plot ---
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, 'o-', color='orangered', label='Training Loss')
        if val_loss:
            plt.plot(epochs_range, val_loss, 'o-', color='darkorange', label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.suptitle('CNN Model Training History', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for suptitle
        plt.show() # Display the plot

    except FileNotFoundError as fnf_error:
        text.insert(END, f"\nError: {fnf_error}\n")
    except Exception as e:
        text.insert(END, f"\nError loading or plotting history: {str(e)}\n")

def predict():
    """
    Loads the trained model, asks user for an image, preprocesses it,
    predicts the class, and displays the image with the prediction result.
    """
    global classes, display_classes, model # Need access to these globals

    text.delete('1.0', END)
    model_dir = 'model'
    model_json_path = os.path.join(model_dir, 'model.json')
    model_weights_path = os.path.join(model_dir, 'model_weights.h5')

    # --- Check Prerequisites ---
    if not classes:
        text.insert(END, "Error: Class information is missing (list is empty).\n")
        text.insert(END, "Please run 'Preprocess Dataset' first to define classes.\n")
        return

    num_classes = len(classes)

    # --- Load Model (if not already loaded or needs reload) ---
    # Check if the global 'model' variable is already loaded and valid
    # Simple check: is it None? More complex checks could verify output shape again.
    if model is None:
        text.insert(END, "Model not found in memory. Attempting to load from file...\n")
        try:
            if not (os.path.exists(model_json_path) and os.path.exists(model_weights_path)):
                raise FileNotFoundError(f"Model files ({model_json_path}, {model_weights_path}) not found. Please train the CNN model first using 'Train CNN Model'.")

            text.insert(END, f"Loading model structure from: {model_json_path}\n")
            with open(model_json_path, "r") as json_file:
                loaded_model_json = json_file.read()

            loaded_model = model_from_json(loaded_model_json)

            # Verify output shape consistency after loading structure
            if loaded_model.output_shape[-1] != num_classes:
                raise ValueError(f"Loaded model structure's output shape ({loaded_model.output_shape[-1]}) does not match the current dataset's class count ({num_classes}). Please retrain the model.")

            text.insert(END, f"Loading model weights from: {model_weights_path}\n")
            loaded_model.load_weights(model_weights_path)
            model = loaded_model # Assign to global variable if successful
            # Optional: Compile model if needed right after loading, e.g., for immediate evaluation
            # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            text.insert(END, "Model loaded successfully and ready for prediction.\n")

        except FileNotFoundError as fnf:
            text.insert(END, f"Error: {str(fnf)}\n")
            model = None # Ensure model is None if loading failed
            return # Stop if model files are missing
        except ValueError as ve:
             text.insert(END, f"Error: {str(ve)}\n")
             model = None
             return # Stop if model shape is wrong
        except Exception as e:
            text.insert(END, f"Error loading model: {str(e)}. Please ensure model files exist and are valid.\n")
            model = None # Ensure model is None if loading failed
            return # Stop on other loading errors
    else:
        text.insert(END, "Using already loaded model for prediction.\n")


    # --- Select and Process Image for Prediction ---
    # Ask user to select an image file
    pred_filename = filedialog.askopenfilename(initialdir=".", # Start in current directory
                                             title="Select Currency Image for Prediction",
                                             filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.gif")])
    if not pred_filename:
        text.insert(END, "Prediction cancelled: No image selected.\n")
        return

    text.insert(END, f"\nSelected image: {pred_filename}\n")

    try:
        # Load the selected image using OpenCV
        image = cv2.imread(pred_filename)
        if image is None:
            raise ValueError(f"Failed to load image file '{pred_filename}'. It might be corrupt, not an image, or the path is invalid.")

        # Preprocess the image: MUST match the preprocessing steps used for training data
        # 1. Resize to 64x64
        img_resized = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
        # 2. Normalize pixel values to [0.0, 1.0]
        img_normalized = img_resized.astype('float32') / 255.0
        # 3. Add batch dimension (model expects input shape like (batch_size, height, width, channels))
        img_expanded = np.expand_dims(img_normalized, axis=0) # Shape becomes (1, 64, 64, 3)

        # --- Make Prediction ---
        text.insert(END, "Preprocessing complete. Predicting class...\n")
        predictions_probabilities = model.predict(img_expanded)
        # predictions_probabilities will be like [[prob_class0, prob_class1, ...]]

        # Get the index of the class with the highest probability
        predicted_index = np.argmax(predictions_probabilities[0])
        # Get the highest probability score (confidence)
        confidence = np.max(predictions_probabilities[0]) * 100

        # Map the predicted index back to the class label
        if predicted_index < 0 or predicted_index >= len(classes):
             raise IndexError(f"Predicted index {predicted_index} is out of bounds for the 'classes' list (size {len(classes)}). Model or class list might be inconsistent.")

        result_label_raw = classes[predicted_index]
        # Use the display name mapping for user-friendly output
        display_label = display_classes.get(result_label_raw, result_label_raw) # Fallback to raw label if not found

        text.insert(END, f"\n--- Prediction Result ---\n")
        text.insert(END, f"  Predicted Class: {display_label} (Raw: '{result_label_raw}', Index: {predicted_index})\n")
        text.insert(END, f"  Confidence: {confidence:.2f}%\n")
        text.insert(END, "-------------------------\n")

        # --- Display Result on Image ---
        # Resize the original image for better display (don't use the 64x64 one)
        display_image_size = (400, 400)
        output_image = cv2.resize(image, display_image_size)

        # Prepare text to overlay on the image
        result_text = f"{display_label} ({confidence:.1f}%)"

        # Choose text color based on prediction (e.g., green for likely real, red for fake)
        # This logic assumes 'fake' or 'invalid' in the raw label indicates a fake note. Adjust as needed.
        text_color = (0, 0, 255) # BGR format: Red default
        if "fake" not in result_label_raw.lower() and "invalid" not in result_label_raw.lower():
            text_color = (0, 255, 0) # Green for likely real

        # Add the text to the image
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        text_position = (10, 30) # (x, y) from top-left corner
        cv2.putText(output_image, result_text, text_position, font_face, font_scale, text_color, thickness, cv2.LINE_AA)

        # Show the image with the prediction in an OpenCV window
        window_title = f"Prediction: {display_label}"
        cv2.imshow(window_title, output_image)
        cv2.waitKey(0) # Wait indefinitely until a key is pressed
        cv2.destroyWindow(window_title) # Close only this specific window

    except FileNotFoundError: # Should be caught by askopenfilename, but good practice
        text.insert(END, f"Error: Image file not found at path: {pred_filename}\n")
    except ValueError as ve:
         text.insert(END, f"Error processing image or predicting: {str(ve)}\n")
    except IndexError as ie:
        text.insert(END, f"Error accessing classes list during prediction: {str(ie)}\n Check model consistency with preprocessed data.\n")
    except Exception as e:
        text.insert(END, f"An unexpected error occurred during prediction: {str(e)}\n")
        # Ensure any opened window is closed on generic error
        cv2.destroyAllWindows()


def close():
    """Closes the Tkinter application and any remaining OpenCV windows."""
    text.insert(END, "\nExiting application...\n")
    # Clean up any potentially lingering OpenCV windows
    cv2.destroyAllWindows()
    main.destroy()

# --- GUI Setup ---
# Main window title label
font_title = ('times', 25, 'bold')
title = Label(main, text='Fake Currency Detection System')
title.config(bg="darkgreen", fg="yellow")
title.config(font=font_title)
title.config(height=2, width=70)
# Use pack for the main title for better resizing behavior (optional, place is also fine)
title.pack(pady=10, fill=X, padx=10)
# title.place(x=5, y=5) # Alternative using place

# Frame to hold the buttons
font_button = ('times', 13, 'bold')
button_frame = Frame(main, bg='darkgreen', bd=2, relief=SUNKEN)
# Use pack or grid for the button frame relative to the title
button_frame.pack(pady=10, padx=10, fill=X)
# button_frame.place(x=50, y=100, width=700, height=150) # Alternative using place

# --- Buttons within the Frame (using grid layout) ---
# Row 0
uploadButton = Button(button_frame, text="Upload Currency Dataset", command=uploadDataset)
uploadButton.config(bg="red", fg="black", font=font_button, width=25) # Fixed width for alignment
uploadButton.grid(row=0, column=0, padx=10, pady=10, sticky='ew')

pathlabel = Label(button_frame, text="No Dataset Selected", anchor='w') # anchor west (left-align)
pathlabel.config(bg='darkgreen', fg='yellow', font=font_button)
pathlabel.grid(row=0, column=1, columnspan=2, padx=10, pady=10, sticky='ew')

# Row 1
processButton = Button(button_frame, text="Preprocess Dataset", command=preprocess)
processButton.config(bg="black", fg="white", font=font_button, width=25)
processButton.grid(row=1, column=0, padx=10, pady=10, sticky='ew')

cnnButton = Button(button_frame, text="Train CNN Model", command=trainCNN)
cnnButton.config(bg="black", fg="white", font=font_button, width=25)
cnnButton.grid(row=1, column=1, padx=10, pady=10, sticky='ew')

# Row 2
graphButton = Button(button_frame, text="Show Training Graph", command=graph)
graphButton.config(bg="black", fg="white", font=font_button, width=25)
graphButton.grid(row=2, column=0, padx=10, pady=10, sticky='ew')

predictButton = Button(button_frame, text="Predict Currency Image", command=predict)
predictButton.config(bg="black", fg="white", font=font_button, width=25)
predictButton.grid(row=2, column=1, padx=10, pady=10, sticky='ew')

# Row 3 (or merge into row 2 if space allows)
# Changed column=2 to column=0 and columnspan=3 to columnspan=2 for Exit button based on previous grid
exitButton = Button(button_frame, text="Exit Application", command=close)
exitButton.config(bg="black", fg="white", font=font_button, width=25)
exitButton.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky='ew') # Adjusted grid position


# Configure column weights in button_frame so they expand proportionally
button_frame.grid_columnconfigure(0, weight=1)
button_frame.grid_columnconfigure(1, weight=1)
# No need for column 2 weight if exit button spans 0 and 1


# --- Text Output Area ---
font_text = ('consolas', 10) # Use a monospaced font for console-like output
# Create a frame to hold the text widget and scrollbars
text_frame = Frame(main, bd=2, relief=SUNKEN)
# Use pack to place it below the button frame and allow it to fill remaining space
text_frame.pack(pady=10, padx=10, fill=BOTH, expand=True)
# text_frame.place(x=10, y=270, width=1180, height=520) # Alternative using place

# Text widget for output messages
text = Text(text_frame, wrap=WORD, height=15) # wrap=WORD prevents horizontal scrollbar mostly
scroll_y = Scrollbar(text_frame, orient=VERTICAL, command=text.yview)
scroll_x = Scrollbar(text_frame, orient=HORIZONTAL, command=text.xview)
text.config(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set, bg="lightgray", fg="black")
text.config(font=font_text)

# Pack scrollbars and text widget within their frame
scroll_y.pack(side=RIGHT, fill=Y)
scroll_x.pack(side=BOTTOM, fill=X)

text.pack(side=LEFT, fill=BOTH, expand=True)

# --- Initial Message ---
text.insert(END, "Welcome to the Fake Currency Detection System.\n")
text.insert(END, "1. Upload the dataset folder.\n")
text.insert(END, "2. Preprocess the dataset.\n")
text.insert(END, "3. Train the CNN model.\n")
text.insert(END, "4. Optionally view the training graph.\n")
text.insert(END, "5. Predict on a new currency image.\n\n")
text.insert(END, "NOTE: Preprocessing and Training can take time and may freeze the GUI.\n")

# --- Main Loop ---
main.config(bg='darkgreen') # Set background for the main window itself
main.mainloop()
