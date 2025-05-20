
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score # Import necessary metrics
# Ensure Keras is imported correctly based on your TensorFlow version
try:
    # TensorFlow 2.x style
    from tensorflow.keras.models import Sequential
    from tensorflow.keras import Input
    from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.models import save_model
except ImportError:
    # Fallback for older standalone Keras
    from keras.models import Sequential
    from keras import Input
    from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
    from keras.utils import to_categorical
    from keras.callbacks import EarlyStopping
    from keras.models import save_model

# --- Suppress TensorFlow/Keras logging ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Set to 3 to filter ERROR messages too (most aggressive)
import logging
logging.getLogger('tensorflow').setLevel(logging.FATAL) # Suppress TF python logging
import warnings
warnings.filterwarnings('ignore') # Suppress general warnings (use with caution)
# --- End Suppress Logging ---

import pickle
import re

# Load dataset with minimal output
def loadDataset(dataset_path):
    data, labels = [], []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    pattern_to_skip = re.compile(r'^\..*')

    try:
        if not os.path.exists(dataset_path):
            # Keep critical error reporting
            print(f"CRITICAL ERROR: The directory '{dataset_path}' was not found.")
            return None, None, None # Indicate failure

        class_names = sorted([d for d in os.listdir(dataset_path)
                             if os.path.isdir(os.path.join(dataset_path, d))])

        if not class_names:
            # Keep critical error reporting
            print(f"CRITICAL ERROR: No class directories found in {dataset_path}")
            return None, None, None

        # --- SILENCED --- print(f"Found {len(class_names)} classes: {', '.join(class_names)}")

        for label, folder in enumerate(class_names):
            folder_path = os.path.join(dataset_path, folder)
            for img_name in os.listdir(folder_path):
                if (pattern_to_skip.match(img_name) or
                    not img_name.lower().endswith(valid_extensions)):
                    continue

                img_path = os.path.join(folder_path, img_name)
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, (64, 64))
                        data.append(img)
                        labels.append(label)
                    # else: # --- SILENCED --- Warning about failed load
                        # pass
                except Exception as e:
                    # --- SILENCED --- Error processing individual file
                    # print(f"Error processing {img_path}: {e}")
                    pass # Ignore individual file errors to minimize output

            # --- SILENCED --- Per-class summary print
            # print(f"Class '{folder}': Processed...")

    except Exception as e:
        # Keep critical error reporting
        print(f"CRITICAL Dataset loading error: {e}")
        raise # Re-raise critical errors

    if not data:
        # Keep critical error reporting
        print("CRITICAL ERROR: No images were loaded. Please check the dataset path and image formats.")
        return None, None, None

    # --- SILENCED --- print(f"Total images loaded: {len(data)}")
    data = np.array(data) / 255.0
    labels = to_categorical(labels, num_classes=len(class_names))
    return train_test_split(data, labels, test_size=0.2, random_state=42), class_names

# --- Main Script Execution ---

# Define dataset path
dataset_path = "Dataset2(Final)" # Make sure this path is correct

# Load the dataset
dataset_result, classes = loadDataset(dataset_path)
if dataset_result is None:
    exit(1) # Exit if loading failed critically
(X_train, X_test, y_train, y_test) = dataset_result
# --- SILENCED --- print(f"Dataset split: ...")

# Build CNN model
num_classes = len(classes)
if num_classes == 0:
    print("CRITICAL ERROR: No classes found or loaded. Cannot build model.")
    exit(1)

model = Sequential()
model.add(Input(shape=(64, 64, 3)))
model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# --- SILENCED --- Model Summary
# print("\n--- Model Summary ---")
# model.summary(print_fn=lambda x: None) # Attempt to silence summary if needed
# print("--------------------\n")

# Train the model with early stopping - SILENTLY
# --- SILENCED --- print("--- Starting Model Training ---")
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0) # verbose=0 for silence
hist = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=50,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=0 # Set verbose to 0 for complete silence during training
)
# --- SILENCED --- print("--- Model Training Finished ---\n")


# --- Calculate Metrics on Training Data ---
# This is the part you specifically requested output for
if len(X_train) > 0:
    predictions_train = model.predict(X_train, verbose=0) # verbose=0 for silence
    predicted_train_classes = np.argmax(predictions_train, axis=1)
    true_train_classes = np.argmax(y_train, axis=1) # Convert one-hot y_train

    # Calculate metrics for the training set
    train_accuracy = accuracy_score(true_train_classes, predicted_train_classes)
    train_precision = precision_score(true_train_classes, predicted_train_classes, average='macro', zero_division=0)
    # train_recall = recall_score(true_train_classes, predicted_train_classes, average='macro', zero_division=0) # Calculated but not printed
    # train_f1 = f1_score(true_train_classes, predicted_train_classes, average='macro', zero_division=0) # Calculated but not printed

    # --- PRINT THE REQUESTED TRAINING SCORES ---
    print(f"Training Accuracy: {train_accuracy*100:.2f}%")
    print(f"Training Precision (Macro Avg): {train_precision*100:.2f}%")
    # --- END OF REQUESTED OUTPUT ---

else:
    print("CRITICAL WARNING: Training set empty, cannot calculate training metrics.")


# --- All subsequent operations (saving, testing, plotting) are silenced ---

# --- SILENCED --- Saving Model
# os.makedirs('model', exist_ok=True)
# model_save_path = 'model/fake_currency_detector.keras'
# try:
#     if 'save_model' in globals(): save_model(model, model_save_path)
#     else: model.save(model_save_path)
#     # print(f"✅ Model saved successfully at '{model_save_path}'")
# except Exception as e:
#     # print(f"❌ Error saving model: {e}")
#     pass

# --- SILENCED --- Saving History
# history_save_path = "model/cnn_history.pkl"
# try:
#     with open(history_save_path, "wb") as f: pickle.dump(hist.history, f)
#     # print(f"✅ Training history saved successfully at '{history_save_path}'")
# except Exception as e:
#     # print(f"❌ Error saving history: {e}")
#     pass

# --- SILENCED --- Evaluation on TEST data
# print("\n--- Evaluating model on test data ---")
# if len(X_test) > 0:
#     predictions = model.predict(X_test, verbose=0)
#     predicted_classes = np.argmax(predictions, axis=1)
#     true_classes = np.argmax(y_test, axis=1)
#     test_accuracy = accuracy_score(true_classes, predicted_classes)
#     test_precision = precision_score(true_classes, predicted_classes, average='macro', zero_division=0)
#     test_recall = recall_score(true_classes, predicted_classes, average='macro', zero_division=0)
#     test_f1 = f1_score(true_classes, predicted_classes, average='macro', zero_division=0)
#     # print("\n--- Evaluation Results (Test Set) ---")
#     # print(f"Accuracy  : {test_accuracy*100:.2f}%")
#     # print(f"Precision : {test_precision*100:.2f}% (Macro Avg)")
#     # print(f"Recall    : {test_recall*100:.2f}% (Macro Avg)")
#     # print(f"F1 Score  : {test_f1*100:.2f}% (Macro Avg)")
#     # print("-----------------------------------\n")
# else:
#     # print("⚠️ Test set is empty. Skipping test set evaluation.")
#     pass


# --- SILENCED --- Confusion Matrix generation and saving
# ... (code for confusion matrix removed for brevity, assume it's commented out) ...

# --- SILENCED --- Per-class metrics display
# ... (code for per-class metrics removed for brevity, assume it's commented out) ...

# --- SILENCED --- Plotting training history and saving
# ... (code for plotting removed for brevity, assume it's commented out) ...

# --- SILENCED --- Final completion message
# print("\n✅ Model training and evaluation completed successfully!")
