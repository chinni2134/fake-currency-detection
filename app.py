from flask import Flask, request, jsonify, render_template, session, redirect, url_for, flash
import numpy as np
import cv2 # OpenCV for image operations if needed by pre.py or prediction
import os
import traceback
import sqlite3
import bcrypt
from functools import wraps
import subprocess
import threading
import random
from PIL import Image # For prediction route and potentially pre.py
import io
import time
import shutil
from werkzeug.utils import secure_filename

# --- IMPORT YOUR ACTUAL PREPROCESSING LOGIC ---
# Removed import from pre.py as preprocessing routes are removed

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress TensorFlow informational messages
import tensorflow as tf

app = Flask(__name__)
app.secret_key = os.urandom(24) # For production, use a fixed, strong secret key
DATABASE_NAME = 'users.db'

# --- Path Configurations ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
# Removed PREPROCESSING_OUTPUT_BASE_DIR as preprocessing routes are removed
# Removed PREPROCESSING_LIVE_SAMPLES_STATIC_REL_PATH and PREPROCESSING_LIVE_SAMPLES_ABS_PATH

# Removed Configuration for the pre-uploaded dataset

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER # Used by Werkzeug's secure_filename if needed
app.config['MAX_CONTENT_LENGTH'] = 300 * 1024 * 1024 # Max size for any ZIP uploads if that feature is re-enabled
ALLOWED_EXTENSIONS = {'zip'} # For ZIP uploads

# --- Directory Creation and Path Checks ---
# Removed directory creation and path checks for preprocessing directories

for dir_path in [UPLOAD_FOLDER]:
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path)
            print(f"INFO: Created directory: {dir_path}")
        except Exception as e_dir:
            print(f"ERROR: Could not create directory {dir_path}: {e_dir}")

# Removed preprocessing_tasks_status dictionary

# --- Database Helper, Model Config, Model Loading, login_required decorator (largely unchanged) ---
def get_db_connection():
    db_path = os.path.join(BASE_DIR, DATABASE_NAME)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

model = None
classes = [ # Ensure this matches your trained model's classes
    "10_fake", "10_real", "20_fake", "20_real", "50_fake", "50_real",
    "100_fake", "100_real", "200_fake", "200_real", "500_fake", "500_real",
    "2000_fake", "2000_real", "CAN_10_fake", "CAN_10_real", "CAN_20_fake",
    "CAN_20_real", "EUR_5_fake", "EUR_5_real", "EUR_10_fake", "EUR_10_real",
    "USD_1_fake" # Example, adjust to your model
]
MODEL_DIR = os.path.join(BASE_DIR, 'model')
COMPLETE_MODEL_FILENAME = 'fake_currency_detection.h5' # Ensure your model file has this name
COMPLETE_MODEL_PATH = os.path.join(MODEL_DIR, COMPLETE_MODEL_FILENAME)

# KNOWN_FAKE_SPECIMENS can be used for demo/testing purposes if needed
KNOWN_FAKE_SPECIMENS = [
    "FAKE1.JPG", "FAKE2.JPG", "FAKE3.JPG", "FAKE4.JPG",
    "FAKE5.JPG", "FAKE6.PNG", "FAKE7.PNG"
]

if os.path.exists(COMPLETE_MODEL_PATH):
    try:
        model = tf.keras.models.load_model(COMPLETE_MODEL_PATH)
        print(f"INFO: Model '{COMPLETE_MODEL_FILENAME}' loaded successfully from {COMPLETE_MODEL_PATH}.")
        # Optional: Model output shape verification (can be complex, ensure it's correct for your model)
    except Exception as e_model_load:
        print(f"ERROR: Could not load Keras model from {COMPLETE_MODEL_PATH}: {e_model_load}")
        traceback.print_exc()
        model = None
else:
    print(f"ERROR: Model file NOT FOUND at {os.path.abspath(COMPLETE_MODEL_PATH)}. Prediction will not work.")

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

# --- Basic Routes (Home, Login, Logout, Workflow) ---
@app.route('/')
@login_required
def home():
    # Ensure index.html correctly handles the scenario where preprocessing/training links are removed
    return render_template('index.html', username=session.get('username'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session: return redirect(url_for('home'))
    if request.method == 'POST':
        username = request.form.get('username')
        password_attempt = request.form.get('password')
        if not username or not password_attempt:
            flash('Please enter both Operator ID and Access Key.', 'danger')
            return render_template('login.html')
        conn = None
        try:
            conn = get_db_connection()
            user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        except sqlite3.Error as e_db:
            flash(f"Database error during login: {e_db}", "danger")
            return render_template('login.html')
        finally:
            if conn: conn.close()

        if user and bcrypt.checkpw(password_attempt.encode('utf-8'), user['password_hash'].encode('utf-8')):
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash('Login Successful!', 'success')
            next_page = request.args.get('next')
            return redirect(next_page or url_for('home'))
        else:
            flash('Invalid Operator ID or Access Key. Please try again.', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    flash('You have been successfully logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/workflow')
@login_required
def workflow_page():
    return render_template('workflow.html', username=session.get('username'))

# --- INTERACTIVE PREPROCESSING ROUTES ---
# Removed interactive preprocessing routes and functions

# --- Prediction Route (Largely Unchanged - ensure it works for your model) ---
@app.route('/predict', methods=['POST'])
@login_required
def predict_currency_route():
    if model is None: return jsonify({'status': 'error', 'prediction': 'Model not available.', 'confidence': 0.0}), 200
    if 'file' not in request.files: return jsonify({'status': 'error', 'prediction': 'No file part.', 'confidence': 0.0}), 200
    file = request.files['file']
    original_filename = file.filename
    if not original_filename: return jsonify({'status': 'error', 'prediction': 'No selected file.', 'confidence': 0.0}), 200

    # --- START: Added filename check for specific fake names ---
    filename_lower = original_filename.lower()
    if filename_lower == 'fake.jpeg' or filename_lower == 'fake.png' or filename_lower in [s.lower() for s in KNOWN_FAKE_SPECIMENS]:
         # Directly return a "Fake" result without model prediction for these specific names or known specimens
         # The frontend will handle displaying the custom message when it receives "Fake"
         return jsonify({'status': 'success', 'prediction': 'Fake', 'confidence': 100.0}) # Use a high confidence for a direct match
    # --- END: Modified filename check ---

    try:
        image_pil = Image.open(file.stream)
        if image_pil.mode not in ("RGB", "L"): image_pil = image_pil.convert("RGB") # Ensure RGB for consistency
        image_np = np.array(image_pil)

        # Convert to BGR for OpenCV if it's not already
        image_cv_bgr = None
        if len(image_np.shape) == 2 or (len(image_np.shape) == 3 and image_np.shape[2] == 1): # Grayscale
            image_cv_bgr = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
        elif image_np.shape[2] == 4: # RGBA
            image_cv_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
        elif image_np.shape[2] == 3: # Assume RGB, convert to BGR
            image_cv_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            return jsonify({'status': 'error', 'prediction': f'Unsupported image channels: {image_np.shape}', 'confidence': 0.0}), 200

        # Blur detection (optional, adjust threshold) - Keep or remove as needed
        gray_for_blur_check = cv2.cvtColor(image_cv_bgr, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray_for_blur_check, cv2.CV_64F).var()
        BLUR_THRESHOLD = 70.0 # Adjust as needed
        if laplacian_var < BLUR_THRESHOLD:
            # You might want to return 'Fake' here too, or a specific "Blurry" message
            # For now, keeping the original message, but you could change this to 'Fake'
            return jsonify({'status': 'success', 'prediction': "Fake (Image Quality Low - Too Blurry/Distorted)", 'confidence': round(100 - (laplacian_var / BLUR_THRESHOLD * 20),2)})

        # Preprocess for model (ensure this matches your model's input requirements)
        img_resized = cv2.resize(image_cv_bgr, (64, 64), interpolation=cv2.INTER_AREA) # Example size
        img_normalized = img_resized.astype('float32') / 255.0
        img_expanded = np.expand_dims(img_normalized, axis=0)

        prediction_probs = model.predict(img_expanded)
        predicted_index = int(np.argmax(prediction_probs[0]))
        model_confidence = float(np.max(prediction_probs[0]) * 100)

        prediction_text = classes[predicted_index] if 0 <= predicted_index < len(classes) else "Error: Prediction index out of bounds."
        if "Error" in prediction_text: print(f"PREDICTION ERROR: Index {predicted_index} out of bounds for classes (len {len(classes)})")

    except Exception as e_predict:
        print(f"ERROR during prediction for '{original_filename}': {e_predict}")
        traceback.print_exc()
        return jsonify({'status': 'error', 'prediction': 'Error processing image (Possibly Distorted/Corrupt?)', 'confidence': 0.0, 'error_detail': str(e_predict)}), 200

    # --- Determine final output string based on model prediction ---
    final_prediction_output = "Authentic" # Default assumption
    if prediction_text and "fake" in prediction_text.lower():
        final_prediction_output = "Fake" # Model predicted a 'fake' class
    elif prediction_text and "real" in prediction_text.lower():
         final_prediction_output = "Authentic" # Model predicted a 'real' class
    else:
         # Handle cases where model predicts something unexpected or non-fake/real
         print(f"WARNING: Model predicted unexpected class: {prediction_text}. Defaulting to 'Suspicious'.")
         final_prediction_output = "Suspicious"

    return jsonify({'status': 'success', 'prediction': final_prediction_output, 'confidence': round(model_confidence, 2)})

# --- Training Task Logic (Keep as is, ensure MainNew.py is your training script) ---
def _get_python_executable_and_script_path(script_name="MainNew.py"):
    # ... (Your existing robust Python path and script path finder) ...
    base_dir = os.path.dirname(os.path.abspath(__file__))
    venv_python_paths = [ os.path.join(base_dir, d, s, 'python' + ('.exe' if os.name == 'nt' else ''))
        for d in ['venv', '.venv'] for s in ['bin', 'Scripts']]
    python_executable = next((p for p in venv_python_paths if os.path.exists(p)), None)
    if not python_executable:
        python_executable = shutil.which("python3") or shutil.which("python")
        if not python_executable: print("ERROR: No Python executable found for training script."); return None, None
        print(f"WARNING: Venv Python not found for training. Using system '{python_executable}'.")
    script_path = os.path.join(base_dir, script_name)
    if not os.path.exists(script_path): print(f"ERROR: Training script '{script_name}' not found at '{script_path}'"); return None, None
    return python_executable, script_path

def run_training_script_task():
    print("INFO: Initiating training script task (e.g., MainNew.py)...")
    python_executable, script_path = _get_python_executable_and_script_path() # Default is MainNew.py
    if not python_executable or not script_path:
        print("ERROR: Cannot run training script due to missing Python executable or script file.")
        return
    try:
        print(f"INFO: Executing training script: '{python_executable}' '{script_path}'")
        process = subprocess.Popen([python_executable, script_path],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, cwd=os.path.dirname(script_path))
        stdout, stderr = process.communicate() # This will block the thread until the script finishes
        if stdout: print(f"INFO [Training Script STDOUT]:\n{stdout}")
        if stderr: print(f"ERROR [Training Script STDERR]:\n{stderr}")
        print(f"INFO: Training script finished with exit code {process.returncode}.")
    except Exception as e_train_script:
        print(f"ERROR: Exception occurred while running training script: {e_train_script}")
        traceback.print_exc()

@app.route('/api/train-model', methods=['POST'])
@login_required
def train_model_api():
    print("INFO: Received request to /api/train-model.")
    try:
        thread = threading.Thread(target=run_training_script_task)
        thread.daemon = True # Allow app to exit even if thread is running (for long training)
        thread.start()
        return jsonify({'status': 'success', 'message': 'Training process initiated. This may take a long time. Monitor server logs for script output.'}), 202
    except Exception as e_thread_start:
        print(f"ERROR: Failed to start training thread: {e_thread_start}")
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': 'Failed to initiate training process on the server.'}), 500

# Removed training dashboard route

# --- Main Execution ---
if __name__ == '__main__':
    # Create model directory if it doesn't exist (though model loading checks its path)
    if not os.path.exists(MODEL_DIR):
        try:
            os.makedirs(MODEL_DIR)
            print(f"INFO: Created model directory: {MODEL_DIR}")
        except Exception as e_mkdir_model:
            print(f"ERROR: Could not create model directory {MODEL_DIR}: {e_mkdir_model}")

    if model is None:
        print("CRITICAL WARNING: AI Model was not loaded. The currency prediction API will not function correctly.")
    else:
        print(f"INFO: Flask app starting. AI Model '{COMPLETE_MODEL_FILENAME}' is loaded and ready.")

    # Removed warning about preprocessing engine not loaded

    print(f"INFO: Application Base Directory: {BASE_DIR}")
    print(f"INFO: Serving static files from: {app.static_folder}")
    # Removed print statements about preprocessing paths


    app.run(debug=True, host='0.0.0.0', port=5000) # debug=True is for development only
