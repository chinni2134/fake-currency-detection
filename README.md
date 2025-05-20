Fake Currency Detection using CNN

This project uses a Convolutional Neural Network (CNN) model to detect fake currency notes from uploaded images. It features a simple and user-friendly web interface built using Flask.
If not flask you can use vercel 
---
 Project Structure

 fake-currency-detection/
├── app.py                    # Flask app entry point
├── Main.py / MainNew.py      # Model training/inference scripts (Main New.py) 
├── preprocessing.py          # Image preprocessing logic
├── testtrain.py              # Testing and training script
├── requirements.txt          # Python dependencies
├── run.bat                   # Script to run the project (Windows)
├── init_db.py                # Initialize SQLite DB
├── users.db                  # User database (SQLite)(for creating Admin Credentials)
├── static/                   # Static assets (CSS, images)
├── templates/                # HTML templates (Flask UI)
├── model/                    # Trained CNN model files(This automatically gets created in your project folder after the model trained
├── uploads/                  # Uploaded images (by users)
├── testimages2/              # Test images for validation
├── Dataset2(Final)/          # Dataset for training
├── Organized_Dataset/        # Add your Trained Images in the folder  
├── reference papers/         # Research papers (just refer)
├── *.ipynb                   # Jupyter notebooks for experimentation(if you are using anaconda )
├── *.pdf / *.pptx / *.jpg    # Supporting documents & presentations
├── venv/                     # Python virtual environment(download all the python dependencies in the Venv)

---

 Features

- Upload currency note images for classification
- CNN-based model determines if the note is **real** or **fake**
- Clean UI with Flask + HTML/CSS frontend

---
### Getting Started

   ## Clone the Repository
```bash
git clone https://github.com/yourusername/fake-currency-detection.git
cd fake-currency-detection

python3 -m venv venv
source venv/bin/activate   ## For Mac os
pip install -r requirements.txt
      or

pip3 install -r requirements.txt

python app.py

     or

python3 app.py

