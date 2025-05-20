from tensorflow.keras.models import load_model

print("Loading model...")
model = load_model("currency_model.h5")
print("Model loaded successfully.")
