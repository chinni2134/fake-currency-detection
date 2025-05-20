import tensorflow as tf

# Prevent GPU crashes related to memory allocation
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

print("TensorFlow version:", tf.__version__)
print("Num GPUs available:", len(gpus))
