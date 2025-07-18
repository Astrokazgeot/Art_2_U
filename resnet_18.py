import os
os.listdir("C:/Users/cclchd-karman/Desktop/resnetprac")
import os
import tensorflow as tf

model_path = "C:/Users/cclchd-karman/Desktop/resnetprac/best_model.h5"

if os.path.exists(model_path):
    print(f"✅ Model found at: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print("✅ Model loaded successfully!")
else:
    print(f"❌ Model not found at: {model_path}")
