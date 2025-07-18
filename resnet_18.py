import tensorflow as tf
from keras.applications.resnet50 import preprocess_input
import numpy as np


# Load the trained model
model = tf.keras.models.load_model("best_model.h5")

# Load test dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    "test/",
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(224, 224),
    shuffle=False  # Important to align predictions with true labels
)

# Preprocess test dataset
def process(image, label):
    return preprocess_input(image), label

test_ds = test_ds.map(process).cache().prefetch(tf.data.AUTOTUNE)

# Evaluate accuracy
loss, acc = model.evaluate(test_ds)
print(f"Test Accuracy: {acc:.4f}, Loss: {loss:.4f}")

# Predictions and metrics
y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0)
y_pred_logits = model.predict(test_ds)
y_pred = np.argmax(y_pred_logits, axis=1)

