import tensorflow as tf
from keras.applications.resnet50 import preprocess_input

# Load the saved model
model = tf.keras.models.load_model("C:/Users/cclchd-karman/Desktop/resnetprac/best_model_manual.h5")
print("✅ Model loaded successfully.")

# Load test dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    'test/',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(224, 224)
)

# Preprocess test data
AUTOTUNE = tf.data.AUTOTUNE
def process(image, label):
    return preprocess_input(image), label

test_ds = test_ds.map(process).cache().prefetch(buffer_size=AUTOTUNE)

# Evaluate the model
loss, accuracy = model.evaluate(test_ds)
print(f"✅ Test Loss: {loss:.4f}")
print(f"✅ Test Accuracy: {accuracy:.4f}")
