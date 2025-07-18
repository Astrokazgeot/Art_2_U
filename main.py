import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.applications import ResNet50
from keras.layers import Flatten, Dense, GlobalAveragePooling2D, Dropout
from keras.applications.resnet50 import preprocess_input
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Load ResNet base
conv_base = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Load datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    'train/',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(224, 224)
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    'valid/',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(224, 224)
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    'test/',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(224, 224)
)

# Model architecture
model = Sequential([
    conv_base,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.6),  # More dropout to fight overfitting
    Dense(21, activation='softmax')
])

# Data augmentation
data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.1),
    keras.layers.RandomZoom(0.1),
])

# Set trainable layers (fine-tune from conv5_block1_out)
conv_base.trainable = True
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'conv5_block1_out':
        set_trainable = True
    layer.trainable = set_trainable

# Preprocess and prepare data
def process(image, label):
    return preprocess_input(image), label

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.map(process).map(lambda x, y: (data_augmentation(x, training=True), y))
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

val_ds = val_ds.map(process).cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.map(process).cache().prefetch(buffer_size=AUTOTUNE)

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath="C:/Users/cclchd-karman/Desktop/resnetprac/best_model.h5",
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

early_stop = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True, verbose=1)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)

# Train
model.fit(
    train_ds,
    epochs=10,
    validation_data=val_ds,
    callbacks=[checkpoint, early_stop, lr_scheduler]
)
# Manually save the model at the end
model.save("C:/Users/cclchd-karman/Desktop/resnetprac/best_model_manual.h5")
print("✅ Model manually saved at: C:/Users/cclchd-karman/Desktop/resnetprac/best_model_manual.h5")
