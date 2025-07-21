import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.applications import ResNet50
from keras.layers import Flatten, Dense, GlobalAveragePooling2D, Dropout
from keras.applications.resnet50 import preprocess_input
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

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

# Check for NaNs in data
for images, labels in train_ds.take(1):
    print("Image batch shape:", images.shape)
    print("Any NaNs in images?", tf.math.reduce_any(tf.math.is_nan(images)).numpy())
    print("Any NaNs in labels?", tf.math.reduce_any(tf.math.is_nan(tf.cast(labels, tf.float32))).numpy())


# Data augmentation
data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.1),
    keras.layers.RandomZoom(0.1),
])

# Apply augmentation BEFORE preprocess_input
def process(image, label):
    image = data_augmentation(image, training=True)
    image = preprocess_input(image)
    return image, label

# Prefetch and shuffle
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(process).cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y)).cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.map(lambda x, y: (preprocess_input(x), y)).cache().prefetch(buffer_size=AUTOTUNE)

# Freeze all layers initially, then unfreeze from conv5_block1_out
conv_base.trainable = True
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'conv5_block1_out':
        set_trainable = True
    layer.trainable = set_trainable

# Model architecture
model = Sequential([
    conv_base,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.6),
    Dense(21, activation='softmax')  # 21 classes
])

# Use a safer learning rate with gradient clipping
optimizer = Adam(learning_rate=1e-4, clipnorm=1.0)

# Compile model
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
checkpoint = ModelCheckpoint(
    filepath="C:/Users/cclchd-karman/Desktop/resnetprac/best_model.h5",
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

early_stop = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True, verbose=1)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)

# Print model summary
model.summary()

# Train
model.fit(
    train_ds,
    epochs=10,
    validation_data=val_ds,
    callbacks=[checkpoint, early_stop, lr_scheduler]
)

# Save manually
model.save("C:/Users/cclchd-karman/Desktop/resnetprac/best_model_manual.h5")
print("✅ Model manually saved at: C:/Users/cclchd-karman/Desktop/resnetprac/best_model_manual.h5")
