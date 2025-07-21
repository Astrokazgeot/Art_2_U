import tensorflow as tf
from tensorflow import keras
from keras.applications import ResNet50
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.applications.resnet50 import preprocess_input
from tensorflow.keras.optimizers import Adam

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


# Get number of classes dynamically
num_classes = len(train_ds.class_names)
print(f"Detected number of classes: {num_classes}")

# Data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.25),
    tf.keras.layers.RandomZoom(0.3),
    tf.keras.layers.RandomContrast(0.4),
    tf.keras.layers.RandomTranslation(0.15, 0.15),
])

# Preprocess + augment
def process_train(image, label):
    image = data_augmentation(image, training=True)
    image = preprocess_input(image)
    return image, label

def process_val(image, label):
    image = preprocess_input(image)
    return image, label

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(process_train).cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.map(process_val).cache().prefetch(buffer_size=AUTOTUNE)

# Load pretrained ResNet base
conv_base = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Fine-tuning only conv5_block1_out and later
conv_base.trainable = True
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'conv4_block1_out':
        set_trainable = True
    layer.trainable = set_trainable

# Build model with Functional API
inputs = Input(shape=(224, 224, 3))
x = conv_base(inputs, training=True)
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.6)(x)
outputs = Dense(num_classes, activation='softmax')(x)

model = keras.Model(inputs, outputs)

# Compile
optimizer = Adam(learning_rate=1e-4, clipnorm=1.0)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint(
    filepath="C:/Users/cclchd-karman/Desktop/resnetprac/best_model.h5",
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True, verbose=1)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)

# Summary
model.summary()

# Train
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    callbacks=[checkpoint, early_stop, lr_scheduler],
  
)

# Save final model manually
model.save("C:/Users/cclchd-karman/Desktop/resnetprac/best_model_manual.h5")
print("✅ Model manually saved at: C:/Users/cclchd-karman/Desktop/resnetprac/best_model_manual.h5")
