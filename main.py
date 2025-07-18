import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.applications import ResNet50
from keras.layers import Flatten,Dense



conv_base=ResNet50(
    weights='imagenet',# keep weights as it was trained on original imagenet dataset
    include_top=False, # means remove top dense layer
    input_shape=(224,224,3) # standard for resnet 
)
# Load both datasets
ds1 = tf.keras.utils.image_dataset_from_directory(

      directory='train/',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(224,224)
)

ds2 = tf.keras.utils.image_dataset_from_directory(
    "valid/",
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(224,224)
)

# Combine them
combined_ds = ds1.concatenate(ds2)

model=Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dense(25,activation='softmax'))  #output layer

data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.1),
    keras.layers.RandomZoom(0.1),
])

conv_base.trainable=True
set_trainable=False
for layer in conv_base.layers:
    if layer.name=='conv4_block1_out':
        set_trainable=True
    if set_trainable:
        layer.trainable=True
    else:
        layer.trainable=False

test_ds=keras.utils.image_dataset_from_directory(
      "test/",
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(224,224)
)

def process(image,label):
    image=tf.cast(image/255.0,tf.float32)
    return image,label


AUTOTUNE = tf.data.AUTOTUNE

combined_ds = combined_ds.map(process).map(lambda x, y: (data_augmentation(x, training=True), y))
combined_ds= combined_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
test_ds= test_ds.map(process).cache().prefetch(buffer_size=AUTOTUNE)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
from keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint("best_model.h5", monitor='val_accuracy', save_best_only=True)

model.fit(combined_ds, epochs=30, validation_data=test_ds, callbacks=[checkpoint])

