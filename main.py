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

model=Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dense(26,activation='softmax'))  #output layer

conv_base.trainable=True
set_trainable=False
for layer in conv_base.layers:
    if layer.name=='conv5_block1_out':
        set_trainable=True
    if set_trainable:
        layer.trainable=True
    else:
        layer.trainable=False

train_ds=keras.utils.image_dataset_from_directory(
    directory='train/',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(224,224)
)

test_ds=keras.utils.image_dataset_from_directory(
    directory='test/',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(224,224)
)

def process(image,label):
    image=tf.cast(image/255.0,tf.float32)
    return image,label
train_ds=train_ds.map(process)
test_ds=test_ds.map(process)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(train_ds,epochs=10,validation_data=test_ds)

