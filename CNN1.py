import tensorflow as tf
import keras as keras
import keras.models as M
import keras.layers as L
import keras.backend as K
from keras.callbacks import ModelCheckpoint
import keras.utils as utils
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
import visualkeras
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

n_classes = 3

height = 500
width = 500

training_datagen = ImageDataGenerator(rescale = 1./255)

train_datagen = training_datagen.flow_from_directory('Train',
                                                     target_size=(height, width),
                                                     batch_size=32,
                                                     class_mode='categorical',)

val_datagen = ImageDataGenerator(rescale=1. / 255)

validation_datagen = val_datagen.flow_from_directory('Val',
                                                     target_size=(height, width),
                                                     batch_size=32,
                                                     class_mode='categorical',)

K.clear_session()

input = L.Input(shape=(height, width, 3))
x = L.Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding='same')(input)
x = L.BatchNormalization()(x)
x = L.Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same')(x)
x = L.BatchNormalization()(x)
x = L.MaxPool2D(pool_size=(2, 2))(x)
x = L.Dropout(rate=0.3)(x)
x = L.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
x = L.BatchNormalization()(x)
x = L.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
x = L.BatchNormalization()(x)
x = L.MaxPool2D(pool_size=(2, 2))(x)
x = L.Dropout(rate=0.3)(x)
x = L.Flatten()(x)
x = L.Dense(units=256, activation='relu')(x)
x = L.BatchNormalization()(x)
x = L.Dropout(rate=0.5)(x)
output = L.Dense(units=n_classes, activation='sigmoid')(x)

model = M.Model(inputs=input, outputs=output, name = 'CNN1')

model.summary()
visualkeras.layered_view(model, to_file='CNN1.png')

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

checkpoint_callback = ModelCheckpoint(filepath = os.path.join('C:/Users/mdpan/Desktop/AI Challenge', 'cnn1.0({val_accuracy:.2f}).h5'), 
                             monitor = 'val_accuracy', 
                             save_best_only = True,
                             mode = 'max')

history  = model.fit(train_datagen,
                    validation_data = validation_datagen,
                    epochs = 10,
                    batch_size = 32,
                    callbacks = [checkpoint_callback])

plt.style.use(['dark_background'])
fig, ax = plt.subplots(figsize=(10, 5))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.style.use(['dark_background'])
fig, ax = plt.subplots(figsize=(10, 5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()