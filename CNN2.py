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
import PD

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

x = L.Conv2D(32, (5, 5), activation='relu',padding='valid')(input)
x = L.Conv2D(32, (1, 1), activation='relu')(x)
x = L.Conv2D(32, (1, 1), activation='relu')(x)
x = L.MaxPooling2D((2,2))(x)
x = L.Dropout(0.5)(x)

x = L.Conv2D(64, (3, 3), activation='relu',padding='valid')(x)
x = L.Conv2D(64, (1, 1), activation='relu')(x)
x = L.Conv2D(64, (1, 1), activation='relu')(x)
x = L.MaxPooling2D((2,2))(x)
x = L.Dropout(0.5)(x)

x = L.Conv2D(128, (3, 3), activation='relu',padding='valid')(x)
x = L.Conv2D(32, (1, 1), activation='relu')(x)
x = L.Conv2D(n_classes, (1, 1))(x)

x = L.GlobalAveragePooling2D()(x)
x = L.Activation(activation='softmax')(x)

model = M.Model(input, x, name='CNN2')


model.summary()
visualkeras.layered_view(model, to_file='CNN2.png')

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

checkpoint_callback = ModelCheckpoint(filepath = os.path.join('C:/Users/mdpan/Desktop/AI Challenge', 'cnn2.0({val_accuracy:.2f}).h5'), 
                             monitor = 'val_accuracy', 
                             save_best_only = True,
                             mode = 'max')

history  = model.fit(train_datagen,
                    validation_data = validation_datagen,
                    epochs = 10,
                    batch_size = 16,
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