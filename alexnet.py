#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 15:38:20 2019

@author: mariana
"""

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
<<<<<<< HEAD
import matplotlib.pyplot as plt
=======
>>>>>>> 21a28cd751ec3d6295f18f58cae6e3e4c677f549

# (2) Get Data
gerador_train = ImageDataGenerator(rescale = 1./255,
                                   rotation_range = 7,
                                   horizontal_flip = True, 
                                   shear_range = 0.2,
                                   height_shift_range = 0.7,
                                   zoom_range = 0.2)
gerador_teste = ImageDataGenerator(rescale = 1./255)

<<<<<<< HEAD
base_train = gerador_train.flow_from_directory('/home/rodrigo/Documentos/MARIANA_TCC/TCC_Oficial/DATASET/train_set/',
=======
base_train = gerador_train.flow_from_directory('/home/mariana/Documents/TCC_Oficial/DATASET/train_set',
>>>>>>> 21a28cd751ec3d6295f18f58cae6e3e4c677f549
                                               target_size = (224,224),
                                               batch_size = 32,
                                               class_mode = 'categorical')

<<<<<<< HEAD
base_teste = gerador_teste.flow_from_directory('/home/rodrigo/Documentos/MARIANA_TCC/TCC_Oficial/DATASET/test_set/',
=======
base_teste = gerador_teste.flow_from_directory('/home/mariana/Documents/TCC_Oficial/DATASET/test_set',
>>>>>>> 21a28cd751ec3d6295f18f58cae6e3e4c677f549
                                               target_size = (224,224),
                                               batch_size = 32,
                                               class_mode = 'categorical')
# (3) Create a sequential model
model = Sequential()

# 1st Convolutional Layer
model.add(Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11),\
 strides=(4,4), padding='valid'))
model.add(Activation('relu'))
# Pooling 
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation before passing it to the next layer
model.add(BatchNormalization())

# 2nd Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation
model.add(BatchNormalization())

# 3rd Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Batch Normalisation
model.add(BatchNormalization())

# 4th Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Batch Normalisation
model.add(BatchNormalization())

# 5th Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation
model.add(BatchNormalization())

# Passing it to a dense layer
model.add(Flatten())
# 1st Dense Layer
model.add(Dense(4096, input_shape=(224*224*3,)))
model.add(Activation('relu'))
# Add Dropout to prevent overfitting
model.add(Dropout(0.4))
# Batch Normalisation
model.add(BatchNormalization())

# 2nd Dense Layer
model.add(Dense(4096))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))
# Batch Normalisation
model.add(BatchNormalization())

# 3rd Dense Layer
model.add(Dense(1000))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))
# Batch Normalisation
model.add(BatchNormalization())

# Output Layer
model.add(Dense(9))
model.add(Activation('softmax'))

model.summary()

# (4) Compile 
model.compile(loss='categorical_crossentropy', optimizer='adam',\
 metrics=['accuracy'])

# (5) Train
<<<<<<< HEAD
history = model.fit_generator(base_train, steps_per_epoch = 11217,
                            epochs = 10, validation_data = base_teste, 
                            validation_steps = 3740)


#dispõe grafico
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
=======
model.fit_generator(base_train, steps_per_epoch = 11217,
                            epochs = 10, validation_data = base_teste, 
                            validation_steps = 3740)
>>>>>>> 21a28cd751ec3d6295f18f58cae6e3e4c677f549
