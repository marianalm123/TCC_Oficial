#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 15:38:20 2019

@author: mariana
"""
<<<<<<< HEAD

=======
    
>>>>>>> 21a28cd751ec3d6295f18f58cae6e3e4c677f549
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
<<<<<<< HEAD
import matplotlib.pyplot as plt

=======
>>>>>>> 21a28cd751ec3d6295f18f58cae6e3e4c677f549

classificador = Sequential()
classificador.add(Conv2D(32,(3,3),input_shape = (64,64,3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))

classificador.add(Conv2D(32,(3,3),input_shape = (64,64,3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))

#classificador.add(Conv2D(32,(3,3),input_shape = (64,64,3), activation = 'relu'))
#classificador.add(BatchNormalization())
#classificador.add(MaxPooling2D(pool_size = (2,2)))

classificador.add(Flatten())

classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))

classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))

classificador.add(Dense(units = 9, activation = 'softmax'))

<<<<<<< HEAD
classificador.summary()

=======
>>>>>>> 21a28cd751ec3d6295f18f58cae6e3e4c677f549
classificador.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
                      metrics = ['categorical_accuracy'])

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
                                               target_size = (64,64),
                                               batch_size = 32,
                                               class_mode = 'categorical')

<<<<<<< HEAD
base_teste = gerador_teste.flow_from_directory('/home/rodrigo/Documentos/MARIANA_TCC/TCC_Oficial/DATASET/test_set/',
=======
base_teste = gerador_teste.flow_from_directory('/home/mariana/Documents/TCC_Oficial/DATASET/test_set',
>>>>>>> 21a28cd751ec3d6295f18f58cae6e3e4c677f549
                                               target_size = (64,64),
                                               batch_size = 32,
                                               class_mode = 'categorical')

<<<<<<< HEAD
history = classificador.fit_generator(base_train, steps_per_epoch = 11217,
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
classificador.fit_generator(base_train, steps_per_epoch = 11217,
                            epochs = 10, validation_data = base_teste, 
                            validation_steps = 3740)








>>>>>>> 21a28cd751ec3d6295f18f58cae6e3e4c677f549




