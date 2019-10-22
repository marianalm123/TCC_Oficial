#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 15:38:20 2019

@author: mariana
"""
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.wrappers.scikit_learn import KerasClassifier

# pesquisa para descobrir quais os melhores parametros
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

def criarRede(optimizer, loos, kernel_initializer, activation, neurons, batch_size):
    classificador = Sequential()        
    
    classificador = Sequential()
    classificador.add(Conv2D(32,(3,3),input_shape = (64,64,3), activation = activation))
    classificador.add(BatchNormalization())
    classificador.add(MaxPooling2D(pool_size = (2,2)))
    
    classificador.add(Conv2D(32,(3,3),input_shape = (64,64,3), activation = activation))
    classificador.add(BatchNormalization())
    classificador.add(MaxPooling2D(pool_size = (2,2)))
    
    #classificador.add(Conv2D(32,(3,3),input_shape = (64,64,3), activation = 'relu'))
    #classificador.add(BatchNormalization())
    #classificador.add(MaxPooling2D(pool_size = (2,2)))
        
    classificador.add(Flatten())
    
    classificador.add(Dense(units = neurons, activation = activation,
                             kernel_initializer = kernel_initializer))
    classificador.add(Dropout(0.2))
    
    classificador.add(Dense(units = neurons, activation = activation,
                             kernel_initializer = kernel_initializer))
    classificador.add(Dropout(0.2))
    
    classificador.add(Dense(units = 9, activation = 'softmax'))
    
    classificador.summary()
    
    classificador.compile(optimizer = optimizer, loss = loss,
                      metrics = ['categorical_accuracy'])
    

    return classificador

classificador = KerasClassifier(build_fn = criarRede, batch_size = 32, epochs = 10)
parametros = {'batch_size':[32,64],
              'epochs':[10,50,100],
              'optimizer': ['adam', 'sgd', 'adagrad'],
              'loos':['categorical_crossentropy'],
              'kernel_initializer':['random_uniform','normal'],
              'activation':['relu', 'tahn'],
              'neurons':[128,64]}
    
gerador_train = ImageDataGenerator(rescale = 1./255,
                                   rotation_range = 7,
                                   horizontal_flip = True,
                                   shear_range = 0.2,
                                   height_shift_range = 0.7,
                                   zoom_range = 0.2)
gerador_teste = ImageDataGenerator(rescale = 1./255)

base_train = gerador_train.flow_from_directory('/home/mariana/Documents/TCC_Oficial/DATASET/train_set/',
                                               target_size = (64,64),
                                               batch_size = 32,
                                               class_mode = 'categorical')

base_teste = gerador_teste.flow_from_directory('//home/mariana/Documents/TCC_Oficial/DATASET/test_set/',
                                               target_size = (64,64),
                                               batch_size = 32,
                                               class_mode = 'categorical')

#history = classificador.fit_generator(base_train, steps_per_epoch = 11217,
#                                          epochs = 10, validation_data = base_teste,
#                                          validation_steps = 3740)



grid_search = GridSearchCV(estimator = classificador,
                           param_grid = parametros,
                           scoring = 'accuracy',
                           cv = 5)
grid_search = grid_search.fit(base_train, base_train)
melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_

print 'Melhores parametros' + str(melhores_parametros)
print 'Melhor precisao' + str(melhor_precisao)

#
##dispõe grafico
#plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
#plt.title('Model Accuracy')
#plt.ylabel('Accuracy')
#plt.xlabel('Epoch')
#plt.legend(['Train', 'Test'], loc='upper left')
#plt.show()
#
## Plot training & validation loss values
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('Model loss')
#plt.ylabel('Loss')
#plt.xlabel('Epoch')
#plt.legend(['Train', 'Test'], loc='upper left')
#plt.show()