import modelTuning as modelTuning
import prepareData as prepareData

from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import SpatialDropout1D
from keras import optimizers

import numpy as np

def createClassifier(classType, Xtrain, ytrain):
    if (classType == 'RFC'):
        modelClass = RandomForestClassifier()
    elif (classType == 'SVC'):
        modelClass = SVC()
    elif (classType == 'GBC'):
        modelClass = GradientBoostingClassifier()
    elif (classType == 'KNN'):
        modelClass = KNeighborsClassifier()
    elif (classType == 'LR'):
        modelClass = LogisticRegression(solver='saga', max_iter=5000)
        #Xtrain, scaler = prepareData.normalize_min_max(Xtrain)
        #ytrain, scaler = prepareData.normalize_min_max(ytrain)
    
    modelClass.fit(Xtrain, ytrain)
    modelTuning.saveModel(modelClass, 'models\\' + classType + '.mod')

def createLSTM(Xtrain):
    model = Sequential()
    model.add(LSTM(units=5, return_sequences=True, input_shape = (Xtrain.shape[1],Xtrain.shape[2]), activation='sigmoid'))
    model.add(Dropout(0.2))
#    model.add(LSTM(units=1, return_sequences=True, input_shape = (Xtrain.shape[1],Xtrain.shape[2]), activation='relu'))
#    model.add(Dropout(0.2))
    model.add(Dense(1, activation='relu'))

    opt = tf.keras.optimizers.Adam(learning_rate=1e-3, decay=1e-5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics='accuracy')
    #model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics='accuracy')
    #sgd = tf.keras.optimizers.SGD(learning_rate=0.01, decay=1e-4, momentum=0.7, nesterov=True)
    #model.compile(loss='mean_squared_error', optimizer=sgd, metrics='accuracy')
    model.build()
    
    return model

def createNN(Xtrain):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(Xtrain.shape[1], input_dim=Xtrain.shape[1], activation='relu'))#, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    #model.add(tf.keras.layers.Dense(round(Xtrain.shape[1]/2), activation='relu'))#, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(round(Xtrain.shape[1]/3), activation='relu'))#, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(1, activation='relu'))

    opt = tf.keras.optimizers.Adam(learning_rate=1e-5)#, decay=1e-5)
    model.compile(optimizer=opt,
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
#    model.build()
    
    return model
