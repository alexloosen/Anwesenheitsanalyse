import modelTuning as modelTuning
import prepareData as prepareData

from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import SpatialDropout1D

def createClassifier(classType, Xtrain, ytrain):
    if (classType == 'RFC'):
        modelClass = RandomForestClassifier()
    elif (classType == 'SVC'):
        modelClass = SVC()
    elif (classType == 'GBC'):
        modelClass = GradientBoostingClassifier()
    elif (classType == 'BC'):
        modelClass = BaggingClassifier()
    elif (classType == 'LR'):
        modelClass = LogisticRegression(max_iter=1000)
        Xtrain, scaler = prepareData.normalize_min_max(Xtrain)
        ytrain, scaler = prepareData.normalize_min_max(ytrain)
        
    modelClass.fit(Xtrain, ytrain)
    modelTuning.saveModel(modelClass, 'models\\' + classType + '.mod')

def createLSTM(Xtrain):
    model = Sequential()
    model.add(LSTM(units=1, return_sequences=True, input_shape = (Xtrain.shape[1],Xtrain.shape[2]), activation='relu'))
    model.add(Dropout(0.2))
    model.add(LSTM(units=1, return_sequences=True, input_shape = (Xtrain.shape[1],Xtrain.shape[2]), activation='relu'))
    model.add(Dropout(0.2))

    #opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics='accuracy')
    model.build()
    
    return model

def createNN(Xtrain):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(Xtrain.shape[1], input_dim=Xtrain.shape[1], activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(round(Xtrain.shape[1]/2), activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(round(Xtrain.shape[1]/3), activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))

    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
#    model.build()
    
    return model
