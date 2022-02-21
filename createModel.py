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
    model.add(tf.keras.layers.Dense(X_presence.shape[1], input_dim=X_presence.shape[1], activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(round(X_presence.shape[1]/2), activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(round(X_presence.shape[1]/3), activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))

    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    model.build()
    
    return model
