from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf
import numpy as np

def CBModel(X_train, y_train):
    model = Sequential()

    # Membuat model
    model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=tf.keras.regularizers.l1(0.01)))
    # model.add(Dropout(0.8))
    model.add(Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.01)))
    model.add(Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.01)))
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer=tf.optimizers.Adamax(learning_rate=0.001), loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=50, batch_size=2, validation_split=0.1)
    print(model.summary())
    model.save('gfgModel.h5')
    print('Model Saved!')