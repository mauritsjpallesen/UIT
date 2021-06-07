import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import sys
# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
#Keras
import keras
from keras import models
from keras import layers

def getTrainAndTestData(testSize):
    data = pd.read_csv('data.csv')
    data = data.drop(['filename'],axis=1)
    genre_list = data.iloc[:, -1]
    encoder = LabelEncoder()
    y = encoder.fit_transform(genre_list)

    scaler = StandardScaler()
    X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize)
    return X_train, X_test, y_train, y_test, encoder, scaler

def getTrainedModel(X_train, y_train):
    model = models.Sequential()
    model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20, batch_size=128)
    return model

if __name__ == '__main__':
    X_train, X_test, y_train, y_test, encoder, scaler = getTrainAndTestData(0.1)
    model = getTrainedModel(X_train, y_train)
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print('test_acc: ', test_acc)
    predictions = model.predict(X_test)
    print(y_test)
    print(np.argmax(predictions, axis=1))
    sys.exit()
