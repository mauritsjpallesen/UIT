import sys
import crop
import librosa
import librosa.display
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import pandas as pd
from sklearn import svm
from featureExtraction import extractFeaturesFromFile
from datasetCreation import createDataSet

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
    clf = svm.SVC(kernel='linear', probability=True)
    clf.fit(X_train, y_train)
    return clf

def predict(model, modelScaler, modelEncoder, filePath):
    features = extractFeaturesFromFile(filePath, 0.25)
    if (features == None):
        return None

    normalizedFeatures = modelScaler.transform(np.array(features, dtype = float).reshape(1, -1))
    prediction = model.predict_proba(normalizedFeatures)[0]
    encodedLabel = np.argmax(prediction)
    predictionProbability = prediction[encodedLabel]
    print('Prediction probability: ', predictionProbability)
    if (predictionProbability < 0.45):
        return None

    label = modelEncoder.inverse_transform([encodedLabel])[0]
    return label

if __name__ == '__main__':

    if ('-c' in sys.argv):
        if (len(sys.argv) != 3):
            print("Usage: ml.py -c <SECONDS_TO_CROP_FROM_PEAK>")
            sys.exit()

        createDataSet(float(sys.argv[2]))
        sys.exit()

    X_train, X_test, y_train, y_test, encoder, scaler = getTrainAndTestData(0.1)
    clf = getTrainedModel(X_train, y_train)
    y_pred = clf.predict_proba(X_test)
    results = []
    for i in range(len(y_pred)):
        actual = np.argmax(y_pred[i])
        expected = y_test[i]
        results.append(actual == expected)
    print(np.argmax(y_pred, axis=1))
    print(y_test)
    print(float(sum(results))/float(len(y_test)))
    sys.exit()
