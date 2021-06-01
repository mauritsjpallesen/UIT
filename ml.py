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
    #Create a svm Classifier
    clf = svm.SVC(kernel='linear') # Linear Kernel

    #Train the model using the training sets
    clf.fit(X_train, y_train)

    return clf

if __name__ == '__main__':

    if ('-c' in sys.argv):
        if (len(sys.argv) != 4):
            print("Usage: ml.py -c <SECONDS_TO_CROP_FROM_PEAK>")
            sys.exit()
        # current values:
        #    seconds   : 0.25
        createDataSet(float(sys.argv[2]), float(sys.argv[3]))
        sys.exit()

    X_train, X_test, y_train, y_test, encoder, scaler = getTrainAndTestData(0.1)
    clf = getTrainedModel(X_train, y_train)
    y_pred = clf.predict(X_test)
    results = []
    for i in range(len(y_pred)):
        actual = y_pred[i]
        expected = y_test[i]
        results.append(actual == expected)
    print(y_pred)
    print(y_test)
    print(float(sum(results))/float(len(y_test)))
    sys.exit()



# Uncomment and place the below code in the extractFeaturesFromFile function to see the features visualized
# Computing the time variable for visualization
# frames = range(len(spectral_centroid))
# t = librosa.frames_to_time(frames)
# def normalize(cropped, axis=0):
#     return sklearn.preprocessing.minmax_scale(cropped, axis=axis)
#
# # Plot audio waveform
# librosa.display.waveplot(cropped, sr=sampleRate, alpha=0.4)
#
# # Plot the Spectral Rolloff
# plt.plot(t, normalize(spectral_rolloff), color='r')
#
# # Plot the Spectral Centroid
# plt.plot(t, normalize(spectral_centroid), color='g')
#
# plt.show()
#
#
# # display MFCC — Mel-Frequency Cepstral Coefficients
# librosa.display.specshow(mfccs, sr=sampleRate, x_axis='time')
# plt.show()
