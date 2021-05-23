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

def extractFeaturesFromFile(audioFile, threshold, seconds):
    data, sampleRate = librosa.load(audioFile, sr=44100)
    cropped = crop.crop_seconds_from_threshold(data, sampleRate, threshold, seconds)
    zero_crossings = librosa.zero_crossings(cropped, pad=False)
    spectral_centroid = librosa.feature.spectral_centroid(cropped, sr=sampleRate)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(cropped, sr=sampleRate)[0]
    mfccs = librosa.feature.mfcc(cropped, sr=sampleRate)

    mfccsMeans = [np.mean(x) for x in mfccs]

    return np.mean(zero_crossings), np.mean(spectral_centroid), np.mean(spectral_rolloff), mfccsMeans

def createDataSet(threshold, seconds):
    file = open('data.csv', 'w', newline='')

    header = 'filename zero_crossing_rate spectral_centroid spectral_rolloff'
    for i in range(1, 21):
        header += f' mfcc{i}'

    header += ' label'
    header = header.split(' ')

    with file:
        writer = csv.writer(file)
        writer.writerow(header)

    surfaces = 'smallBox bigBox metal mousePad woodenTable'.split()
    for s in surfaces:
        directory = f'./audio/{s}/'
        for filename in os.listdir(directory):
            filePath = directory + filename
            zero_crossings, spectral_centroid, spectral_rolloff, mfccsMeans = extractFeaturesFromFile(filePath, threshold, seconds)

            to_append = f'{filename} {zero_crossings} {spectral_centroid} {spectral_rolloff}'
            for e in mfccsMeans:
                to_append += f' {e}'

            to_append += f' {s}'

            file = open('data.csv', 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())

def getTrainAndTestData(testSize):
    data = pd.read_csv('data.csv')
    data = data.drop(['filename'],axis=1)
    genre_list = data.iloc[:, -1]
    encoder = LabelEncoder()
    y = encoder.fit_transform(genre_list)
    scaler = StandardScaler()
    X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize)
    return X_train, X_test, y_train, y_test

def getTrainedSvmModel(X_train, y_train):
    #Create a svm Classifier
    clf = svm.SVC(kernel='linear') # Linear Kernel

    #Train the model using the training sets
    clf.fit(X_train, y_train)

    return clf


if __name__ == '__main__':

    # createDataSet(0.4, 0.25)
    X_train, X_test, y_train, y_test = getTrainAndTestData(0.4)
    clf = getTrainedSvmModel(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(y_pred)
    print(y_test)
    # if (len(sys.argv) != 3):
    #     print("Usage: ml.py <FOLDER1> <FOLDER2>")
    #     sys.exit()



# Uncomment ant place the below code in the extractFeaturesFromFile function to see the features visualized
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
