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

def getTrainAndTestData(train_size):
    data = pd.read_csv('data.csv')
    data = data.drop(['filename'],axis=1)
    genre_list = data.iloc[:, -1]
    encoder = LabelEncoder()
    y = encoder.fit_transform(genre_list)

    scaler = StandardScaler()
    X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))
    X_train, X_test, Y_train, Y_test = [], [], [], []
    for i in range(5):
        x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=train_size)
        X_train.append(x_train)
        X_test.append(x_test)
        Y_train.append(y_train)
        Y_test.append(y_test)

    return X_train, X_test, Y_train, Y_test, encoder, scaler, X

def getTrainedModel():
    X_train, X_test, y_train, y_test, encoder, scaler, X = getTrainAndTestData(0.5)
    clfs = []
    for i in range(5):
        clf = svm.SVC(kernel='linear', probability=True)
        clf.fit(X_train[i], y_train[i])
        clfs.append(clf)

    return clfs, encoder, scaler, X

def predict(clfs, modelScaler, modelEncoder, filePath):
    features = extractFeaturesFromFile(filePath, 0.25)
    if (features == None):
        return 'None'

    normalizedFeatures = modelScaler.transform(np.array(features, dtype = float).reshape(1, -1))
    prediction = 0
    for j in range(5):
        prediction += clfs[j].predict_proba(normalizedFeatures)
    prediction = (prediction/5)[0]
    encodedLabel = np.argmax(prediction)
    predictionProbability = prediction[encodedLabel]
    print('Prediction probability: ', predictionProbability)
    if (predictionProbability < 0.45):
        return 'None'

    label = modelEncoder.inverse_transform([encodedLabel])[0]
    return label

if __name__ == '__main__':

    if ('-c' in sys.argv):
        if (len(sys.argv) != 4):
            print("Usage: ml.py -c <SECONDS_TO_CROP_FROM_PEAK>")
            sys.exit()

        createDataSet(float(sys.argv[2]), sys.argv[3])
        # sys.exit()

    #median
    # y_pred = np.zeros((5,50,5))
    # for j in range(5):
    #     y_pred[j,:] = clfs[j].predict_proba(X)
    # y_pred = np.median(y_pred, axis=0)
    clfs, encoder, scaler, X = getTrainedModel()
    #mean
    y_pred = 0
    for j in range(5):
        y_pred += clfs[j].predict_proba(X)
    y_pred = y_pred/5
        # for z in range(len(y_pred)):
        #     actual = np.argmax(y_pred[z])
        #     expected = y_test[z]
        #     results.append(actual == expected)

    # print(np.argmax(y_pred, axis=1))
    # print(y_test)
    #a = np.corrcoef(clf.coef_.T, clf.coef_.T)
    # print(float(sum(results))/float(len(y_test)))
    if "all" in sys.argv[3]:
        plt.imshow(y_pred, aspect=5/150)
    else:
        plt.imshow(y_pred, aspect=5/50)
    plt.xticks(ticks = list(range(5)), labels=encoder.classes_)
    plt.colorbar()
    if ('-c' in sys.argv):
        plt.title(sys.argv[3] + "\nOverall mean certainty: " + str(np.mean(np.max(y_pred, axis=0))))
        plt.savefig(sys.argv[3] + "_elephant.pdf")
    else:
        plt.title(sys.argv[1] + "\nOverall mean certainty: " + str(np.mean(np.max(y_pred, axis=0))))
        plt.savefig(sys.argv[1] + "_elephant.pdf")
    plt.show()

    # sys.exit()
