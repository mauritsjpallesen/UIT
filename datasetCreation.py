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

def createDataSet(seconds, dir):
    file = open('data.csv', 'w', newline='')

    header = 'filename zero_crossing_rate spectral_centroid spectral_rolloff spectral_bandwidth chroma_stft rms'
    for i in range(1, 21):
        header += f' mfcc{i}'

    header += ' label'
    header = header.split(' ')

    with file:
        writer = csv.writer(file)
        writer.writerow(header)

    # surfaces = 'smallBox bigBox metal mousePad woodenTable'.split()
    surfaces = 'backLeg frontLeg ear trunk upperBody'.split()
    for s in surfaces:
        if dir == 'mau':
            directory = f'./audio/elephant/mau/{s}/'
        elif dir == 'mac':
            directory = f'./audio/elephant/mac/{s}/'
        elif dir == 'all':
            directory = f'./audio/elephant/all/{s}/'
        else:
            directory = f'./audio/elephant/samson/{s}/'
        for filename in os.listdir(directory):
            filePath = directory + filename
            features = extractFeaturesFromFile(filePath, seconds)
            to_append = f'{filename}'
            for f in features:
                to_append += f' {f}'

            to_append += f' {s}'

            file = open('data.csv', 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())
