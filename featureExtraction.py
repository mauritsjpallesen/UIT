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
    cropped = crop.crop_seconds_from_peak(data, sampleRate, seconds)
    if (len(cropped) == 0):
        return None

    zero_crossings = librosa.zero_crossings(cropped, pad=False)
    spectral_centroid = librosa.feature.spectral_centroid(cropped, sr=sampleRate)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(cropped, sr=sampleRate)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(cropped, sr=sampleRate)[0]
    rms = librosa.feature.rms(cropped)
    chroma_stft = librosa.feature.chroma_stft(cropped, sr=sampleRate)
    mfccs = librosa.feature.mfcc(cropped, sr=sampleRate)
    mfccsMeans = [np.mean(x) for x in mfccs]

    features = [np.mean(zero_crossings), np.mean(spectral_centroid), np.mean(spectral_rolloff), np.mean(spectral_bandwidth), np.mean(chroma_stft), np.mean(rms)]
    for m in mfccsMeans:
        features.append(m)

    return features
