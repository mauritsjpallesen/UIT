import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pydub import AudioSegment
import crepe
import numpy as np
from scipy.io import wavfile

def getAverageFrequency(filePath, confidenceThreshold):
    sr, midWhistle = wavfile.read(filePath)
    time, frequency, confidence, activation = crepe.predict(midWhistle, sr, viterbi=True)
    zipped = zip(frequency, confidence)
    filtered = [z[0] for z in zipped if z[1] > confidenceThreshold]
    return np.average(filtered)
