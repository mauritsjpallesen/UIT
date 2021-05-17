import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pydub import AudioSegment
import crepe
import numpy as np
from scipy.io import wavfile

def getPrediction(filePath):
	sr, midWhistle = wavfile.read(filePath)
	time, frequency, confidence, activation = crepe.predict(midWhistle, sr, viterbi=True)

	return {
		"time": time,
		"frequency": frequency,
		"confidence": confidence
	}

def getAverageFrequncy(prediction, confidenceThreshold):
	zipped = zip(prediction["frequency"], prediction["confidence"])
	filtered = [x[0] for x in zipped if x[1] > confidenceThreshold]
	return np.average(filtered)
