import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pydub import AudioSegment
import crepe
import numpy as np
from scipy.io import wavfile

confidenceThreshold = 0.75

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

prediction1 = getPrediction('./audio/1.wav')
prediction2 = getPrediction('./audio/2.wav')
prediction3 = getPrediction('./audio/3.wav')
prediction4 = getPrediction('./audio/4.wav')


average1 = getAverageFrequncy(prediction1, confidenceThreshold)
average2 = getAverageFrequncy(prediction2, confidenceThreshold)
average3 = getAverageFrequncy(prediction3, confidenceThreshold)
average4 = getAverageFrequncy(prediction4, confidenceThreshold)

print(average1)
print(average2)
print(average3)
print(average4)

