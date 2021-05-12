import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pydub import AudioSegment
import crepe
import numpy as np
from scipy.io import wavfile

sr, lowWhistle = wavfile.read('./audio/whistle-low.wav')
time, frequency, confidence, activation = crepe.predict(lowWhistle, sr, viterbi=True)

zipped = zip(frequency, confidence)
filtered = [x[0] for x in zipped if x[1] > 0.75]

print("Average frequency for low-whistle: ", np.average(filtered))

sr, midWhistle = wavfile.read('./audio/whistle-mid.wav')
time, frequency, confidence, activation = crepe.predict(midWhistle, sr, viterbi=True)

zipped = zip(frequency, confidence)
filtered = [x[0] for x in zipped if x[1] > 0.75]

print("Average frequency for mid-whistle: ", np.average(filtered))

sr, highWhistle = wavfile.read('./audio/whistle-high.wav')
time, frequency, confidence, activation = crepe.predict(highWhistle, sr, viterbi=True)

zipped = zip(frequency, confidence)
filtered = [x[0] for x in zipped if x[1] > 0.75]

print("Average frequency for high-whistle: ", np.average(filtered))


# print("Time\tFrequency\tConfidence")
# for i in range(len(time)):
#     print("%.2f\t%.2f\t\t%.2f" % (time[i], frequency[i], confidence[i]))
