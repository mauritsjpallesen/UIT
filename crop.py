import matplotlib.pyplot as plt
from scipy.io import wavfile
import copy
import numpy as np
import sys
import os
import librosa
import librosa.display

def show_crop(fp, thres, n):
    rate, data = wavfile.read(fp)

    avg_data = data.sum(axis=1) // 2 if wavDataIsStereo(data) else data
    print(avg_data)
    print(avg_data.shape)
    sidx, eidx, cropped_data = crop_sound(avg_data, thres, n)

    N = avg_data.shape[0]
    s = N / rate
    t_step = 1.0 / rate

    t = np.arange(0, s, t_step)

    cropped_data = copy.deepcopy(avg_data)
    cropped_data[:sidx] = [0] * sidx
    cropped_data[eidx:] = [0] * (len(cropped_data)-eidx)

    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle(fp.split('/')[-1] + " trimming")
    ax1.plot(t, avg_data, 'g')
    ax2.plot(t, cropped_data, 'b')
    ax2.axvline(x = t[sidx-1], color='r')
    ax2.axvline(x = t[eidx], color='r')
    plt.show()

def crop_sound(data, threshold, n):
	start_idx = 0
	end_ind = len(data)-1

	# start index (inclusive)
	for i in range(len(data)):
		if abs(data[i]) > threshold and np.all(abs(data[i:i+n]) > threshold):
			start_idx = i
			break

	# end index (not inclusive)
	for i in range(start_idx, len(data)):
		if abs(data[i]) < threshold and np.all(abs(data[i:i+n]) < threshold):
			end_idx = i
			break

	return start_idx, end_idx, data[start_idx:end_idx]

def crop_seconds_from_threshold(data, rate, threshold, seconds):
    start_index = -1
    for i in range(len(data)):
        if (abs(data[i]) > threshold):
            start_index = i
            break

    end_index = start_index + rate * seconds
    if (end_index >= len(data)):
        end_index = len(data) - 1

    return data[start_index:int(end_index)]

def plot_all_audios_in_folder(folderPath, threshold, seconds):
    files = os.listdir(folderPath)
    plt.figure()
    for i in range(len(files)):
        data, rate = librosa.load(folderPath + files[i], sr=44100)
        cropped = crop_seconds_from_threshold(data, rate, threshold, seconds)

        librosa.display.waveplot(cropped, sr=rate, alpha=0.25)

    plt.show()

def plot_all_audios_in_folders(directoryPaths, threshold, seconds):

    fig, axs = plt.subplots(2, 2)
    fig.suptitle('Trigger on various surfaces')
    # y label: normalized amplitude

    for ax in axs.flat:
        ax.set(ylabel='Normalized amplitude')

    for row in range(len(axs)):
        for col in range(len(axs[row])):
            dir = directoryPaths.pop()
            files = os.listdir(dir)
            for f in files:
                data, rate = librosa.load(dir + f, sr=44100)
                cropped = crop_seconds_from_threshold(data, rate, threshold, seconds)
                axs[row, col].set_title(camelCaseToPlotTitle(dir))
                librosa.display.waveplot(cropped, sr=rate, alpha=0.5, ax=axs[row, col])

    for ax in axs.flat:
        ax.label_outer()

    fig.tight_layout()

    plt.show()

def camelCaseToPlotTitle(folderPath):
    folderName = folderPath.split('/')[-2]
    return ''.join(map(lambda x: x if x.islower() else " "+x, folderName)).lower().capitalize()


def wavDataIsStereo(data):
    return data.ndim == 2

if __name__ == '__main__':

    if (sys.argv[1] == '-plot'):
        plot_all_audios_in_folders(['./audio/bigBox/', './audio/smallBox/', './audio/mousePad/', './audio/woodenTable/'], float(sys.argv[2]), float(sys.argv[3]))
        sys.exit();

    if (len(sys.argv) < 3):
        print("Usage: cropt.py <FILE> <THRESHOLD> <N>")
        print("   or: cropt.py <DIRECTORY> <THRESHOLD> <N>")
        sys.exit()

    if (os.path.isfile(sys.argv[1])):
        show_crop(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
    elif (os.path.isdir(sys.argv[1])):
        plot_all_audios_in_folder(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]))
    else:
        print(sys.argv[1] + " must be a file or a directory")
