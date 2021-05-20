import matplotlib.pyplot as plt
from scipy.io import wavfile
import copy
import numpy as np
import sys
import os

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

def plot_all_audios_in_folder(folderPath, threshold, n):
    files = os.listdir(folderPath)
    plt.figure()
    for i in range(len(files)):
        rate, data = wavfile.read(folderPath + files[i])
        avg_data = data.sum(axis=1) // 2 if wavDataIsStereo(data) else data

        N = avg_data.shape[0]
        s = N / rate
        t_step = 1.0 / rate

        t = np.arange(0, s, t_step)

        if (t.shape != avg_data.shape):
            continue

        plt.plot(t, avg_data)

    plt.show()

def wavDataIsStereo(data):
    return data.ndim == 2

if __name__ == '__main__':
    if (len(sys.argv) != 4):
        print("Usage: cropt.py <FILE> <THRESHOLD> <N>")
        print("   or: cropt.py <DIRECTORY> <THRESHOLD> <N>")
        sys.exit()

    if (os.path.isfile(sys.argv[1])):
        show_crop(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
    elif (os.path.isdir(sys.argv[1])):
        plot_all_audios_in_folder(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
    else:
        print(sys.argv[1] + " must be a file or a directory")
