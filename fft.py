import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fftpack import fft, fftfreq
import copy
import numpy as np
import sys
from crop import show_crop, crop_sound


def wav_fft(fp, thres, n):
    print(f"Sound: {fp.split('/')[-1]}")
    rate, data = wavfile.read(fp)

    avg_data = data.sum(axis=1) // 2
    print(avg_data.shape)
    sidx, eidx, cropped_data = crop_sound(avg_data, thres, n)
    print(f"Analyzing from {sidx} to {eidx}")

    N = cropped_data.shape[0]
    s = N / rate
    t_step = 1.0 / rate

    t = np.arange(0, s, t_step)

    fft_data = abs(fft(cropped_data))
    fft_data_half = fft_data[:N//2]

    f = fftfreq(N, t_step)
    f_half = f[:N//2]

    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle(fp.split('/')[-1] + " fourier transform")
    ax1.plot(t, cropped_data, 'g')
    ax2.plot(f_half, fft_data_half, 'b')
    plt.savefig('figures/' + fp.split('/')[-1].split('.')[0])
    plt.show()

    f3 = argmax_n(fft_data_half, 3)

    print(f"Most contributing frequencies: {f_half[f3].astype('int')}")
    print()

def argmax_n(x, n):
    return np.flip(np.argsort(x)[-n:])

if __name__ == '__main__':
	if len(sys.argv) != 4:
		print("Usage: cropt.py <FILE> <THRESHOLD> <N>")
		sys.exit()
	wav_fft(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
