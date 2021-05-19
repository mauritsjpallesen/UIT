import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fftpack import fft, fftfreq
import numpy as np

def wav_fft(fp):
    print(f"Sound: {fp.split('/')[-1]}")
    rate, data = wavfile.read(fp)

    avg_data = data.sum(axis=1) // 2
    sidx, eidx, cropped_data = crop_sound(avg_data, 1000, 5)
    print(f"Analyzing from {sidx} to {eidx}")

    N = cropped_data.shape[0]
    s = N / float(rate)
    t_step = 1.0 / rate

    t = np.arange(0, s, t_step)

    fft_data = abs(fft(cropped_data))
    fft_data_half = fft_data[:N//2]

    f = fftfreq(N, t_step)
    f_half = f[:N//2]

    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle(fp.split('/')[-1])
    ax1.plot(t, cropped_data, 'r')
    ax2.plot(f_half, fft_data_half, 'g')
    plt.savefig('figures/' + fp.split('/')[-1].split('.')[0])
    plt.show()

    f3 = argmax_n(fft_data_half, 3)

    print(f"Most contributing frequencies: {f_half[f3].astype('int')}")
    print()

def argmax_n(x, n):
    return np.flip(np.argsort(x)[-n:])

def crop_sound(data, threshold, n):
    start_idx = 0
    end_ind = len(data)-1

    # start index
    for i in range(len(data)):
        if abs(data[i]) > threshold and np.all(abs(data[i:i+n]) > threshold):
            start_idx = i
            break

    # end index
    for i in range(start_idx, len(data)):
        if abs(data[i]) < threshold and np.all(abs(data[i:i+n]) < threshold):
            end_idx = i
            break

    return start_idx, end_idx, data[start_idx:end_idx]

if __name__ == '__main__':
    wav_fft('audio/tube10.wav')
    wav_fft('audio/tube20.wav')
