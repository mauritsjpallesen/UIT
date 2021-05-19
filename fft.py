import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fftpack import fft, fftfreq
import numpy as np

def wav_fft(fp):
    rate, data = wavfile.read(fp)

    avg_data = data.sum(axis=1) / 2

    N = avg_data.shape[0]
    s = N / float(rate)
    t_step = 1.0 / rate

    t = np.arange(0, s, t_step)

    fft_data = abs(fft(avg_data))
    fft_data_half = fft_data[:N//2]

    f = fftfreq(N, t_step)
    f_half = f[:N//2]

    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle(fp.split('/')[-1])
    ax1.plot(t, avg_data, 'r')
    ax2.plot(f_half, fft_data_half, 'g')
    plt.savefig('figures/' + fp.split('/')[-1].split('.')[0])
    plt.show()

    return t, avg_data, f_half, fft_data_half

t_10, data_10, f_10, fft_10 = wav_fft('audio/tube10.wav')
t_20, data_20, f_20, fft_20 = wav_fft('audio/tube20.wav')
