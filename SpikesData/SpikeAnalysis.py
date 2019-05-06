from scipy.io import loadmat
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


if __name__ == "__main__":
    data = loadmat("Data.mat")
    data = data['He'][0][:1920000]
    t = [i/32000 for i in range(32000*60)]
    plt.figure('pure signal')
    plt.plot(t, data, label='pure signal')
    plt.legend()

    # Filter the data, and plot both the original and filtered signals.
    y = butter_lowpass_filter(data, 300, 32000)
    plt.figure('filtered data(low pass filter)')
    plt.plot(t, y, label='filtered signal (low pass)')
    plt.legend()

    y = butter_highpass_filter(data, 300, 32000)
    plt.figure('filtered data(high pass filter)')
    plt.plot(t, y, label='filtered signal (high pass)')
    plt.legend()

    plt.show()
