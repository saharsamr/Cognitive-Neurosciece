from scipy.io import loadmat
from scipy.signal import butter, lfilter, find_peaks
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


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
    data_one_min = data['He'][0][:1920000]
    t = [i/32000 for i in range(32000*60)]
    plt.figure('pure signal')
    plt.plot(t, data_one_min, label='pure signal')
    plt.legend()

    # Filter the data, and plot both the original and filtered signals.
    y = butter_lowpass_filter(data_one_min, 300, 32000)
    plt.figure('filtered data(low pass filter)')
    plt.plot(t, y, label='filtered signal (low pass)')
    plt.legend()

    spike_activities = butter_highpass_filter(data_one_min, 300, 32000)
    plt.figure('filtered data(high pass filter)')
    plt.plot(t, spike_activities, label='filtered signal (high pass)')
    plt.legend()

    plt.figure('spikes in 5 minutes')
    data_five_min = data['He'][0][:5*1920000]
    axes = plt.gca()
    axes.set_ylim([-0.0001, 0.0003])
    spike_peaks = find_peaks(spike_activities, threshold=0.000008)[0]
    spikes = np.zeros((len(spike_peaks), 31))
    for i, peak in enumerate(spike_peaks):
        spikes[i] = spike_activities[peak-11:peak+20]
        plt.plot(spikes[i]-min(spikes[i]))
    plt.title('spikes in 5 minutes')

    k_means = KMeans(n_clusters=3).fit(spikes)

    plt.show()
