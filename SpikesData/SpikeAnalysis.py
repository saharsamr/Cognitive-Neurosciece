from scipy.io import loadmat
from scipy.signal import butter, lfilter, find_peaks
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


# Implementing a low pass filter
def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


# Implementing a high pass filter
def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def find_spike_samples_and_plot(spike_peaks, spike_activities, label):
    plt.figure(label)
    spikes = np.zeros((len(spike_peaks), 31))
    for i, peak in enumerate(spike_peaks):
        # After finding peaks, we should select samples that are related to that
        # peak to have the shape of an spike so we select 11 samples befor and 20
        # samples after each peak.
        spikes[i] = spike_activities[peak - 11:peak + 20]
        plt.plot(spikes[i] - min(spikes[i]))
    plt.title(label)
    return spikes


if __name__ == "__main__":
    # loading data from .mat file.
    data = loadmat("Data.mat")
    # seperate the samples for first 1 minute of record.
    data_one_min = data['He'][0][:1920000]
    t = [i/32000 for i in range(32000*60)]
    # plot the whole signal
    plt.figure('pure signal')
    plt.plot(t, data_one_min, label='pure signal')
    plt.legend()

    # Filter the data with low pass filter, and plot filtered signals (LFP).
    y = butter_lowpass_filter(data_one_min, 300, 32000)
    plt.figure('filtered data(low pass filter)')
    plt.plot(t, y, label='filtered signal (low pass)')
    plt.legend()

    # Filter the data with high pass filter, and plot filtered signals (spike activities).
    spike_activities = butter_highpass_filter(data_one_min, 300, 32000)
    plt.figure('filtered data(high pass filter)')
    plt.plot(t, spike_activities, label='filtered signal (high pass)')
    plt.legend()

    plt.figure('spikes in 5 minutes')
    # seperate the five first minutes of samples.
    data_five_min = data['He'][0][:5*1920000]
    # and achieve the spike activities of that.
    spike_activities = butter_highpass_filter(data_five_min, 300, 32000)
    axes = plt.gca()
    axes.set_ylim([-0.0001, 0.0003])
    # find spike peaks with a threshold of 0.008 mV to distinguish spikes.
    spike_peaks = find_peaks(spike_activities, threshold=0.000008)[0]
    # as discussed in the function, get the samples related to each spike
    # peak to form an action potential
    spikes = find_spike_samples_and_plot(spike_peaks, spike_activities, 'spikes in 5 minutes')

    # cluster the whole spikes in previous parts using k-means algorithm
    k_means = KMeans(n_clusters=3).fit(spikes)

    # here we seperate the data of each cluster.
    cluster1 = []
    cluster2 = []
    cluster3 = []
    for i, label in enumerate(k_means.labels_):
        if label == 0:
            cluster1.append(spike_peaks[i])
        elif label == 1:
            cluster2.append(spike_peaks[i])
        else:
            cluster3.append(spike_peaks[i])

    # find spikes samples according to peaks of each cluster and plot the action potentials.
    find_spike_samples_and_plot(cluster1, spike_activities, 'spikes for first cluster')
    find_spike_samples_and_plot(cluster2, spike_activities, 'spikes for second cluster')
    find_spike_samples_and_plot(cluster3, spike_activities, 'spikes third first cluster')

    # use PCA to reduce feature space dimension to 2 (from 31).
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(spikes)

    # after redusing the feature space dimension, we use k-means again to cluster data
    # with these 2 remained features.
    k_means = KMeans(n_clusters=3).fit(principalComponents)

    # seperating the data of each cluster.
    cluster1 = []
    cluster2 = []
    cluster3 = []
    for i, label in enumerate(k_means.labels_):
        if label == 0:
            cluster1.append(principalComponents[i])
        elif label == 1:
            cluster2.append(principalComponents[i])
        else:
            cluster3.append(principalComponents[i])

    # plotting the scatter of those 3 cluster in one plot with different colors.
    plt.figure('scatter after pca')
    axes = plt.gca()
    axes.set_ylim([-0.0005, 0.0005])
    axes.set_xlim([-0.00026, 0.0004])
    plt.scatter([d[0] for d in cluster1], [d[1] for d in cluster1], c='r', label='cluster1')
    plt.scatter([d[0] for d in cluster2], [d[1] for d in cluster2], c='b', label='cluster2')
    plt.scatter([d[0] for d in cluster3], [d[1] for d in cluster3], c='g', label='cluster3')
    plt.legend()

    plt.show()
