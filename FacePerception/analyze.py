import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy as sy


def psycho_func(x, alpha=0, beta=1):
    return 1. / (1 + np.exp(-(x-alpha)/beta))


def extract_data(data, angle, morph_level):
    extracted = np.array([[i[1], i[2]] for i in data if int(i[0]) == int(angle)])
    proportion = []
    for level in morph_level:
        male = 0
        female = 0
        for sample in extracted:
            if sample[0] == level and sample[1] == 1:
                male += 1
            elif sample[0] == level and sample[1] == 0:
                female += 1
        proportion.append(female/(male+female))
    return proportion


def plot_subjects(data, morph_levels, label):
    plt.figure(label)
    par0 = sy.array([0, 1])
    prop_90 = extract_data(data, 90, morph_levels)
    prop_180 = extract_data(data, 180, morph_levels)
    fake_morphs = np.linspace(-50, 50, 100)

    par, curve = curve_fit(psycho_func, morph_levels, prop_90, par0, maxfev=1000)
    plt.plot(fake_morphs, psycho_func(fake_morphs, par[0], par[1]), label='90')

    par, curve = curve_fit(psycho_func, morph_levels, prop_180, par0, maxfev=1000)
    plt.plot(fake_morphs, psycho_func(fake_morphs, par[0], par[1]), label='180')

    plt.xlabel('Female morphing signal')
    plt.ylabel('Proportion of female responses')

    plt.legend()


if __name__ == "__main__":
    path = "result/*.csv"
    morphs = [-50, -37.5, -25, -12.5, 0, 12.5, 25, 37.5, 50]
    for fname in glob.glob(path):
        data = np.genfromtxt(fname, delimiter=',')
        plot_subjects(data, morphs, fname)
    plt.show()
