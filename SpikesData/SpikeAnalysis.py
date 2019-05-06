from scipy.io import loadmat
import matplotlib.pyplot as plt


if __name__ == "__main__":
    data = loadmat("Data.mat")
    data = data['He'][0][:1920000]
    t = [i/32000 for i in range(32000*60)]
    plt.figure('pure signal')
    plt.plot(t, data)

    plt.show()
