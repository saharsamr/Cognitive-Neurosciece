import numpy as np
from matplotlib import pyplot as plt


def next_rk4(u, t, m, n, h, dt):
    u_k1 = dt * d_u(u, t, m, n, h)
    m_k1 = dt * d_m(u, m)
    n_k1 = dt * d_n(u, n)
    h_k1 = dt * d_h(u, h)
    new_u = u + 0.5 * u_k1
    new_m = m + 0.5 * m_k1
    new_n = n + 0.5 * n_k1
    new_h = h + 0.5 * h_k1

    u_k2 = dt * d_u(new_u, t+0.5*dt, new_m, new_n, new_h)
    m_k2 = dt * d_m(new_u, new_m)
    n_k2 = dt * d_n(new_u, new_n)
    h_k2 = dt * d_h(new_u, new_h)
    new_u = u + 0.5 * u_k2
    new_m = m + 0.5 * m_k2
    new_n = n + 0.5 * n_k2
    new_h = h + 0.5 * h_k2

    u_k3 = dt * d_u(new_u, t + 0.5 * dt, new_m, new_n, new_h)
    m_k3 = dt * d_m(new_u, new_m)
    n_k3 = dt * d_n(new_u, new_n)
    h_k3 = dt * d_h(new_u, new_h)
    new_u = u + u_k3
    new_m = m + m_k3
    new_n = n + n_k3
    new_h = h + h_k3

    u_k4 = dt * d_u(new_u, t + dt, new_m, new_n, new_h)
    m_k4 = dt * d_m(new_u, new_m)
    n_k4 = dt * d_n(new_u, new_n)
    h_k4 = dt * d_h(new_u, new_h)

    u = u + (1 / 6) * (u_k1 + 2*u_k2 + 2*u_k3 + u_k4)
    m = m + (1 / 6) * (m_k1 + 2*m_k2 + 2*m_k3 + m_k4)
    n = n + (1 / 6) * (n_k1 + 2*n_k2 + 2*n_k3 + n_k4)
    h = h + (1 / 6) * (h_k1 + 2*h_k2 + 2*h_k3 + h_k4)

    return u, m, n, h


def alpha_m(u):
    v = u + 40
    return (v / 10) / (1 - np.exp([(-1 * v) / 10], dtype=np.float128))


def beta_m(u):
    v = u + 65
    return 4 * np.exp([(-1 * v) / 18], dtype=np.float128)


def alpha_n(u):
    v = u + 55
    return (v / 100) / (1 - np.exp([(-1 * v) / 10], dtype=np.float128))


def beta_n(u):
    v = u + 65
    return np.exp([(-1 * v) / 80], dtype=np.float128) / 8


def alpha_h(u):
    v = u + 65
    return 0.07 * np.exp([(-1 * v) / 20], dtype=np.float128)


def beta_h(u):
    v = u + 35
    return 1 / (1 + np.exp([(-1 * v) / 10], dtype=np.float128))


def d_m(u, m):
    return alpha_m(u) * (1 - m) - beta_m(u) * m


def d_n(u, n):
    return alpha_n(u) * (1 - n) - beta_n(u) * n


def d_h(u, h):
    return alpha_h(u) * (1 - h) - beta_h(u) * h


def I(t):
    if 0 <= t <= 200:
        return 20
    return 0


def d_u(u, t, m, n, h, e_na=55, e_k=-77, e_l=-65, g_na=40, g_k=35, g_l=0.3, c=10, i=I):
    return (1 / c) * (
            (-1 * g_na * m**3 * h * (u - e_na)) +
            (-1 * g_k * n**4 * (u - e_k)) +
            (-1 * g_l * (u - e_l)) +
            i(t)
    )


if __name__ == "__main__":
    g_na = 40
    g_k = 35
    e_na = 55
    e_k = -77

    u_l = np.zeros(2000)
    m_l = np.zeros(2000)
    n_l = np.zeros(2000)
    h_l = np.zeros(2000)

    u, m, n, h = -60, 0.034, 0, 0.6
    times = np.linspace(0, 2, num=2000)
    for i, t in enumerate(times):
        u, m, n, h = next_rk4(u, t, m, n, h, .1)
        u_l[i] = u
        m_l[i] = m
        n_l[i] = n
        h_l[i] = h

    plt.figure("action potential")
    plt.plot(times, u_l)
    plt.ylabel("Membrane Voltage (mV)")
    plt.xlabel("Time (ms)")

    plt.figure("parameters-time")
    plt.plot(times, n_l, color="r", label="n")
    plt.plot(times, m_l, color="g", label="m")
    plt.plot(times, h_l, color="b", label="h")
    plt.ylabel("m, h, n")
    plt.xlabel("Time (ms)")
    plt.legend()

    g_na_l = np.multiply(m_l**3, h_l) * g_na
    g_k_l = n_l**4 * g_k
    plt.figure("g_k and g_na")
    plt.plot(times, g_k_l, color="r", label="g_k")
    plt.plot(times, g_na_l, color="b", label="g_na")
    plt.ylabel("Conductance for Na and K Channels (mS/cm^2)")
    plt.xlabel("Time (ms)")
    plt.legend()

    i_na_l = np.multiply(g_na_l, (u_l - e_na))
    i_k_l = np.multiply(g_k_l, (u_l - e_k))
    plt.figure("K and Na currents")
    plt.plot(times, i_k_l, color="b", label="K")
    plt.plot(times, i_na_l, color="r", label="Na")
    plt.ylabel("Current of K and Na Channels (micro A/cm^2)")
    plt.legend()

    plt.show()

