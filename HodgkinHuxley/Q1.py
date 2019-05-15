import numpy as np
from matplotlib import pyplot as plt

# single stimulus
def I(t, strength):
    if 0 <= t <= 200:
        return strength
    return 0


# two stimulus with a delay
def I2(t, strength):
    if 0 <= t <= 0.2:
        return 20
    elif 15.2 <= t <= 15.4:
        return strength
    return 0

# runge kutta method for hodgkin huxley model. we should update parameters parallel.
# so in each step i should save the updated value for u, m, n, and h to use in next step.
def next_rk4(u, t, m, n, h, dt, strength, c=10, i=I):
    u_k1 = dt * d_u(u, t, m, n, h, strength, c=c, i=i)
    m_k1 = dt * d_m(u, m)
    n_k1 = dt * d_n(u, n)
    h_k1 = dt * d_h(u, h)
    new_u = u + 0.5 * u_k1
    new_m = m + 0.5 * m_k1
    new_n = n + 0.5 * n_k1
    new_h = h + 0.5 * h_k1

    u_k2 = dt * d_u(new_u, t+0.5*dt, new_m, new_n, new_h, strength, c=c, i=i)
    m_k2 = dt * d_m(new_u, new_m)
    n_k2 = dt * d_n(new_u, new_n)
    h_k2 = dt * d_h(new_u, new_h)
    new_u = u + 0.5 * u_k2
    new_m = m + 0.5 * m_k2
    new_n = n + 0.5 * n_k2
    new_h = h + 0.5 * h_k2

    u_k3 = dt * d_u(new_u, t + 0.5 * dt, new_m, new_n, new_h, strength, c=c, i=i)
    m_k3 = dt * d_m(new_u, new_m)
    n_k3 = dt * d_n(new_u, new_n)
    h_k3 = dt * d_h(new_u, new_h)
    new_u = u + u_k3
    new_m = m + m_k3
    new_n = n + n_k3
    new_h = h + h_k3

    u_k4 = dt * d_u(new_u, t + dt, new_m, new_n, new_h, strength, c=c, i=i)
    m_k4 = dt * d_m(new_u, new_m)
    n_k4 = dt * d_n(new_u, new_n)
    h_k4 = dt * d_h(new_u, new_h)

    u = u + (1 / 6) * (u_k1 + 2*u_k2 + 2*u_k3 + u_k4)
    m = m + (1 / 6) * (m_k1 + 2*m_k2 + 2*m_k3 + m_k4)
    n = n + (1 / 6) * (n_k1 + 2*n_k2 + 2*n_k3 + n_k4)
    h = h + (1 / 6) * (h_k1 + 2*h_k2 + 2*h_k3 + h_k4)

    return u, m, n, h


# implement runge kutta method with hodgkin huxley equation that affect the
# presence of tetraethylammonium
def next_rk4_tetraethylammonium(u, t, m, h, dt, strength, c=10, i=I):
    u_k1 = dt * d_u_tetraethylammonium(u, t, m, h, strength, c=c, i=i)
    m_k1 = dt * d_m(u, m)
    h_k1 = dt * d_h(u, h)
    new_u = u + 0.5 * u_k1
    new_m = m + 0.5 * m_k1
    new_h = h + 0.5 * h_k1

    u_k2 = dt * d_u_tetraethylammonium(new_u, t+0.5*dt, new_m, new_h, strength, c=c, i=i)
    m_k2 = dt * d_m(new_u, new_m)
    h_k2 = dt * d_h(new_u, new_h)
    new_u = u + 0.5 * u_k2
    new_m = m + 0.5 * m_k2
    new_h = h + 0.5 * h_k2

    u_k3 = dt * d_u_tetraethylammonium(new_u, t + 0.5 * dt, new_m, new_h, strength, c=c, i=i)
    m_k3 = dt * d_m(new_u, new_m)
    h_k3 = dt * d_h(new_u, new_h)
    new_u = u + u_k3
    new_m = m + m_k3
    new_h = h + h_k3

    u_k4 = dt * d_u_tetraethylammonium(new_u, t + dt, new_m, new_h, strength, c=c, i=i)
    m_k4 = dt * d_m(new_u, new_m)
    h_k4 = dt * d_h(new_u, new_h)

    u = u + (1 / 6) * (u_k1 + 2*u_k2 + 2*u_k3 + u_k4)
    m = m + (1 / 6) * (m_k1 + 2*m_k2 + 2*m_k3 + m_k4)
    h = h + (1 / 6) * (h_k1 + 2*h_k2 + 2*h_k3 + h_k4)

    return u, m, h


# implement function derivatives for calculating the result.
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


def d_u(u, t, m, n, h, strength, e_na=55, e_k=-72, e_l=-50, g_na=120, g_k=36, g_l=0.3, c=10, i=I):
    return (1 / c) * (
            (-1 * g_na * m**3 * h * (u - e_na)) +
            (-1 * g_k * n**4 * (u - e_k)) +
            (-1 * g_l * (u - e_l)) +
            i(t, strength)
    )


def d_u_tetraethylammonium(u, t, m, h, strength, e_na=55, e_l=-50, g_na=120, g_l=0.3, c=10, i=I):
    return (1 / c) * (
            (-1 * g_na * m**3 * h * (u - e_na)) +
            (-1 * g_l * (u - e_l)) +
            i(t, strength)
    )


# plot strength_duration diagram for finding rheobase and chronaxie
def plot_strength_duration_diagram():
    depolarization_threshold = -55
    strengths = [i for i in range(1, 50)]
    durations = [0 for _ in range(1, 50)]
    u_l = np.zeros(2000)
    m_l = np.zeros(2000)
    n_l = np.zeros(2000)
    h_l = np.zeros(2000)
    for j, strength in enumerate(strengths):
        u, m, n, h = -65, 0.034, 0, 0.6
        times = np.linspace(0, 2, num=2000)
        for i, t in enumerate(times):
            u, m, n, h = next_rk4(u, t, m, n, h, .01, strength=strength)
            u_l[i] = u
            m_l[i] = m
            n_l[i] = n
            h_l[i] = h
            if u >= depolarization_threshold:
                durations[j] = t * 0.01
                break

    plt.figure('strength-duration')
    plt.plot(durations, strengths)
    min_strength = min(strengths)
    plt.plot(durations, [min_strength for _ in durations], 'g--', label='rheobase')
    plt.ylabel("stimulus strength")
    plt.xlabel("stimulus duration")
    plt.legend()


if __name__ == "__main__":
    # plotting neuron spike:

    # initial values
    g_na = 120
    g_k = 36
    e_na = 55
    e_k = -77

    # lists to save value in different moment of stimulus and after that
    u_l = np.zeros(2000)
    m_l = np.zeros(2000)
    n_l = np.zeros(2000)
    h_l = np.zeros(2000)


    # find shape o spike
    u, m, n, h = -65, 0.034, 0, 0.6
    times = np.linspace(0, 2, num=2000)
    for i, t in enumerate(times):
        u, m, n, h = next_rk4(u, t, m, n, h, .01, strength=20)
        u_l[i] = u
        m_l[i] = m
        n_l[i] = n
        h_l[i] = h

    plt.figure("action potential")
    plt.plot(times, u_l)
    plt.ylabel("Membrane Voltage (mV)")
    plt.xlabel("Time (ms)")

    # plot m, n, and h in time
    plt.figure("parameters-time")
    plt.plot(times, n_l, color="r", label="n")
    plt.plot(times, m_l, color="g", label="m")
    plt.plot(times, h_l, color="b", label="h")
    plt.ylabel("m, h, n")
    plt.xlabel("Time (ms)")
    plt.legend()

    # plot g_k and g_na in time
    g_na_l = np.multiply(m_l**3, h_l) * g_na
    g_k_l = n_l**4 * g_k
    plt.figure("g_k and g_na")
    plt.plot(times, g_k_l, color="r", label="g_k")
    plt.plot(times, g_na_l, color="b", label="g_na")
    plt.ylabel("Conductance for Na and K Channels (mS/cm^2)")
    plt.xlabel("Time (ms)")
    plt.legend()

    # plot the current of K and Na channels
    i_na_l = np.multiply(g_na_l, (u_l - e_na))
    i_k_l = np.multiply(g_k_l, (u_l - e_k))
    plt.figure("K and Na currents")
    plt.plot(times, i_k_l, color="b", label="K")
    plt.plot(times, i_na_l, color="r", label="Na")
    plt.ylabel("Current of K and Na Channels (micro A/cm^2)")
    plt.xlabel("Time (ms)")
    plt.legend()

    # plot strength-duration diagram
    plot_strength_duration_diagram()

    # investigate the effect of increase in capacitor
    u, m, n, h = -65, 0.034, 0, 0.6
    for i, t in enumerate(times):
        u, m, n, h = next_rk4(u, t, m, n, h, .01, strength=20, c=20)
        u_l[i] = u
        m_l[i] = m
        n_l[i] = n
        h_l[i] = h

    plt.figure("action potential with c = 20")
    plt.plot(times, u_l)
    plt.ylabel("Membrane Voltage (mV)")
    plt.xlabel("Time (ms)")

    # investigate the effect of tetraethylammonium in HH equations
    u, m, n, h = -65, 0.034, 0, 0.6
    for i, t in enumerate(times):
        u, m, h = next_rk4_tetraethylammonium(u, t, m, h, .01, strength=20)
        u_l[i] = u
        m_l[i] = m
        h_l[i] = h

    plt.figure("action potential tetraethylammonium")
    plt.plot(times, u_l)
    plt.ylabel("Membrane Voltage (mV)")
    plt.xlabel("Time (ms)")

    # two stimulus with 15 ms delay
    u_l = np.zeros(17000)
    m_l = np.zeros(17000)
    n_l = np.zeros(17000)
    h_l = np.zeros(17000)

    u, m, n, h = -65, 0.034, 0, 0.6
    times = np.linspace(0, 17, num=17000)
    for i, t in enumerate(times):
        u, m, n, h = next_rk4(u, t, m, n, h, .01, strength=40, i=I2)
        u_l[i] = u
        m_l[i] = m
        n_l[i] = n
        h_l[i] = h

    plt.figure("action potential with  different current")
    plt.plot(times, u_l)
    plt.ylabel("Membrane Voltage (mV)")
    plt.xlabel("Time (ms)")

    plt.show()

