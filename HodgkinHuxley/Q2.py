import numpy as np
import matplotlib.pyplot as plt


# implement derivative and currents to use in runge kutta method
def d_omega(omega, v, tau_omega=144, a=4, e_l=-70.6):
    return (a * (v - e_l) - omega) / tau_omega


def I(t):
    if 0 <= t <= 200:
        return 0.5
    if 500 <= t <= 1000:
        return 0.8
    return 0


def I2(t):
    i = int(t / 100)
    if i % 2 == 0:
        return 2.5
    else:
        return 0


def d_v(v, t, omega, g_l=30, e_l=-70.6, delta_t=2, v_t=-50.4, c=281, i=I):
    return (1 / c) * (-1 * g_l * (v - e_l) +
                      g_l * delta_t * np.exp([(v - v_t) / delta_t]) +
                      i(t) - omega)


# runge kutta for the method
def next_rk4(v, t, omega, dt, i):

    # at spike time
    if v >= 20:
        v = -70.6
        omega = omega + 0.0805
        return v, omega

    v_k1 = dt * d_v(v, t, omega, i=i)
    omega_k1 = dt * d_omega(omega, v)
    new_v = v + 0.5 * v_k1
    new_omega = omega + 0.5 * omega_k1

    v_k2 = dt * d_v(new_v, t+0.5*dt, new_omega, i=i)
    omega_k2 = dt * d_omega(new_omega, new_v)
    new_v = v + 0.5 * v_k2
    new_omega = omega + 0.5 * omega_k2

    v_k3 = dt * d_v(new_v, t + 0.5 * dt, new_omega, i=i)
    omega_k3 = dt * d_omega(new_omega, new_v)
    new_v = v + v_k3
    new_omega = omega + omega_k3

    v_k4 = dt * d_v(new_v, t + dt, new_omega, i=i)
    omega_k4 = dt * d_omega(new_omega, new_v)

    v = v + (1 / 6) * (v_k1 + 2*v_k2 + 2*v_k3 + v_k4)
    omega = omega + (1 / 6) * (omega_k1 + 2*omega_k2 + 2*omega_k3 + omega_k4)

    return v, omega


if __name__ == "__main__":

    # first part
    v, omega = -65, 80
    v_l = np.zeros(10000)
    times = np.linspace(0, 1000, 10000)
    for i, t in enumerate(times):
        v, omega = next_rk4(v, t, omega, .1, I2)
        v_l[i] = v
    plt.figure('first')
    plt.plot(times, v_l)
    plt.ylabel('membrane potential (mv)')
    plt.xlabel('time (ms)')

    # second part
    v, omega = -65, 80
    for i, t in enumerate(times):
        v, omega = next_rk4(v, t, omega, .1, I)
        v_l[i] = v
    plt.figure('second')
    plt.plot(times, v_l)
    plt.ylabel('membrane potential (mv)')
    plt.xlabel('time (ms)')

    plt.show()


