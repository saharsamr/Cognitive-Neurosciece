def next_y_rk4(f, t, y, dt):
    k1 = dt*f(t, y)
    k2 = dt*f(t+0.5*dt, y+0.5*k1)
    k3 = dt*f(t+0.5*dt, y+0.5*k2)
    k4 = dt*f(t+dt, y+k3)
    return y+(1/6)*(k1+k2+k3+k4)

