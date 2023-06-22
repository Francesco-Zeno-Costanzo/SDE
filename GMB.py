import numpy as np
import matplotlib.pyplot as plt

# geometric Brownian motion
# Heun method

def Heun(N, tf, y0, f, g, dg, args_f=(), args_g=(), args_dg=()):
    '''
    Heun method for SDE: dx = f*dt + g*dw

    Parameters
    ----------
    N : int
        number of steps
    tf : float
        time extention
    y0 : float
        initial condition
    f, g : callable
        function in the equation
    dg : callable
        derivative of g
    args_f : tuple, optional
        extra argument to pass at f
    args_g : tuple, optional
        extra argument to pass at g
    args_dg : tuple, optional
        extra argument to pass at dg

    Returns
    -------
    ts : 1darray
        time
    ys : 1darray
        solution
    '''

    dt = tf/N  # time step

    ts = np.zeros(N + 1)
    ys = np.zeros(N + 1)

    ys[0] = y0 # initial condition

    for i in range(N):
        dW = np.random.normal(loc=0.0, scale=np.sqrt(dt)) # winer process

        y0 = ys[i] + f(ys[i], *args_f)*dt + g(ys[i], *args_g)*dW + 0.5*g(ys[i], *args_g)*dg(ys[i], *args_dg)*dW**2
        y1 = ys[i] + f(y0   , *args_f)*dt + g(y0   , *args_g)*dW + 0.5*g(y0   , *args_g)*dg(y0   , *args_dg)*dW**2

        ys[i+1] = 0.5*(y0 + y1)
        ts[i+1] = ts[i] + dt

    return ts, ys


if __name__ == '__main__':

    def f(z, mu):

        return mu*z

    def g(z, sigma):

        return sigma*z

    def dg(z, sigma):

        return sigma


    # Parameter of simulation
    N = 10000
    tf = 4
    y0 = 3
    mu = 1
    sigma = 0.5

    for _ in range(5):
        ts, ys = Heun(N, tf, y0, f, g, dg, args_f=(mu,), args_g=(sigma,), args_dg=(sigma,))
        plt.plot(ts, ys)

    plt.figure(1)
    plt.title('moto geometrico Browniano')
    plt.xlabel("time")
    plt.yscale('log')
    plt.grid()
    plt.show()
