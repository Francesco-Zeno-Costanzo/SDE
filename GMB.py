import numpy as np
import matplotlib.pyplot as plt

# geometric Brownian motion
# Heun method

def Heun(N, tf, y0, f, g, dg):
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
        
        y0 = ys[i] + f(ys[i])*dt + g(ys[i])*dW + 0.5*g(ys[i])*dg()*dW**2
        y1 = ys[i] + f(y0)   *dt + g(y0)   *dW + 0.5*g(y0)   *dg()*dW**2
        
        ys[i+1] = 0.5*(y0 + y1)
        ts[i+1] = ts[i] + dt
    
    return ts, ys


def f(z):
    mu = 1
    return mu*z

def g(z):
    sigma = 0.5
    return sigma*z

def dg():
    sigma = 0.5
    return sigma
  
# Parameter of simulation
N = 10000
tf = 4
y0 = 3

for _ in range(5):
    ts, ys = Heun(N, tf, y0, f, g, dg)
    plt.plot(ts, ys)

plt.figure(1)
plt.title('moto geometrico Browniano')
plt.xlabel("time")
plt.yscale('log')
plt.grid()
plt.show()
