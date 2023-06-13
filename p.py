import numpy as np
import matplotlib.pyplot as plt
import GMB

# geometric Brownian motion
# Heun method

def f(z):
    """
    funzione che moltiplica il dt
    """
    mu = 0.05
    return mu*z

def g(z):
    """
    funzione che moltimplica il processo di wiener
    """
    sigma = 0.5
    return sigma*z

def dg():
    """
    derivata di g
    """
    sigma = 0.5
    return sigma


#parametri simulazioni
N = 1000
M = 1000
tf = 10
dt = tf/N
data = np.zeros((N+1, M))
#faccio 5 simulazioni diverse
plt.figure(0)
for j in range(M):
    #array dove conservare la soluzione, ogni volta inizializzati
    ts = np.zeros(N + 1)
    ys = np.zeros(N + 1)

    ys[0] = 100#condizioni iniziali
    

    """
    for i in range(N):
        ts[i+1] = ts[i] + dt
        y0 = ys[i] + f(ys[i])*dt + g(ys[i])*dW(dt) + 0.5*g(ys[i])*dg()*(dW(dt)**2)
        y1 = ys[i] + f(y0)*dt    + g(y0)*dW(dt)    + 0.5*g(y0)*dg()*(dW(dt)**2)
        ys[i+1] = 0.5*(y0 + y1)
    """
    ts, ys = GMB.Heun(N, tf, ys[0], f, g, dg)

    data[:, j] = ys
    plt.plot(ts, ys)

ret = np.log(data[:-1, :]/data[1:, :])
plt.figure(1)
plt.title('moto geometrico Browniano')
plt.xlabel("time")
for i in range(M):
    plt.plot(ts[:-1], ret[:, i])
plt.grid()

plt.figure(2)
plt.hist(data[-1, :], bins=40)

plt.figure(3)
plt.hist(ret[-1, :], bins=40)
plt.show()
