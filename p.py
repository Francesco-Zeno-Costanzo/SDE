import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


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


start_story = '2017-06-20' # Start of history
end_story   = '2023-06-20' # End of history
ticker      = '^GSPC'      # SP500

SP500 = yf.download(ticker, start=start_story, end=end_story, interval='1d', progress=False)
L = len(SP500)   # len in days

close_data = SP500['Close']
open_data  = SP500['Open']
ret = np.log(close_data/open_data)

# non so precisamente come settali
drift = (close_data[-1]/close_data[1])**(365.0/L) - 1 # annual return
volatility = 0.15#(close_data/close_data.shift(1)-1)[1:].std()*np.sqrt(L)

def f(z):
    mu = drift
    return mu*z

def g(z):
    sigma = volatility
    return sigma*z

def dg(z):
    sigma = volatility
    return sigma


# Parameter of simulation
M = 100  # numer of simulation
tf = 6   # time in years -> so dt=tf/(L-1) is 1 day in years unitis
data = np.zeros((L, M))

for i in range(M):

    t, sol = Heun(L-1, tf, close_data[0], f, g, dg)
    data[:, i] = sol

# so is easy to plot, same format of SP500
Sgbm = pd.DataFrame(data, index=SP500.index, columns=list(range(1, M+1)))

gmb_mean = data.mean(axis=1)
teo_mean = close_data[0]*np.exp(drift*t)

#Plot

plt.figure(1)
for i in range(1, M+1):
    plt.plot(Sgbm[i], 'b', lw=0.2)

plt.plot(close_data, 'r')
plt.xlabel('Date')
plt.ylabel('Close')
plt.grid()

plt.figure(2)
plt.plot(gmb_mean, 'b')
plt.plot(teo_mean, 'r')
plt.xlabel('Date')
plt.ylabel('mean')
plt.grid()

plt.show()
