
"""
finding streamlines based on given function f(u_x, u_y)
"""

import numpy as np
import scipy as sp
from scipy.integrate import solve_ivp as ivp
import matplotlib.pyplot as plt


def f(u_x, u_y): 
    return 0.65*u_y/u_x
    #return 1

x = np.linspace(0.0001, 1, 50)
y0 = np.linspace(0, 0.01, 10)


solq1 = ivp(f, [-0.0001,-1], y0, t_eval = -x)
solq2 = ivp(f, [0.0001,1], y0, t_eval = x)
solq3 = ivp(f, [-0.0001,-1], -y0, t_eval = -x)
solq4 = ivp(f, [0.0001,1], -y0, t_eval = x)



for xs in range(len(y0)):
    plt.plot(x, solq2.y[xs])
    plt.plot(-x,solq1.y[xs])
    plt.plot(-x,solq3.y[xs])
    plt.plot(x,solq4.y[xs])

plt.ylim(-1,1)
plt.xlim(-1,1)
plt.show()

#%%

def f(u_x, u_y): return u_y/u_x


x = np.array([0.0001, 1])
y0 = np.linspace(0, 0.001, 50)


solq2 = ivp(f, [0.0001,1], y0, t_eval = x)


for xs in range(len(y0)):
    plt.plot(x, solq2.y[xs])
    
plt.ylim(-1,1)
plt.xlim(-1,1)
plt.show()

#%%

