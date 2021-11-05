
"""
finding streamlines based on given function f(u_x, u_y)
"""

import numpy as np
import scipy as sp
from scipy.integrate import solve_ivp as ivp
import matplotlib.pyplot as plt


size = 4
y0 = size * -1
x0 = y0

points = 100




x = np.linspace(x0, size, points)
y = np.linspace(x0, size, points)
X, Y = np.meshgrid(x, y)

y0 = np.linspace(x0, 2, size)

x_hat = np.array([1,0])
y_hat = np.array([0,1])

#u_x = 1
#u_y = 1 

def vel(u_x, u_y): return u_x*x_hat + u_y*y_hat


def f(u_x, u_y): return u_y/u_x

sol = ivp(f, [x0, size], [x0], t_eval = x)

ts = []
for x in range(points):
    ts.append(np.ndarray.tolist(sol.t))
ts = np.array(ts)

ys = []
for x in sol.y[0]:
    one_y = []
    for y in range(len(sol.y[0])):
        one_y.append(x)
    ys.append(one_y)        
ys = np.array(ys)      

fig = plt.figure(figsize = (12, 7))
  

plt.streamplot(X, Y, ts, ys, density = 0.5)
#plt.ylim(-0.5, 4)

