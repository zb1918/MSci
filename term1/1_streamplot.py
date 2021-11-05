"""
displaying streamlines with manually inputted velocities (u,v)
"""

import numpy as np
import scipy as sp
from scipy.integrate import solve_ivp as ivp
import matplotlib.pyplot as plt


size = 2
x = np.linspace(0.01, size, size)
y = np.linspace(0.01, size, size)
X, Y = np.meshgrid(x, y)


u = np.array([[0.01,1],
             [0.01,1]])

v = np.array([[0.01,0.01],
             [1,1]])


plt.streamplot(X, Y, u, v, density = 0.2)

