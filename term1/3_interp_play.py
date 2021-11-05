import numpy as np
from scipy.integrate import solve_ivp as ivp
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline as spline
import scipy.interpolate as inter

x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8 ,9, 10])


y = np.sin(x)

fx = inter.interp1d(x, y)


xnew = np.linspace(0, 10, 100)
plt.plot(xnew, fx(xnew))
#plt.plot(x, y)
