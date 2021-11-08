
"""
Created on Mon Nov  8 12:55:48 2021

@author: simon
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate as inter
from scipy.integrate import solve_ivp as ivp
from mpl_toolkits.mplot3d import Axes3D

#%%
### create a coarse grid and define a function over it ###

coarse_x = np.arange(-20,21,2)
coarse_y = np.arange(-10,11,1)

xax_c, yax_c = np.meshgrid(coarse_x,coarse_y)

def test_f(x,y):
    f = np.exp((-(x**2)-(y**2))/16)
    return f

fun = test_f(xax_c, yax_c)

#%%
### interpolate the function and evaluate it on a finer grid ###

get_f = inter.RectBivariateSpline(coarse_x, coarse_y, fun)

fine_x = np.arange(-20, 20.2, 0.2)
fine_y = np.arange(-10, 10.1, 0.1)

xax_f, yax_f = np.meshgrid(fine_x, fine_y)

int_fun = get_f(xax_f, yax_f, grid=False)

#%%
### plot the two functions on the two grids ###

fig1 = plt.figure()
ax = Axes3D(fig1, auto_add_to_figure=False)
fig1.add_axes(ax)
ax.plot_surface(xax_c, yax_c, fun) 
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#%%
fig2 = plt.figure()
ax = Axes3D(fig2, auto_add_to_figure=False)
fig2.add_axes(ax)
ax.plot_surface(xax_f, yax_f, int_fun) 
plt.xlabel('x')
plt.ylabel('y')
plt.show()

