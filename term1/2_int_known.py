
"""
testing 
streamlines of interpolation from discrete grid points of known velocity functions 
against 
streamlines of known velocity functions
"""

import numpy as np
import scipy as sp
from scipy.integrate import solve_ivp as ivp
import matplotlib.pyplot as plt


def u_r(r, theta):
    return 1

def u_theta(r, theta):
    return np.sin(theta)


def radial(r, theta):
     return u_theta(r, theta)/(u_r(r, theta)*r)

res = 100
inits = 15
r_lim = 9


thetas = np.linspace(0, 2, inits)*np.pi
thspan = np.array([thetas[0], thetas[-1]])

radii = np.linspace(1, r_lim, res)
radius = np.array([1])
rspan = np.array([radii[0], radii[-1]])

full_thetas = np.linspace(0,2,100)*np.pi
full_radius = np.repeat(radius[0], 100)
full_radii = np.linspace(radius[0], r_lim, r_lim+1)

offsets =np.linspace(0, 2*np.pi, 20)

sol = ivp(radial, rspan, thetas, t_eval = radii)


def cart_y(r, theta):
    return r*np.cos(theta)
def cart_x(r,theta):
    return r*np.sin(theta)
def plot_cart(r, theta, colour, lw = 2):
    plt.plot(cart_x(r, theta), cart_y(r, theta), color = colour, lw = lw)
    
for rad in full_radii:
    plot_cart(np.repeat(rad, 100), full_thetas, colour = "grey", lw = 0.5)    
for offset in offsets:
    plot_cart(np.linspace(1,r_lim,10), np.repeat(offset, 10), colour="grey", lw = 0.5)
plot_cart(full_radius, full_thetas, "black", lw = 1)

    
for ys in sol.y:
    #for offset in offsets:
    plot_cart(sol.t, ys, "red", lw = 1.5)
               
    
size = 5    
''''''
plt.ylim(-size,size)
plt.xlim(-size,size)

plt.show()

#%%



