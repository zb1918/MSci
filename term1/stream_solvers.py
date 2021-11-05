import numpy as np
from scipy.integrate import solve_ivp as ivp
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline as spline


plt.style.use("cool-style.mplstyle")

#----------------------functional forms of known streamlines------------------#
def u_r(r, theta):
    """
    hello hello hello
    """
    return 1
def u_theta(r, theta):
    return np.cos(theta)
def dydt(r, theta, f_r, f_t):
    #returns the radial differential equation to be solved 
    #in the form of dt/dr = u_t/(r*u_r)
    return f_t(r, theta)/(f_r(r, theta)*r)

def u_r_new(a):
    return lambda r,t: r*a

def u_theta_new(a):
    return lambda r,t: np.cos(t*a)




#---------------------------plotting in cartesian-----------------------------#
#----------------------from spherical polar cast in 2D------------------------#
def cart_y(r, theta):
    return r*np.cos(theta)
def cart_x(r,theta):
    return r*np.sin(theta)
def plot_cart(r, theta, colour = "green", lw = 1):
    plt.plot(cart_x(r, theta), cart_y(r, theta), color = colour, lw = lw)

def plot_mult(t, y, color = "green", lw = 1):
    
    for ys in range(len(y)):
        plot_cart(t, y[ys], color, lw = lw)   


#----------------------casting z onto given grid------------------------------#
def cast(coarse_r, coarse_t, z, args =1):
    #function z is taken and cast onto coarse_r coarse_t grid
    #returns 2D array of the value of z at each point in the grod
    
    z_cast = []
    for r in coarse_r:
        z_cast_0 = []
        for t in coarse_t:
            z_cast_0.append(z(r, t))
        z_cast.append(z_cast_0)
    return z_cast

#----------------------generation and evaluation of---------------------------#
#------------interpolated functional forms of streamlines---------------------#

def rbs(r, t, z):    
    #returns the function as a result of having interpolated z
    return spline(r, t, z)

def dydt_rbs(r, t, fr, ft):   
    #finds the nterpolated functions fr and ft evaluated at finer points r, t
    #and forms the differential equation 
    #needed to be solved in the form dt/dr = u_t/(r*u_r)
    return ft(r, t, grid=False)/(fr(r, t, grid=False)*r)
    