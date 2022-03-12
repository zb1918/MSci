import numpy as np
from scipy.integrate import solve_ivp as ivp
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline as spline
from scipy.interpolate import interp1d as interp
from scipy.interpolate import interp2d as interp2
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from scipy import special
from scipy import constants


H_molecule = 1.00784                # mass of H in a.m.u.
amu = 1.66054e-24                   # 1 a.m.u. in g
H_mu = H_molecule * amu             # mass of a single H molecule in g
gamma = 5/3                         # 1 + 2/d.o.f. where d.o.f. of H is taken to be 3
kb = 1.38e-16                       # boltzmann constant in cgs
He_molecule = 4.0026                # mass of He in a.m.u.
He_mu = He_molecule * amu           # mass of a single He molecule in g
kb = 1.38e-16                       # boltzmann constant in c.g.s. (cm2 g s-2 K-1)   
c = constants.c                     # speed of light in m s-1
c_cgs = c * 100                     # speed of light in c.g.s. (cm s-1)


#-------------------------------misc. functions-------------------------------#
def sigmoid(x, L, x0, k, b):
    y = L / (1 + np.exp(-k * (x - x0))) + b
    return (y)

def flatten(t):
    return [item for sublist in t for item in sublist]

def make_fine(array, f):
    fine_array = []
    for i in range(len(array) - 1):
        sub_arr = np.linspace(array[i], array[i + 1], f + 1) 
        fine_array.append(sub_arr[0:f])
    fine_array = np.array(fine_array)
    fine_array = fine_array.flatten()
    return fine_array
 
    
#----------------------functional forms of known streamlines------------------#
def u_r(r, theta):
    """
    hello hello hello
    """
    return r**-0.5
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

def plot_cart(r, theta, color = "navy", lw = 1, ls = 'solid'):
    plt.plot(cart_x(r, theta), cart_y(r, theta), color = color, lw = lw, ls = ls)
    
def plot_mult(t, y, color = "navy", lw = 1, ls = 'solid'):
    
    for ys in range(len(y)):
        plot_cart(t, y[ys], color, lw = lw, ls = ls)   


#----------------------casting z onto given grid------------------------------#
def cast(coarse_r, coarse_t, z):
    """
    

    """
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
    #finds the interpolated functions fr and ft evaluated at finer points r, t
    #and forms the differential equation 
    #needed to be solved in the form dt/dr = u_t/(r*u_r)
    return ft(r, t, grid=False)/(fr(r, t, grid=False)*r)

def dtdy_rbs(t, r, fr, ft):   
    #finds the interpolated functions fr and ft evaluated at finer points r, t
    #and forms the differential equation 
    #needed to be solved in the form dt/dr = u_t/(r*u_r)
    return r*fr(r, t, grid=False)/ft(r, t, grid=False)

#--------------------------lambda functions for slider use--------------------#
def rad(r_min, r_lim, res):
    return lambda f: np.linspace(r_min, r_lim, res)


#--------------------------sigmoid fitting for temperature--------------------#
def fit_sigmoid(rads, thes, temp):
    """
    interpolates temperature into function f_T
    fits a sigmoid of f_T(r) for all angles
    
    Parameters
    ----------
    rads : 1d array
        radii at which temperature is determined.
    thes : 1d array
        angles at which temperature is determined.
    temp : 2d matrix
        temperature of grid.

    Returns
    -------
    None.

    """
    f_T = interp2(rads, thes, np.log(temp), kind = 'cubic')
 
    fine_rb = np.linspace(rads[0], rads[-1], len(rads)*100)
    
    for t in thes[0:116]:
        # temps = temp[np.array(np.where(thes == t)).item()]
        # temperatures along a single angle
        
        p0 = np.array([max(f_T(rads, t)), np.median(rads), 1, min(f_T(rads, t))]) # this is an mandatory initial guess
        popt, pcov = curve_fit(sigmoid, rads, f_T(rads, t), p0, method='lm')
        plt.plot(fine_rb, sigmoid(fine_rb, *popt), 'r-')
        
        #p0 = np.array([max(temps), np.median(rads), 1, min(temps)])
        #popt, pcov = curve_fit(sigmoid, rads, temps, p0, method='lm')
        #plt.plot(fine_rb, sigmoid(fine_rb, *popt), 'r-')
        
def ang_to_v(l):
    # converts wavelengths in Angstrom to freqencies in rad s-1
    return (c * 2 * np.pi) / (l * 1e-10)

        
def voigt(x, x0, A, T):
    '''
    returns voigt profile around central point v0
    constants sigma and gamma determined by v0 and A
    
    Parameters
    ----------
    x   : wavelength in Angstroms
    x0  : central wavelength in Angstroms
    A   : transition constant
    T   : temperature at location 
    
    Returns
    -------
    phi (voigt lineshape) at single frequency
    
    '''
    
    v = ang_to_v(x)
    v0 = ang_to_v(x0)
    
    sigma = np.sqrt(kb * T/ (2 * H_mu)) * (v0 / c_cgs)
    gamma = A / (4 * np.pi) # Lorentzian part's HWHM
    return special.voigt_profile(v - v0, sigma, gamma)
    



