"""
Created on Mon Feb 21 13:39:23 2022
@author: Simone Di Giampasquale
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from scipy.io import loadmat
from scipy import interpolate as inter
from scipy.integrate import solve_ivp as ivp
from scipy.special import voigt_profile as voigt
from scipy.spatial import Delaunay
from astropy.modeling.models import Voigt1D
from numpy import random as rd
from scipy import optimize as op

#%%
### CONVOLVE ABSORBED INTENSITY WITH A GAUSSIAN FROM TELESCOPE ###

R = float(100000)                   # telescope resolution
dlambda = 10830.33977e-8/R          # width of the gaussian

I_abs_hydro = np.load('I_abs_hydro.npy')
I_abs_mag = np.load('I_abs_mag.npy')

c = 2.99792458e10
nu = np.linspace((c/(10832e-8)),(c/(10828e-8)),500)    # frequency array
wl = c/nu                                              # wavelength array
wl0 = 10830.33977e-8

# not-normalised Gaussians for telescope
G_tel_mag_nn = np.exp((-(wl-wl0-(0.032e-8))**2)/(2*(dlambda**2)))    
G_tel_hydro_nn = np.exp((-(wl-wl0+(0.389e-8))**2)/(2*(dlambda**2)))

# normalise by dividing by the sum
G_tel_mag = G_tel_mag_nn/np.sum(G_tel_mag_nn)
G_tel_hydro = G_tel_hydro_nn/np.sum(G_tel_hydro_nn)

I_obs_mag = -np.convolve(-I_abs_mag+0.9803,G_tel_mag,mode='same')+0.9803
I_obs_hydro = -np.convolve(-I_abs_hydro+0.98,G_tel_hydro,mode='same')+0.98

plt.figure()
plt.plot(wl/1e-8,-G_tel_hydro+0.98,'b')
plt.plot(wl/1e-8,I_abs_hydro,'b')
plt.plot(wl/1e-8,I_obs_hydro,'k')
plt.axvline(x=wl0/1e-8)
plt.show()

plt.figure()
plt.plot(wl/1e-8,-G_tel_mag+0.9803,'r')
plt.plot(wl/1e-8,I_abs_mag,'r')
plt.plot(wl/1e-8,I_obs_mag,'k')
plt.axvline(x=wl0/1e-8)
plt.show()

#%%
### PICK RANDOM POINT FROM THE MODEL WITH ERROR BARS
### CURVEFIT A GAUSSIAN TO THE DATAPOINTS
### COMPARE POSITION OF THE SHIFT TO SIGMAS
### BLUESHIFT DETECTABLE IF AWAY BY ~2 SIGMA

get_I_hydro = inter.CubicSpline(wl[::-1],I_obs_hydro[::-1])
def G(l,A0,A,l0,sigma):
    gauss = A0 - A*np.exp(-((l-l0)**2)/(2*sigma**2))
    return gauss

yes=0
no=0
for i in range(1000):    
    data_wl =np.arange(10829e-8,10831e-8,dlambda)
    random_scaling = rd.uniform(-0.004,0.004,len(data_wl))
    data_I = (1-random_scaling)*get_I_hydro(data_wl)
    data_er = 0.002*data_I
        
    popt, pcov = op.curve_fit(G,data_wl,data_I,p0=[0.98,0.04,wl0-0.389e-8,2e-9],sigma=data_er)
    shift = np.abs(popt[2]-wl0)
    
    #plt.plot(wl,I_obs_hydro,'k')
    #plt.axvline(x=wl0)
    #plt.errorbar(data_wl,data_I,yerr=data_er,fmt='rs')
    #plt.plot(wl,G(wl,popt[0],popt[1],popt[2],popt[3]))
    
    if shift > 5*np.sqrt(pcov[2,2]):
        yes+=1
    else:
        no+=1