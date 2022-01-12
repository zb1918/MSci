"""
Created on Sun Dec 12 21:46:23 2021
@author: Simone Di Giampasquale
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import interpolate as inter
from scipy.integrate import solve_ivp as ivp
from scipy.optimize import curve_fit

#%%
# loads dictionary of MHD variables
# r - is b grid for r
# th - is the b grid for th
# v1 - is the vr velocity ALREADY averaged to the b grid
# v2 is the vth velocity ALREADY averaged to the b grid
# rho is the mass density 
# Xe is the electron fraction
# U is the energy density
# B1 is the Br magnetic field ALREADY averaged onto the b grid
# B2 is the Bth magnetid field ALREADY averaged onto the b grid

MHD_sim = loadmat("Bp3G_sim.mat")

rb = MHD_sim['r']       # radius 1D array
thb = MHD_sim['th']     # theta 1D array
vrb = MHD_sim['v1']     # u_r
vthb = MHD_sim['v2']    # u_theta
D = MHD_sim['rho']      # mass density
Br = MHD_sim['B1']      # B_r
Bth = MHD_sim['B2']     # B_theta
U = MHD_sim['U']        # internal energy pu volume
Xe = MHD_sim['Xe']      # electron fraction ie fraction of ionised hydrogen
Xh = 1-Xe               # fraction of non ionised hydrogen
F = 1e6                 # flux in erg/s/cm^2

B = np.sqrt((Br**2)+(Bth**2))   # absolute value of B field

## Calculate temperature ##
g = 5/3                  # adiabatic gamma for monoatomic gas
P = (g-1)*U              # pressure
mu = 1.00784*1.66e-24    # mean molecular mass
k = 1.380649e-16         # boltzmann constant in cgs units
T = P*mu/(D*k)           # temperature in cgs units

## Scale radii by planet radius ##
radii = rb/rb[0]
r_max = radii[-1][0]

rax,tax = np.meshgrid(radii,thb)  # radius, theta grids
X = np.outer(radii, np.sin(thb))  # x grid --> X AXIS POINTS TOWARDS THE STAR
Z = np.outer(radii, np.cos(thb))  # z grid --> Z AXIS PERPENDICULAR TO THE ORBIT

thetas = np.arange(0,np.pi,0.02)
theta_eval = np.array([thetas]).T

#%%
### INTERPOLATE B_r, B_th TO FIND FIELD LINES, WHICH COINCIDE WITH STREAMLINES

get_Br = inter.RectBivariateSpline(radii.T[0], thb.T[0], Br)
get_Bt = inter.RectBivariateSpline(radii.T[0], thb.T[0], Bth)

