"""
Created on Mon Jan 31 14:34:32 2022
@author: Simone Di Giampasquale
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from scipy.io import loadmat
from scipy import interpolate as inter
from scipy.integrate import solve_ivp as ivp
from scipy.spatial import Delaunay

#%%
## Load simulation data ##
hydro_sim = loadmat("pure_HD.mat")

rb = hydro_sim['r']              #1D array of radius for grid
thb = hydro_sim['th']            #1D array of theta for grid
vrb = hydro_sim['vr']            # u_r
vthb = hydro_sim['vth']          # u_theta
u = np.sqrt((vrb**2)+(vthb**2))  # |u|
D = hydro_sim['D']               # mass density
n = D/(1.00784*1.66e-24)         # number density
U = hydro_sim['U']               # internal energy pu volume
Xe = hydro_sim['ne']             # electron fraction ie fraction of ionised hydrogen
ne = n*Xe                        # electron number density
Xh = 1-Xe                        # fraction of non ionised hydrogen
n0 = n*Xh                        # neutral hydrogen number density
XHe = 0.3                        # mass fraction of hydrogen (ISM)
mHe = 4.002602*1.66e-24          # mass of Helium

## Calculate temperature ###
g = 5/3                  # adiabatic gamma for monoatomic gas
P = (g-1)*U              # pressure
mu = 1.00784*1.66e-24    # mean molecular mass
k = 1.380649e-16         # boltzmann constant in cgs units
T = P*mu/(D*k)           # temperature in cgs units

## Scale by radius of the planet and define cartesian and polar grids ##
radii = rb/rb[0]  
r_max = radii[-1][0]
rax,tax = np.meshgrid(radii,thb)
X = np.outer(radii,np.sin(thb))
Z = np.outer(radii,np.cos(thb))

## Interpolate what we need on a cartesian grid ##
logD = np.log10(D)
get_logD = inter.RectBivariateSpline(radii.T[0],thb.T[0],logD)

#%%
### IMPORT LOG(F3) VALUES AND COORDINATES THEN INTERPOLATE ###
points = np.load('f3_coords.npy')
logf3_values = np.load('logf3_values.npy')

tri = Delaunay(points)   # does the triangulation
get_logf3 = inter.LinearNDInterpolator(tri,logf3_values)

#%%
### SET UP A CARTESIAN GRID TO ANALIZE ON ###
x = np.linspace(1.03,8,600)
z = np.linspace(14,-14,1200)
x_grid, z_grid = np.meshgrid(x,z)

#%%
### CALCULATE COLUMN DENSITY ###
dx = np.diff(x)[0]
dy = np.abs(np.diff(z)[0])

N = []                         #column density at different impact parameters (x values)
for i in range(len(x)):        # calculate N_j at constant x values along z (ie y)
    f3_line = 10**get_logf3(x[i],z)
    r_i = np.sqrt(x[i]**2+z**2)
    th_i = np.arctan2(x[i],z)
    D_line = 10**get_logD(r_i,th_i,grid=False)
    
    N_j = np.nansum(XHe*D_line*f3_line*dy*rb[0]/mHe)
    N.append(N_j)

## weight N by area of rings pi*(R^2-r^2)
x_1 = x+dx
Areas = np.pi*((x_1**2)-(x**2))*(rb[0]**2)
## actually don't

plt.plot(x,np.asarray(N))
plt.yscale('log')

#%%
### PROJECT VELOCITY ALONG LINE OF SIGHT (Z COMPONENT) FOR RED/BLUE-SHIFT ###
v_x = (vrb*np.sin(tax.T)+vthb*np.cos(tax.T))
v_z = (vrb*np.cos(tax.T)-vthb*np.sin(tax.T))
plt.contourf(X,Z,v_z,200)
#plt.quiver(X,Z,v_x,v_z,width=0.0005,headwidth=3,headlength=3)
plt.colorbar()

sign_uz = np.sign(v_z)

get_uz = inter.RectBivariateSpline(radii.T[0],thb.T[0],v_z)
get_sign_uz = inter.RectBivariateSpline(radii.T[0],thb.T[0],np.sign(v_z))