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
from scipy.special import voigt_profile as voigt
from scipy.spatial import Delaunay
from astropy.modeling.models import Voigt1D

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

## Interpolate what we need ##
logD = np.log10(D)
get_logD = inter.RectBivariateSpline(radii.T[0],thb.T[0],logD)
get_T = inter.RectBivariateSpline(radii.T[0],thb.T[0],T,kx=1,ky=1)

#%%
### IMPORT LOG(F3) VALUES AND COORDINATES THEN INTERPOLATE ###
points = np.load('f3_coords_001_F3.npy')
logf3_values = np.load('logf3_values_001_F3.npy')

tri = Delaunay(points)   # does the triangulation
get_logf3 = inter.LinearNDInterpolator(tri,logf3_values)

#%%
### SET UP A CARTESIAN GRID TO ANALIZE ON ###
x = np.linspace(1.03,7,600)
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
plt.figure()
plt.contourf(X,Z,v_z,200)
#plt.quiver(X,Z,v_x,v_z,width=0.0005,headwidth=3,headlength=3)
plt.colorbar()

sign_uz = np.sign(v_z)

get_uz = inter.RectBivariateSpline(radii.T[0],thb.T[0],v_z)
get_sign_uz = inter.RectBivariateSpline(radii.T[0],thb.T[0],np.sign(v_z))

plt.figure()
plt.contourf(X,Z,get_uz(rax,tax,grid=False).T,200)
plt.colorbar()

#%%
e = 4.8032e-10          # electron charge
me = 9.1094e-28         # electron mass
c = 2.99792458e10       # speed of light

os = np.array([2.9958e-01, 1.7974e-01, 5.9902e-02])  #oscillator strengths in decreasing wl
nu_0 = np.array([c/(10830.33977e-8), c/(10830.25010e-8), c/(10829.09114e-8)])  #transition frequencies
A = np.array([5.1080e+07, 3.0648e+07, 1.0216e+07])   # Einstein coefficient of 10830A transition

R = float(100000)                        # telescope resolution
dlambda = 10830.33977e-8/R               # wl width of gaussian to convolve

nu_lim = np.abs((c/10832e-8)-(c/10830e-8))
nu = np.linspace((c/(10832e-8)),(c/(10828e-8)),500)    # frequency array
wl = c/nu

#%%
tau = []                       # list of optical depths from each column
for i in range(len(x)):        # calculate I_j at constant x values along z (ie y)
    t_j = []             # optical depth in cells along the ith column
    for j in range(len(z)):
        r_i = np.sqrt(x[i]**2+z[j]**2)
        th_i = np.arctan2(x[i],z[j])
        
        f3_cell = 10**get_logf3(x[i],z[j])
        D_cell = 10**get_logD(r_i,th_i,grid=False)
        T_cell = get_T(r_i,th_i,grid=False)        
        v_los = get_uz(r_i,th_i,grid=False)
        
        alpha = np.sqrt((2*np.log(2)*k*T_cell)/mHe)*(nu_0/c)  # Gaussian widths
        gamma = A/(4*np.pi)                                   # lorentzian widths
        
        nu_shift = nu_0 - nu_0*v_los/c            # doppler shifted frequencies
        peak= 2/(np.pi*gamma)                     # peak of the lorentzian for normalisation
        
        # Voigt line profiles
        v0 = Voigt1D(nu_shift[0],peak[0],gamma[0],alpha[0])
        v1 = Voigt1D(nu_shift[1],peak[1],gamma[1],alpha[1])
        v2 = Voigt1D(nu_shift[2],peak[2],gamma[2],alpha[2])
        
        Phi = np.array([v0(nu),v1(nu),v2(nu)])
        
        sigma = (((np.pi*(e**2)*os)/(me*c))*Phi.T).T
        
        # optical depth in i,j cell
        t_cell = (sigma[0]+sigma[1]+sigma[2])*XHe*D_cell*f3_cell*dy*rb[0]/mHe
        
        t_j.append(t_cell)
        
    t_i = np.nansum(np.asarray(t_j,dtype=float),axis=0,dtype=float) # optical depth of a column 
    tau.append(t_i)  

#%%
R_s = 6.955e10
R_D = rb[-1]
R_i = np.copy(x)*rb[0]
dR = np.diff(R_i)[0]

I_rings = []
for i in range(len(tau)):
    I_rings.append((np.exp(-tau[i]))*(2*R_i[i]*dR))
    
I_abs = (np.nansum(np.asarray(I_rings),axis=0))/(R_i[-1]**2)
np.save('I_abs_hydro.npy',I_abs)

plt.figure()
plt.axvline(nu_0[0])
plt.plot(nu,I_abs,'k')

plt.figure()
plt.axvline(c/nu_0[0]/1e-8)
plt.plot(wl/1e-8,I_abs,'b')

I_abs_mag = np.load('I_abs_mag.npy')
plt.plot(wl/1e-8,I_abs_mag,'r')

