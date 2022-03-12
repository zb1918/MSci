# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 10:21:17 2022

@author: zaza
"""

import numpy as np
import scipy as sc
from matplotlib import pyplot as plt
from scipy import constants
from scipy import special
from scipy import integrate
import stream_solvers

plt.style.use("cool-style.mplstyle")

def voigt(x, x0, A, T):
    # returns voigt profile around central point v0
    # constants sigma and gamma determined by v0 and A
    '''
    input
    x   : wavelength in Angstroms
    x0  : central wavelength in Angstroms
    A   : transition constant
    
    returns
    phi (voigt lineshape) at single frequency
    
    '''
    
    v = ang_to_v(x)
    v0 = ang_to_v(x0)
    sigma = np.sqrt(kb * T/ He_mu) * (v0 / c_cgs)
    gamma = A / (4 * np.pi) # Lorentzian part's HWHM
    return special.voigt_profile(v - v0, sigma, gamma)

def ang_to_v(l):
    # converts wavelengths in Angstrom to freqencies in s-1
    return (c) / (l * 1e-10) * 2 * np.pi

T = 5e3                             # temperature in K
H_molecule = 1.00784                # mass of H in a.m.u.
amu = 1.66e-24                      # 1 a.m.u. in g
H_mu = H_molecule * amu             # mass of a single H molecule in g
kb = 1.38e-16                       # boltzmann constant in c.g.s. (cm2 g s-2 K-1)   
c = constants.c                     # speed of light in m s-1
c_cgs = c * 100                     # speed of light in c.g.s. (cm s-1)
He_molecule = 4.0026                # mass of He in a.m.u.
He_mu = He_molecule * amu           # mass of a single He molecule in g
w_min = 10825
w_max = 10835

#-------------- wavelengths in angstrom --------------------------------------#
w0 = 10829.09114
w1 = 10830.25010               
w2 = 10830.33977
      
#-------------- oscillator strengths -----------------------------------------#
fik0 = 5.9902e-2
fik1 = 1.7974e-1
fik2 = 2.9958e-1

#-------------- gkAki in s-1 -------------------------------------------------#
gA0 = 1.0216e7
gA1 = 3.0648e7
gA2 = 5.1080e7

x = np.linspace(w_min, w_max, 1000)

fig, ax = plt.subplots()
ax.ticklabel_format(style = 'plain', axis = 'x')
plt.plot(x, fik0 * voigt(x, w0, gA0, T), label = "j = 0")
plt.plot(x, fik1 * voigt(x, w1, gA1, T), label = "j = 1")
plt.plot(x, fik2 * voigt(x, w2, gA2, T), label = "j = 2")
plt.xlabel("λ (Å)")
plt.ylabel(r"$f_{ik}$ $\phi$ ($\nu$)")
plt.xticks([w_min, 0.5 * (w_min + w_max), w_max])
plt.legend()

#%%
# doppler (only gaussian)
uf = 0
v0 = ang_to_v(w1)

def doppler(w, w0):
    v = ang_to_v(w)
    v0 = ang_to_v(w0)
    
    A = np.sqrt(H_mu / (2 * np.pi * kb * T))
    del_u = c_cgs * ((v - v0) / v0)
    del_u -= uf
    B = H_mu / (2 * kb * T)
    
    return A * np.exp(-B * del_u**2) * c_cgs / v0

#%%
y_voigt = []
for xi in x:
    y_voigt.append(voigt(xi, v0, gA1))
int_voigt = np.trapz(y_voigt, x)
