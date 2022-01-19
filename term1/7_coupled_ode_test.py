# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 10:41:41 2022

@author: zaza
"""

import numpy as np
import scipy as sc
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp as ivp


def get_rhs(f, a, b):           # need to find an expression for r and theta from l
   
    g1 = a*f[0] + b*f[1]
    g2 = b*f[0] + a*f[1]
    
    g = np.array([g1,g2])
    
    return g

#%%


T = 1e4                                    # Temperature
ne = 1e-16/(1.00784*1.66e-24)              # electron number density
n0 = 0.001*ne                              # neutral hydrogen number density
F1 = ((10e3)/(24.6*1.602e-12))*(7.82e-18)  # photoionisation rate of 1S
F3 = ((10e3)/(4.8*1.602e-12))*(7.82e-18)   # photoionisation rate of 2S triplet
u = 1e6                                    # absolute value of velocity
a1 = 1.54e-13                              # recombination rate of 1S 
a3 = 1.49e-12                              # recombination rate of 2S triplet
## Atomic factors ###
A31 = 1.272e-4                  # radiative de-excitation rate 2S tri -> 1S
q13a = 4.5e-20                  # collisional excitation rate 1S -> 2S tri
q31a = 2.6e-8                   # collisional excitation rate 2S tri -> 2S sing
q31b = 4.0e-9                   # collisional excitation rate 2S tri -> 2P sing
Q31 = 5e-10                     # collisional de-exciration rate 2S tri -> 1S

A = (-ne*a1-F1-ne*q13a)
B = (-ne*a1+A31+ne*q31a+ne*q31b+n0*Q31)
C = (-ne*a3+ne*q13a)
D = (-ne*a3-A31-F3-ne*q31a-ne*q31b-n0*Q31)

def g(l,frac):
    n1 = frac[0]
    n3 = frac[1]
    
    g1 = (ne*a1 + A*n1 + B*n3)/u
    g2 = (ne*a3 + C*n1 + D*n3)/u
    
    rhs = np.array([g1,g2])
    return rhs

f0 = np.array([1,0])
l0 = (1e10,5e10)
l_eval = np.linspace(l0[0],l0[-1],200)

f_sol = ivp(g, l0, f0, method='LSODA', atol=1e-10)

l = f_sol.t
n1_sol = f_sol.y[0]
n3_sol = f_sol.y[1]

plt.figure()
plt.plot(l,f_sol.y[0],'k')
#plt.plot(l,n3_sol,'b')

plt.show()