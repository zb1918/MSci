# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 15:57:26 2022

@author: zaza
"""

import numpy as np
import scipy as sc
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp as ivp
import stream_solvers as slm
from scipy.interpolate import interp1d as interp

#---------------------------constants in cgs----------------------------------#
H_molecule = 1.00784        # mass of H in a.m.u.
amu = 1.66e-24              # 1 a.m.u. in g
H_mu = H_molecule * amu     # mass of a single H molecule in g
gamma = 5/3                 # 1 + 2/d.o.f. where d.o.f. of H is taken to be 3
kb = 1.38e-16               # boltzmann constant in cgs

#------------------------calculating temperature------------------------------#
T = 1000


F1 = ((10e3)/(24.6*1.602e-12))*(7.82e-18)   # photoionisation rate of 1S
F3 = ((10e3)/(4.8*1.602e-12))*(7.82e-18)    # photoionisation rate of 2S triplet

t1 = 0                                      # lifetime of 1S
t3 = 0                                      # lifetime of 2S triplet

a1 = 1.54e-13                               # recombination rate of 1S 
a3 = 1.49e-13                               # recombination rate of 2S triplet

## Atomic factors ###
A31 = 1.272e-4                  # radiative de-excitation rate 2S tri -> 1S
q13a = 4.5e-20                  # collisional excitation rate 1S -> 2S tri
q31a = 2.6e-8                   # collisional excitation rate 2S tri -> 2S sing
q31b = 4.0e-9                   # collisional excitation rate 2S tri -> 2P sing
Q31 = 5e-10                     # collisional de-exciration rate 2S tri -> 1S


u_i = 1e6


li = np.linspace(5e9, 1e11, 100)
l0 = [li[0], li[-1]]

n = np.linspace(5e6, 1e6, 100)

f_ne = interp(li, n)

def rhs(l, f):
    f1 = f[0]
    f3 = f[1]
    
    ne = f_ne(l).item()
    n0 = f_ne(l).item()
    u = u_i
    
    g1 = u**-1 * (
        (1- f1 - f3) * ne * a1 + 
        f3 * A31 - 
        f1 * F1 * np.exp(-t1) - 
        f1 * ne * q13a + 
        f3* ne * q31a + 
        f3 * ne * q31b + 
        f3 * n0 * Q31
        )
    g2 = u**-1 * (
        (1- f1 - f3) * ne * a3 - 
        f3 * A31 - 
        f3 * F3 * np.exp(-t3) + 
        f1 * ne * q13a - 
        f3* ne * q31a - 
        f3 * ne * q31b - 
        f3 * n0 * Q31
        )
    g = [g1, g2]
    return g


sol_f = ivp(rhs, l0, [1, 0], method = 'LSODA', t_eval = li,
                    rtol = 1e-10, atol = 1e-7)

n3 = sol_f.y[1]
x = sol_f.t
plt.plot(x, n3)