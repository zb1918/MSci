# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 11:36:22 2022

@author: zaza
"""

import numpy as np
import scipy as sc
from matplotlib import pyplot as plt
import stream_solvers as slm
from scipy.io import loadmat
from scipy import interpolate as inter
from scipy.integrate import solve_ivp as ivp

mhd_sim = loadmat("term1/mhd_sim.mat")

stream_no = 5

f = np.loadtxt("term1/sols/mhd_sol_t.csv", dtype = 'f', delimiter = ',', unpack = True)

f = np.loadtxt("term1/sols/mhd_sol_y.csv", dtype = 'f', delimiter = ',', unpack = True)

for stream_no in range(0, 33):
    sol_y = f.T[stream_no]
    sol_t = f.T[stream_no]
    
    
    # plot the streamline investigated:
    #slm.plot_cart(sol_t, sol_y)
    
    rb = mhd_sim['r']                   #1D array of radius for grid
    thb = mhd_sim['th']                 #1D array of theta for grid
    
    rb = rb.T[0]
    rb_sc = rb/abs(rb[0])
    rb_sc_max = rb_sc[-1]
    thb = thb.T[0]
    
    X = np.outer(rb_sc, np.sin(thb))
    Z = np.outer(rb_sc, np.cos(thb))
    
    Br = mhd_sim['B1']
    Bt = mhd_sim['B2']
    Bc = mhd_sim['B3']
    
    vrb = mhd_sim['v1']                 # u_r
    vthb = mhd_sim['v2']                # u_theta
    vc = mhd_sim['v3']
    u = np.sqrt((vrb**2) + (vthb**2))   # |u|
    
    D = mhd_sim['rho']                    # mass density
    n = D / (1.00784 * 1.66e-24)        # number density
    
    U = mhd_sim['U']                    # internal energy pu volume
    
    Xe = mhd_sim['Xe']                  # electron fraction ie fraction of ionised hydrogen
    ne = n * Xe                         # electron number density
    Xh = 1 - Xe                         # fraction of non ionised hydrogen
    n0 = n * Xh                         # neutral hydrogen number density
    
    f_r = slm.rbs(rb_sc, thb, vrb)
    f_t = slm.rbs(rb_sc, thb, vthb)
    f_Br = slm.rbs(rb_sc, thb, Br)
    f_Bt = slm.rbs(rb_sc, thb, Bt)
    f_u = slm.rbs(rb_sc, thb, u)
    f_ne = slm.rbs(rb_sc, thb, ne)
    f_n0 = slm.rbs(rb_sc, thb, n0)
    
    '''
    plt.contourf(X, Z, ne, 64, cmap = "BuPu")
    c = plt.colorbar()
    c.set_label(r"$\log_{10}$(ne) [g/cm3]")
    
    
    ne_i = []
    for r in sol_t:
        
        i = np.where(sol_t == r)[0].item() # index of r in sol_t (i.e. where it is)
        t = sol_t[i]
        ne_i.append(f_ne(r, t).item())
        
    #plt.plot(sol_t, ne_i)
    '''
    
    
    #---------------------------constants in cgs----------------------------------#
    H_molecule = 1.00784        # mass of H in a.m.u.
    amu = 1.66e-24              # 1 a.m.u. in g
    H_mu = H_molecule * amu     # mass of a single H molecule in g
    gamma = 5/3                 # 1 + 2/d.o.f. where d.o.f. of H is taken to be 3
    kb = 1.38e-16               # boltzmann constant in cgs
    
    #------------------------calculating temperature------------------------------#
    T = U * H_mu * (gamma - 1)/(D * kb)
    T = T.T
    
    
    F1 = ((10e3)/(24.6*1.602e-12))*(7.82e-18)   # photoionisation rate of 1S
    F3 = ((10e3)/(4.8*1.602e-12))*(7.82e-18)    # photoionisation rate of 2S triplet
    
    t1 = 1                                      # lifetime of 1S
    t3 = 1                                      # lifetime of 2S triplet
    
    a1 = 1.54e-13                               # recombination rate of 1S 
    a3 = 1.49e-13                               # recombination rate of 2S triplet
    
    ## Atomic factors ###
    A31 = 1.272e-4                  # radiative de-excitation rate 2S tri -> 1S
    q13a = 4.5e-20                  # collisional excitation rate 1S -> 2S tri
    q31a = 2.6e-8                   # collisional excitation rate 2S tri -> 2S sing
    q31b = 4.0e-9                   # collisional excitation rate 2S tri -> 2P sing
    Q31 = 5e-10                     # collisional de-exciration rate 2S tri -> 1S
    
    def rhs(l, f, u, ne, n0):
        f1 = f[0]
        f3 = f[1]
        g1 = u**-1 * ((1- f1 - f3) * ne * a1 + f3 * A31 - f1 * F1 * np.exp(-t1) - f1 * ne * q13a + f3* ne * q31a + f3 * ne * q31b + f3 * n0 * Q31)
        g2 = u**-1 * ((1- f1 - f3) * ne * a3 - f3 * A31 - f3 * F3 * np.exp(-t3) + f1 * ne * q13a - f3* ne * q31a - f3 * ne * q31b - f3 * n0 * Q31)
        g = [g1, g2]
        return g
    
    
    ne_1 = 0
    n0_i = 0
    u_i = 0
    
    l0 = [sol_t[0], sol_t[-1]]
    
    l = []
    n1_sol = []
    n3_sol = []
    for r in sol_t:
        i = np.where(sol_t == r)[0].item() # index of r in sol_t (i.e. where it is)
        t = sol_t[i]
        
        ne_i = f_ne(r, t).item()
        n0_i = f_n0(r, t).item()
        u_i = f_u(r, t).item()
        
        sol_f = ivp(rhs, l0, [1, 0], method = 'LSODA', args = (u_i, ne_i, n0_i), t_eval = [r])
        
        l.append(sol_f.t)
        n1_sol.append(sol_f.y[0])
        n3_sol.append(sol_f.y[1])
    
    plt.plot(l, n3_sol)