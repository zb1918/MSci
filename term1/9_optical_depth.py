"""
imports arrays of streamliens  
solves the helium fractions along each streamline
"""

import numpy as np
import scipy as sc
from matplotlib import pyplot as plt
import stream_solvers as slm
from scipy.io import loadmat
from scipy.interpolate import interp1d as interp
from scipy.interpolate import LinearNDInterpolator as interpnd
from scipy.integrate import solve_ivp as ivp
from scipy.spatial import Delaunay
import itertools
import pickle
import time
import datetime
from scipy import interpolate as inter
from scipy import constants

pl_color = 'moccasin'

'''
analysing hydro lines

'''

hydro_sim = loadmat("term1/sims/hyd_sim.mat")

file_y = "term1/sols/hyd_sol_y.p"
file_t = "term1/sols/hyd_sol_t.p"

str_color = 'blue'

#-------------- constants in cgs----------------------------------------------#
H_molecule = 1.00784                # mass of H in a.m.u.
amu = 1.66054e-24                   # 1 a.m.u. in g
H_mu = H_molecule * amu             # mass of a single H molecule in g
gamma = 5/3                         # 1 + 2/d.o.f. where d.o.f. of H is taken to be 3
kb = 1.38e-16                       # boltzmann constant in cgs
He_molecule = 4.0026                # mass of He in a.m.u.
He_mu = He_molecule * amu           # mass of a single He molecule in g
kb = 1.38e-16                       # boltzmann constant in c.g.s. (cm2 g s-2 K-1)   
c = constants.c                     # speed of light in m s-1
e = constants.e                     # elementary charge in SI
m_e = constants.m_e                 # mass of electron in kg
m_e_cgs = m_e * 1e3                 # mass of electron in g
c_cgs = c * 100                     # speed of light in c.g.s. (cm s-1)

#-------------- range of wavelengths in angstrom -----------------------------#
w_min = 10825
w_max = 10835

#-------------- wavelengths in angstrom --------------------------------------#
w0 = 10829.09114
w1 = 10830.25010               
w2 = 10830.33977
      
#-------------- oscillator strengths (untiless) ------------------------------#
fik0 = 5.9902e-2
fik1 = 1.7974e-1
fik2 = 2.9958e-1

#-------------- gkAki in s-1 -------------------------------------------------#
gA0 = 1.0216e7
gA1 = 3.0648e7
gA2 = 5.1080e7

#----------------------------scaling radial distances-------------------------#
rb = hydro_sim['r']             
rb = rb.T[0]                # 1D array of radial distances
rb_sc = rb/rb[0]            # 1D array of radial distances normalised: r[0] = 1
rb_max = rb[-1]             # upper bound for radial distance 
rb_sc_max = rb_sc[-1]       # upper bound for radial distance normalised 

thb = hydro_sim['th']
thb = thb.T[0]              # 1D array of angles

#----------------------------extracting from .MAT file------------------------#
vrb = hydro_sim['vr']
vthb = hydro_sim['vth']
u = np.sqrt((vrb**2) + (vthb**2))   # |u|

D = hydro_sim['D']
U = hydro_sim['U']
Xe = hydro_sim['ne']
n = D / H_mu                        # number density of Hydrogen
ne = n * Xe                         # electron number density
XH = 1 - Xe                         # fraction of non ionised Hydrogen
n0 = n * XH                         # neutral Hydrogen number density
XHe = 0.25                          # fraction of gas that is Helium
nHe = XHe * D / He_mu               # number density of Helium
mHe = XHe * D                       # mass density of Helium

sig_0 = (np.pi * e**2) / (m_e * c)  # pi e^2 / mc (absorption cross section prefactor) in m^2
sig_0_cgs = sig_0 * 1e3 * 1e3


X = np.outer(rb_sc, np.sin(thb)) # meshgrid of X coordinates
Z = np.outer(rb_sc, np.cos(thb)) # meshgrid of Z coordinates
'''
grid_x = np.multiply(rb_sc, np.sin(thb))
grid_z = np.multiply(rb_sc, np.cos(thb))
'''

grid_x = np.linspace(0, 7, 600)
grid_z = np.linspace(-14, 7, 600)

cart_X, cart_Z = np.meshgrid(grid_x, grid_z)

rax, tax = np.meshgrid(rb_sc, thb)

#----------------------interpolating coarse grid points-----------------------#
f_r = slm.rbs(rb_sc, thb, vrb)
f_t = slm.rbs(rb_sc, thb, vthb)
f_u = slm.rbs(rb_sc, thb, u)
f_ne = slm.rbs(rb_sc, thb, ne)
f_n0 = slm.rbs(rb_sc, thb, n0)
f_nHe = slm.rbs(rb_sc, thb, nHe)

#------------------------calculating temperature------------------------------#
T = U * H_mu * (gamma - 1)/(D * kb)
#T = T.T
f_T = inter.RectBivariateSpline(rb_sc, thb, T, kx=1, ky=1) 

pts = []
nHes = []
Ts = []

#------------------------cartesian interpolation------------------------------#

for r in rb_sc:
    for t in thb:
        pts.append([r * np.sin(t), r * np.cos(t)])
        nHes.append(f_nHe(r, t).item())
        Ts.append(f_T(r, t).item())
        
tri = Delaunay(pts)
f_nHe_cart = interpnd(tri, nHes)
f_T_cart = interpnd(tri, Ts)


F1 = ((10e3)/(24.6*1.602e-12))*(7.82e-18)   # photoionisation rate of 1S
F3 = ((10e3)/(4.8*1.602e-12))*(7.82e-18)    # photoionisation rate of 2S triplet

F1 = ((10e3)/(24.6*1.602e-12))*(7.82e-18)  # photoionisation rate of 1S
F3 = ((10e3)/(4.8*1.602e-12))*(5.48e-18)   # photoionisation rate of 2S triplet

t1 = 0                                      # optical depth of 1S
t3 = 0                                      # optical depth of 2S triplet

#obselete values, a1 and 13 are now found using the corresponding f_'s
a1 = 2.17e-13                               # recombination rate of 1S 
a3 = 1.97e-14                               # recombination rate of 2S triplet

def f_a1(r, t):
    T_i = f_T(r, t, grid = False)
    if T_i <= 1.5e3:
        a1 = 2.17e-13
    else:
        a1 = 1.54e-13
    return a1

def f_a3(r, t):
    T_i = f_T(r, t, grid = False)
    if T_i <= 1.5e3:
        a3 = 1.97e-14
    else:
        a3 = 1.49e-14  
    return a3

## Atomic factors ###
A31 = 1.272e-4                  # radiative de-excitation rate 2S tri -> 1S
q13a = 4.5e-20                  # collisional excitation rate 1S -> 2S tri
q31a = 2.6e-8                   # collisional excitation rate 2S tri -> 2S sing
q31b = 4.0e-9                   # collisional excitation rate 2S tri -> 2P sing
Q31 = 5e-10                     # collisional de-exciration rate 2S tri -> 1S


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

#%%

file_f3 = 'term1/output/f3_hyd_8.p'
file_ps = 'term1/output/ps_hyd_8.p'

with open(file_f3, 'rb') as f:
    f3_read = pickle.load(f) 
with open(file_ps, "rb") as f:   
    ps_read = pickle.load(f) 

    
tri = Delaunay(ps_read)
f_f3 = interpnd(tri, f3_read)

# simo's values:
points = np.load('term1/output/f3_coords.npy')
logf3_values = np.load('term1/output/logf3_values.npy')
tri = Delaunay(points)   # does the triangulation
get_logf3 = interpnd(tri, logf3_values)

#%%
plt.contourf(cart_X, cart_Z, f_f3(cart_X, cart_Z), 200, cmap = "BuPu")

#plt.contourf(X, Z, f_f3(X, Z), 200, cmap = "BuPu")
plt.colorbar()

#%%
osc = [fik0, fik1, fik2]
w_c = [w0, w1, w2]
g_A = [gA0, gA1, gA2]


w_min = 10828
w_max = 10832
ws = np.linspace(w_min, w_max, 200)

dz = grid_z[1] - grid_z[0]

taus = []
for b in grid_x:
    taus_at_b = []
    print(round(b, 1), " / ", grid_x[-1])
    
        
    for w_i in ws:
        tau_at_w = 0
        # at wavelength w_i
        
        for z in grid_z:
            # at point b, z
            n3 = f_nHe_cart(b, z).item()
            f3 = f_f3(b, z).item()
            T = f_T_cart(b, z).item()
            
            for i in range(3):
                # at transition i
                
                oscillator_str = osc[i]
                w_central = w_c[i]
                gA = g_A[i]
                
                phi = slm.voigt(w_i, w_central, gA, T)
                tau_i = phi * oscillator_str * sig_0_cgs * n3 * dz * rb[0] # * sig_lollo because lollo is very sigxy to me
                tau_at_w += tau_i
                
        taus_at_b.append(tau_at_w)
    taus.append(taus_at_b)            
taus = np.array(taus)

#%%
bs = grid_x        
bs_shifted = bs + 0.5 * (bs[1] - bs[0])   
bs_new = []
taus_new = []

for i in range(len(bs_shifted)):
    if bs_shifted[i] > 1:
        bs_new.append(bs_shifted[i])
        taus_new.append(taus[i])
       
taus_l_b = np.array(taus_new).T
flux = []
for tau_l in taus_l_b:
    flux_tot = 0
    tau_tot = 0
    for j in range(len(bs_new)):  
        b = bs_new[j]
        
        if j == 0:
            b_prev = 1
        else:
            b_prev = bs_new[j - 1]

        r2_diff = b**2 - b_prev**2
            
        if tau_l[j] >= 0:
            tau = tau_l[j]
        else:
            tau = 0
        flux_tot += (r2_diff * np.exp(-tau))
    flux.append(flux_tot)        
flux = np.array(flux)

flux_frac = 100* flux / bs_new[-1]**2

y = flux_frac
fig, ax = plt.subplots()
plt.plot(ws, y, color = 'navy')
plt.xlabel("λ (Å)")
plt.ylabel(r"$F_{in}$ / $F_{out}$")
ax.ticklabel_format(style = 'plain', axis = 'x')
plt.xticks([w_min, .5 * (w_min + w_max), w_max])
plt.vlines(w_c[0], min(y), max(y), color = 'red', lw = 0.3)
plt.vlines(w_c[1], min(y), max(y), color = 'red', lw = 0.3)
plt.vlines(w_c[2], min(y), max(y), color = 'red', lw = 0.3)
#plt.xlim(10828.5, 10831)
plt.hlines(max(y), min(ws), max(ws),  color = "navy", ls = 'dotted')