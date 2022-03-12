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
from scipy import constants
from scipy import interpolate as inter

pl_color = 'moccasin'


'''
analysing mhd lines

'''

mhd_sim = loadmat("term1/sims/mhd_sim.mat")

file_y = "term1/sols/mhd_sol_y.p"
file_t = "term1/sols/mhd_sol_t.p"

str_color = 'red'

#----------------------------scaling radial distances-------------------------#
rb = mhd_sim['r']                   # 1D array of radius for grid
thb = mhd_sim['th']                 # 1D array of theta for grid
rb = rb.T[0]
rb_sc = rb/abs(rb[0])
rb_sc_max = rb_sc[-1]
thb = thb.T[0] 

X = np.outer(rb_sc, np.sin(thb))
Z = np.outer(rb_sc, np.cos(thb))
rax, tax = np.meshgrid(rb_sc, thb)

Br = mhd_sim['B1']
Bt = mhd_sim['B2']
Bc = mhd_sim['B3']

vrb = mhd_sim['v1']                 # u_r
vthb = mhd_sim['v2']                # u_theta
vc = mhd_sim['v3']
u = np.sqrt((vrb**2) + (vthb**2))   # |u|

D = mhd_sim['rho']                  # mass density
Xe = mhd_sim['Xe']                  # fraction of ionised H
U = mhd_sim['U']                    # internal energy pu volume


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
e_cgs = 4.803204251e-10             # elementary charge in c.g.s.
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

n = D / H_mu                        # number density of Hydrogen
ne = n * Xe                         # electron number density
XH = 1 - Xe                         # fraction of non ionised Hydrogen
n0 = n * XH                         # neutral Hydrogen number density
XHe = 0.25                          # fraction of gas that is Helium
nHe = XHe * D / He_mu               # number density of Helium
mHe = XHe * D                       # mass density of Helium

#sig_0 = (np.pi * e**2) / (m_e * c)  # pi e^2 / mc (absorption cross section prefactor) in m^2
#sig_0_cgs = sig_0 * 1e3 * 1e3
sig_0_cgs = (np.pi * e_cgs**2) / (m_e_cgs * c_cgs)

X = np.outer(rb_sc, np.sin(thb)) # meshgrid of X coordinates
Z = np.outer(rb_sc, np.cos(thb)) # meshgrid of Z coordinates
'''
grid_x = np.multiply(rb_sc, np.sin(thb))
grid_z = np.multiply(rb_sc, np.cos(thb))
'''


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



#------------------------cartesian interpolation------------------------------#
pts = []
nHes = []
Ts = []

for r in rb_sc:
    for t in thb:
        pts.append([r * np.sin(t), r * np.cos(t)])
        nHes.append(f_nHe(r, t, grid = False))
        Ts.append(f_T(r, t, grid = False))
        
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


file_f3 = 'term1/output/f3_mhd_8.p'
file_ps = 'term1/output/ps_mhd_8.p'

with open(file_f3, 'rb') as f:
    f3_read = pickle.load(f) 
with open(file_ps, "rb") as f:   
    ps_read = pickle.load(f) 

    
tri = Delaunay(ps_read)
f_f3 = interpnd(tri, f3_read)

#%%
grid_x = np.linspace(1.04, 2, 500)
grid_x = np.append(grid_x[0:-1], np.linspace(2, 6, 300))

grid_z = np.linspace(-6, 6, 400)

cart_X, cart_Z = np.meshgrid(grid_x, grid_z)

nHe_cart = f_nHe_cart(cart_X, cart_Z)
T_cart = f_T_cart(cart_X, cart_Z)
f3_cart = f_f3(cart_X, cart_Z)
n3_cart = nHe_cart * f3_cart

w_min = 10828
w_max = 10832
ws = np.linspace(w_min, w_max, 100)

dz = grid_z[1] - grid_z[0]
tau_bs = []
for w in ws:
    start = time.time()
    phi_0 = slm.voigt(w, w0, gA0, T_cart) * fik0
    phi_1 = slm.voigt(w, w1, gA1, T_cart) * fik1
    phi_2 = slm.voigt(w, w2, gA2, T_cart) * fik2
    phi = phi_0 + phi_1 + phi_2
    tau = phi * n3_cart * sig_0_cgs * dz * rb[0]
    
    tau_nan = [[c if c > 0 else 0 for c in row] for row in tau]
        
    tau_b = np.sum(tau_nan, 0)
    
    tau_bs.append(tau_b)
    end = time.time()
    
    index = np.where(ws == w)[0][0]
    if index % 5 == 0:   
        todo = len(ws) - index
        eta = todo * (end - start)
        print(index, '\t / \t', len(ws), '\t', 'eta', str(datetime.timedelta(seconds = round(eta, 0))))
    
tau_bs = np.array(tau_bs)


tau_ws = tau_bs.T

bs = grid_x        
bs_new = []
taus_new = []

for i in range(len(tau_ws)):
    if bs[i] > 1:
        bs_new.append(bs[i])
        taus_new.append(tau_ws[i])
        

taus_new = np.array(taus_new)
abs_frac = np.exp(-taus_new)

b_del = (bs_new[1] - bs_new[0]) * 0.5
bs_shifted = bs_new + b_del
bs_shifted_down = np.array([1])
bs_shifted_down = np.append(bs_shifted_down, bs_shifted[0:-1])
b_diff = bs_shifted**2 - bs_shifted_down**2
flux = abs_frac * b_diff[:, None]
   
flux = np.array(flux)   
flux = np.sum(flux, 0) 
flux_frac = 100 * flux / bs_shifted[-1]**2   
#%%
# tau as a function of impact parameter

plt.ylabel(r"absorption coefficient $\tau$ ")
plt.xlabel(r"impact parameter $b$ [$r_{pl}$]")
tau_b_tot = np.sum(taus_new.T, 0)
plt.plot(bs_new, tau_b_tot)
plt.hlines(1, min(bs_new), max(bs_new), color = "red", lw = 0.5)
plt.ylim(0, 10)
plt.show()

#%%

w_c = [w0, w1, w2]

y = flux_frac
fig, ax = plt.subplots()
plt.plot(ws, y, color = 'navy')
plt.xlabel("λ (Å)")
plt.ylabel(r"$F_{in}$ / $F_{out}$  (%)")
ax.ticklabel_format(style = 'plain', axis = 'x')
plt.xticks([w_min, .5 * (w_min + w_max), w_max])
plt.vlines(w_c[0], min(y), max(y), color = 'red', lw = 0.3)
plt.vlines(w_c[1], min(y), max(y), color = 'red', lw = 0.3)
plt.vlines(w_c[2], min(y), max(y), color = 'red', lw = 0.3)
#plt.xlim(10828.5, 10831)
plt.hlines(max(y), min(ws), max(ws),  color = "navy", ls = 'dotted')
