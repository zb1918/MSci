# -*- coding: utf-8 -*-
"""
reads the files for helium fraction and plots column density

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

plt.style.use("cool-style.mplstyle")

pl_color = 'moccasin'


'''
analysing hydro lines

'''

hydro_sim = loadmat("term1/sims/hyd_sim.mat")

file_y = "term1/sols/hyd_sol_y.p"
file_t = "term1/sols/hyd_sol_t.p"

str_color = 'blue'

#--------------------------- scaling radial distances ------------------------#
rb = hydro_sim['r']             
rb = rb.T[0]                # 1D array of radial distances
rb_sc = rb/rb[0]            # 1D array of radial distances normalised: r[0] = 1
rb_max = rb[-1]             # upper bound for radial distance 
rb_sc_max = rb_sc[-1]       # upper bound for radial distance normalised 

thb = hydro_sim['th']
thb = thb.T[0]              # 1D array of angles

#--------------------------- extracting from .MAT file -----------------------#
vrb = hydro_sim['vr']
vthb = hydro_sim['vth']
u = np.sqrt((vrb**2) + (vthb**2))   # |u|

#-------------- constants in [units] -----------------------------------------#
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

D = hydro_sim['D']
U = hydro_sim['U']
Xe = hydro_sim['ne']
n = D / H_mu                        # number density of Hydrogen
ne = n * Xe                         # electron number density
XH = 1 - Xe                         # fraction of non ionised Hydrogen
n0 = n * XH                         # neutral Hydrogen number density
XHe = 0.3                          # fraction of gas that is Helium
nHe = XHe * D / He_mu               # number density of Helium
mHe = XHe * D                       # mass density of Helium

X = np.outer(rb_sc, np.sin(thb)) # meshgrid of X coordinates
Z = np.outer(rb_sc, np.cos(thb)) # meshgrid of Z coordinates

grid_x = np.linspace(0, 14, 200)
grid_z = np.linspace(-14, 14, 200)

cart_X, cart_Z = np.meshgrid(grid_x, grid_z)

rax, tax = np.meshgrid(rb_sc, thb)

#--------------------- interpolating coarse grid points ----------------------#
f_r = slm.rbs(rb_sc, thb, vrb)
f_t = slm.rbs(rb_sc, thb, vthb)
f_u = slm.rbs(rb_sc, thb, u)
f_ne = slm.rbs(rb_sc, thb, ne)
f_n0 = slm.rbs(rb_sc, thb, n0)
f_nHe = slm.rbs(rb_sc, thb, nHe)


file_f3 = 'term1/output/f3_hyd_8.p'
file_ps = 'term1/output/ps_hyd_8.p'

#-------------- reading file and triangulating+interpolating -----------------#

with open(file_f3, 'rb') as f:
    f3_read = pickle.load(f) 
with open(file_ps, "rb") as f:   
    ps_read = pickle.load(f) 

tri = Delaunay(ps_read)
f_f3 = interpnd(tri, f3_read)

#--------------- cartesian interpolation of helium density -------------------#
pts = []
nHes = []

for r in rb_sc:
    for t in thb:
        pts.append([r * np.sin(t), r * np.cos(t)])
        nHes.append(f_nHe(r, t).item())
        
tri = Delaunay(pts)
f_nHe_cart = interpnd(tri, nHes)

#%%
#----------------------------- column density --------------------------------#
dz = (grid_z[1] - grid_z[0]) * rb[0]
col_d = []
bs = []
for b in grid_x[0:100]:
    if b >= 1:
        bs.append(b)
        d_b = []
        for z in grid_z:
            if f_f3(b, z) > 0:
                d_b.append(f_f3(b, z) * f_nHe_cart(b, z) * dz)
        col_d.append(sum(d_b))
        print(b, ' / ', grid_x[-1])
    
plt.plot(bs, col_d)
plt.yscale('log')
plt.xlim(0, 6)
#%%
# are you not convinced that the interpolated helium density is as good as the 
# original helium density 2D array (nHe) ???????!!?!?!

plt.subplot(1,2,1)
plt.contourf(X, Z, nHe, levels = np.linspace(5e7, 1e9, 100), cmap = "BuPu")
plt.colorbar()
plt.subplot(1,2,2)

plt.contourf(cart_X, cart_Z, f_nHe_cart(cart_X, cart_Z), levels = np.linspace(5e7, 1e9, 100), cmap = "BuPu")
plt.colorbar()
plt.show()
#%%
# f3 contour, change the levels to narrow down where f3 is BIG

plt.contourf(cart_X, cart_Z, f_f3(cart_X, cart_Z), levels = np.linspace(4.85e-7, 5.2e-7, 300), cmap = "BuPu")

#plt.contourf(X, Z, f_f3(X, Z), 200, cmap = "BuPu")
plt.colorbar()

#%%
