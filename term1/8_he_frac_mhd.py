# -*- coding: utf-8 -*-
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
m_e = constants.m_e                 # mass of electron in kg
m_e_cgs = m_e * 1e3                 # mass of electron in g
c_cgs = c * 100                     # speed of light in c.g.s. (cm s-1)


n = D / H_mu                        # number density of Hydrogen
ne = n * Xe                         # electron number density
XH = 1 - Xe                         # fraction of non ionised Hydrogen
n0 = n * XH                         # neutral Hydrogen number density
XHe = 0.25                          # fraction of gas that is Helium
nHe = XHe * D / He_mu               # number density of Helium
mHe = XHe * D                       # mass density of Helium

X = np.outer(rb_sc, np.sin(thb)) # meshgrid of X coordinates
Z = np.outer(rb_sc, np.cos(thb)) # meshgrid of Z coordinates

grid_x = np.linspace(0, 14, 200)
grid_z = np.linspace(-14, 14, 200)
cart_X, cart_Z = np.meshgrid(grid_x, grid_z)

#----------------------interpolating coarse grid points-----------------------#
f_r = slm.rbs(rb_sc, thb, vrb)
f_t = slm.rbs(rb_sc, thb, vthb)
f_u = slm.rbs(rb_sc, thb, u)
f_ne = slm.rbs(rb_sc, thb, ne)
f_n0 = slm.rbs(rb_sc, thb, n0)
f_nHe = slm.rbs(rb_sc, thb, nHe)

#f_ne = inter.RectBivariateSpline(rb_sc, thb, ne, kx=1, ky=1)
#f_n0 = inter.RectBivariateSpline(rb_sc, thb, n0, kx=1, ky=1)

# failed attempt to interpolate ne onto cart grid
#f_ne_cart = slm.rbs(grid_x, grid_z, ne)

#------------------------calculating temperature------------------------------#
T = U * H_mu * (gamma - 1)/(D * kb)
#T = T.T
f_T = inter.RectBivariateSpline(rb_sc, thb, T, kx=1, ky=1) 


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


def rhs(l, f, f_r_l, f_t_l):
    f1 = f[0]
    f3 = f[1]
    
    r = f_r_l(l).item()
    t = f_t_l(l).item()
    
    u = f_u(r, t, grid = False).item()
    ne = f_ne(r, t, grid = False).item()
    n0 = f_n0(r, t, grid = False).item()
    
    a1 = f_a1(r, t)
    a3 = f_a3(r, t)
    
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

with open(file_t, 'rb') as ftp:
    sols_t = pickle.load(ftp)
with open(file_y, 'rb') as fyp:
    sols_y = pickle.load(fyp) 
    
rs =    []
ts =    []    
f3s =   []

stream_tot = len(sols_t)
#stream_tot = 1

for stream_no in range(stream_tot): # one streamline at a time
    sol_y = np.array(sols_y[stream_no]) # thetas
    sol_t = np.array(sols_t[stream_no]) # radial distances

    start = time.time()

    #print(sol_t[-1], sol_y[10])
    
    '''
    # plot the streamline investigated:
    fig, ax = plt.subplots()     
    slm.plot_cart(sol_t, sol_y)
    planet = plt.Circle((0, 0), 1, color=pl_color)
    ax.add_patch(planet)
    plt.show()
    
    '''
    #---------generating arc length array (l) along one streamline------------#    
     
    dr = np.diff(sol_t) * rb[0]
    dt = np.diff(sol_y)
    
    # inflow streamlines have point at which dt = 0; fixed with following code
    zero_pt = np.where(dt == 0)[0] # where the zero point actually occurs
    dr = np.delete(dr, zero_pt)
    dt = np.delete(dt, zero_pt)
    sol_t = np.delete(sol_t, zero_pt)
    sol_y = np.delete(sol_y, zero_pt)

    dr_dt = np.divide(dr, dt)
    
    # initial deltas should be zero
    dr = np.insert(dr, 0, 0)
    dt = np.insert(dt, 0, 0)
    dr_dt = np.insert(dr_dt, 0, 0)
    
    # arc length formula - both formulae below work equally well
    dl = np.sqrt(((sol_t * dt)**2) + (dr**2))
    #dl = np.absolute(np.sqrt(sol_t**2 + dr_dt**2)*dt)
    
    l = np.cumsum(dl)   # cumulative sum to generate arc length array
    l0 = np.array([l[0], l[-1]])  # span of integration
    ###########################################################################    


    # interpolate to obtain r and theta (t and y) as functions of arc length
    f_r_l = interp(l, sol_t)
    f_t_l = interp(l, sol_y)    
    
    #^^^^^^^^^^^^^^^^^^^^^making arc length array finer^^^^^^^^^^^^^^^^^^^^^^^#
    arc_l = slm.make_fine(l, 5)
    #vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv#

    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^INTEGRATION^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
    sol_f = ivp(rhs, l0, [1, 1e-10], method = 'LSODA',
                args = (f_r_l, f_t_l), t_eval = arc_l,
                rtol = 1e-12, atol = 1e-10)

    sol_l   = sol_f.t           # arc lengths at which evaluations are made
    sol_f1  = sol_f.y[0]        # He singlet population along streamline
    sol_f3  = sol_f.y[1]        # He triplet population along streamline
    #vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv#
    
    stream_r = []
    stream_t = []
    
    for lx in sol_l:
        stream_r.append(f_r_l(lx).item() * rb[0])
        stream_t.append(f_t_l(lx).item())
        
    rs.append(stream_r)    
    ts.append(stream_t)    
    f3s.append(sol_f3)
    
    plt.plot(stream_r / rb[0], sol_f3, color = str_color)
    plt.xticks([1, 3, 5, 7, 9, 11, 13])    
    
    plt.ylabel("fraction in triplet state")
    plt.xlabel(r"radius [$r_{pl}$]")
    #plt.savefig('term1/images/he_frac_mhd.png')
    plt.show()
    '''
    fig, ax = plt.subplots()     
    slm.plot_cart(stream_r, stream_t)
    planet = plt.Circle((0, 0), 1, color=pl_color)
    ax.add_patch(planet)
    plt.show()
    '''
    end = time.time()
    print(stream_no + 1, '/', len(sols_t), '  ', round(end - start, 2), ' seconds elapsed',
          '\t', 'eta', str(datetime.timedelta(seconds = round((end - start) * (stream_tot - stream_no - 1), 0))))

    
   
flat_rs = slm.flatten(rs)
flat_rs_sc = [r / rb[0] for r in flat_rs]
flat_ts = slm.flatten(ts)
flat_f3s = slm.flatten(f3s)

#%%
file_f3 = 'term1/output/f3_mhd_8.p'
file_ps = 'term1/output/ps_mhd_8.p'


#%%
# combining into coordinates and triangulation + interpolation
pts_polar = [list(pair) for pair in zip(flat_rs_sc, flat_ts)]
pts_carte = [[coord[0]*np.sin(coord[1]), coord[0]*np.cos(coord[1])] for coord in pts_polar]

tri = Delaunay(pts_carte)
f_f3 = interpnd(tri, flat_f3s)

#plt.contourf(cart_X, cart_Z, f_f3(cart_X, cart_Z), 200, cmap = "BuPu")
plt.contourf(X, Z, f_f3(X, Z), 200, cmap = "BuPu")
plt.colorbar()


#%%
'''

# saving the f3s and pts_carte arrays
with open(file_f3, "wb") as f:   
    pickle.dump(flat_f3s, f)
with open(file_ps, "wb") as f:   
    pickle.dump(pts_carte, f)
'''
#%%
# read helium triplet fraction and points files respectively

with open(file_f3, 'rb') as f:
    f3_read = pickle.load(f) 
with open(file_ps, "rb") as f:   
    ps_read = pickle.load(f) 
new_ps = []
new_f3 = []
for p in range(len(ps_read)):
    if (ps_read[p][0]**2 + ps_read[p][1]**2) > 1:
        new_ps.append(ps_read[p])
        new_f3.append(f3_read[p])
        
# interpolating the fractions with the points    
tri = Delaunay(new_ps)
f_f3 = interpnd(tri, new_f3)

#----------------cartesian interpolation of helium density--------------------#

pts = []
nHes = []

for r in rb_sc:
    for t in thb:
        pts.append([r * np.sin(t), r * np.cos(t)])
        nHes.append(f_nHe(r, t).item())
        
tri = Delaunay(pts)
f_nHe_cart = interpnd(tri, nHes)
#%%
plt.subplot(1,2,1)
plt.contourf(X, Z, nHe, levels = np.linspace(5e7, 1e9, 100), cmap = "BuPu")
plt.colorbar()
plt.subplot(1,2,2)

plt.contourf(cart_X, cart_Z, f_nHe_cart(cart_X, cart_Z), levels = np.linspace(5e7, 1e9, 100), cmap = "BuPu")
plt.colorbar()
plt.show()
#%%
plt.contourf(cart_X, cart_Z, f_f3(cart_X, cart_Z), levels = np.linspace(4.85e-7, 5.2e-7, 300), cmap = "BuPu")

#plt.contourf(X, Z, f_f3(X, Z), 200, cmap = "BuPu")
plt.colorbar()

#%%
"""
column density 

"""

dz = (grid_z[1] - grid_z[0]) * rb[0]
col_d = []
bs = []
for b in grid_x:
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