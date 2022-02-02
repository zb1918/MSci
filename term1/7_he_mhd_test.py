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

pl_color = 'moccasin'


#%%

'''
analysing hydro lines

'''

hydro_sim = loadmat("term1/sims/hyd_sim.mat")

file_y = "term1/sols/hyd_sol_y.p"
file_t = "term1/sols/hyd_sol_t.p"

str_color = 'blue'
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
n = D / (1.00784 * 1.66e-24)        # number density
ne = n * Xe                         # electron number density
Xh = 1 - Xe                         # fraction of non ionised hydrogen
n0 = n * Xh                         # neutral hydrogen number density

X = np.outer(rb_sc, np.sin(thb)) # meshgrid of X coordinates
Z = np.outer(rb_sc, np.cos(thb)) # meshgrid of Z coordinates
'''
grid_x = np.multiply(rb_sc, np.sin(thb))
grid_z = np.multiply(rb_sc, np.cos(thb))
'''

grid_x = np.linspace(0, 14, 128)
grid_z = np.linspace(-14, 14, 128)

cart_X, cart_Z = np.meshgrid(grid_x, grid_z)

rax, tax = np.meshgrid(rb_sc, thb)

#----------------------interpolating coarse grid points-----------------------#
f_r = slm.rbs(rb_sc, thb, vrb)
f_t = slm.rbs(rb_sc, thb, vthb)
f_u = slm.rbs(rb_sc, thb, u)
f_ne = slm.rbs(rb_sc, thb, ne)
f_n0 = slm.rbs(rb_sc, thb, n0)

# failed attempt to interpolate ne onto cart grid
#f_ne_cart = slm.rbs(grid_x, grid_z, ne)

#%%

'''
analysing mhd lines

'''

mhd_sim = loadmat("term1/sims/mhd_sim.mat")

file_y = "term1/sols/mhd_sol_y.p"
file_t = "term1/sols/mhd_sol_t.p"
  
str_color = 'red'

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
n = D / (1.00784 * 1.66e-24)        # number density

U = mhd_sim['U']                    # internal energy pu volume

Xe = mhd_sim['Xe']                  # electron fraction ie fraction of ionised hydrogen
ne = n * Xe                         # electron number density
Xh = 1 - Xe                         # fraction of non ionised hydrogen
n0 = n * Xh                         # neutral hydrogen number density

#----------------------interpolating coarse grid points-----------------------#
f_r = slm.rbs(rb_sc, thb, vrb)
f_t = slm.rbs(rb_sc, thb, vthb)
f_Br = slm.rbs(rb_sc, thb, Br)
f_Bt = slm.rbs(rb_sc, thb, Bt)
f_u = slm.rbs(rb_sc, thb, u)
f_ne = slm.rbs(rb_sc, thb, ne)
f_n0 = slm.rbs(rb_sc, thb, n0)

#%%

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
def rhs(l, f, u, ne, n0):
    f1 = f[0]
    f3 = f[1]
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
    
rs = []
ts = []    
f3s = []

stream_tot = 500
for stream_no in range(stream_tot): # one streamline at a time
#for stream_no in range(0, 1):
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
    

    
    # arc length formula
    dl = np.absolute(np.sqrt(sol_t**2 + dr_dt**2)*dt)
    l = np.cumsum(dl) # cumulative sum to generate arc length array
    ###########################################################################    

    
    # interpolate to obtain r and theta (t and y) as functions of arc length
    f_r = interp(l, sol_t)
    f_t = interp(l, sol_y)
    
    # reset values
    ne_1 = 0
    n0_i = 0
    u_i = 0
    # span of integration (from beginning to end of streamline)
    l0 = [l[0], l[-1]]
    
    sol_l = []  # arc lengths of streamline at which n1 and n3 determined
    sol_n1 = [] # He singlet population along streamline
    sol_n3 = [] # He triplet population along streamline
    
    #---------------------making arc length array finer-----------------------#
    # 
    arc_l = []
    for i in range(len(l) - 1):
    #for i in range(200): # cutting at arbitrary length
        subl = np.linspace(l[i], l[i + 1], 5) # last parameter = fineness
        arc_l.append(subl[0:4])
    arc_l = np.array(arc_l)
    arc_l = arc_l.flatten()
    ###########################################################################
    
    stream_r = []
    stream_t = []
    for li in arc_l: # at each point along length of streamline
        r = f_r(li).item()
        t = f_t(li).item()
        
        stream_r.append(r)
        stream_t.append(t)

        ne_i = f_ne(r, t).item()
        n0_i = f_n0(r, t).item()
        u_i = np.absolute(f_u(r, t).item())
        
        sol_f = ivp(rhs, l0, [1, 0], method = 'LSODA',
                    args = (u_i, ne_i, n0_i), t_eval = [li],
                    rtol = 1e-10, atol = 1e-7)

        sol_l.append(sol_f.t)
        sol_n1.append(sol_f.y[0])
        sol_n3.append(sol_f.y[1])
        f3s.append(sol_f.y[1].item())
    rs.append(stream_r)
    ts.append(stream_t)
    
    plt.plot(stream_r, sol_n3, color = str_color)
    plt.ylabel("fraction in triplet state")
    plt.xlabel("radial distane")
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
          '  ', 'eta', str(datetime.timedelta(seconds = round((end - start) * (stream_tot - stream_no - 1), 0))))

    
#rs = np.array(rs).flatten()
#ts = np.array(ts).flatten()    
rs = slm.flatten(rs)
ts = slm.flatten(ts)
#%%
# combining into coordinates and triangulation + interpolation
pts_polar = [list(pair) for pair in zip(rs, ts)]
pts_carte = [[coord[0]*np.sin(coord[1]), coord[0]*np.cos(coord[1])] for coord in pts_polar]

tri = Delaunay(pts_carte)
f_f3 = interpnd(tri, f3s)

plt.contourf(cart_X, cart_Z, f_f3(cart_X, cart_Z), 200, cmap = "BuPu")
plt.colorbar()

#%%
# file locations
file_f3 = 'term1/output/f3_hyd.p'
file_ps = 'term1/output/ps_hyd.p'

#%%
'''
# saving the f3s and pts_carte arrays
with open(file_f3, "wb") as f:   
    pickle.dump(f3s, f)
with open(file_ps, "wb") as f:   
    pickle.dump(pts_carte, f)
'''
#%%
# reading and displaying f3s and pts_carte array files
with open(file_f3, 'rb') as f:
    f3_read = pickle.load(f) 
with open(file_ps, "rb") as f:   
    ps_read = pickle.load(f) 

ps_read.append([0, 1])
f3_read.append(0)

ps_read.append([1, 0])
f3_read.append(0)

    
tri = Delaunay(ps_read)
f_f3 = interpnd(tri, f3_read)


short_cart_X = [x[70:80] for x in cart_X]
short_cart_Z = [z[70:80] for z in cart_Z]

plt.contourf(cart_X, cart_Z, f_f3(cart_X, cart_Z), 200, cmap = "BuPu")
#plt.contourf(short_cart_X, short_cart_Z, f_f3(short_cart_X, short_cart_Z), 200, cmap = "BuPu")
plt.colorbar()


#%%
t_is = []
ne_trips = []

dx = (grid_x[1] - grid_x[0]) * rb[0]


for i in range(len(grid_x)):
    ne_trip = []
    x = grid_x[i]
    for j in range(len(grid_z)):
        z = grid_z[j]
        
        r_i = (z**2 + x**2)**0.5
        t_i = np.arctan(x/z)
        
        ne_i = f_ne(r_i, t_i).item()
        f3_i = f_f3(x, z).item()
        ne_trip.append(f3_i * ne_i * dx)
        
        #t_is.append(f3_i)
    ne_trip = [ne for ne in ne_trip if ne > 0]
    ne_trips.append(np.sum(ne_trip))

plt.plot(grid_x, np.log10(ne_trips))

