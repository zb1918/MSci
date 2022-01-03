import numpy as np
from scipy.integrate import solve_ivp as ivp
import matplotlib.pyplot as plt
import stream_solvers as slm
from scipy.io import loadmat
from scipy.interpolate import interp1d as interp

plt.style.use("cool-style.mplstyle")
pl_color = 'slategrey'
#%%

"""
extraction of data and physical calculations
"""
hydro_sim = loadmat("term1/sims/pure_HD.mat") #extract pure_HD simulation

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
D = hydro_sim['D']
U = hydro_sim['U']
ne = hydro_sim['ne']


X = np.outer(rb_sc,np.sin(thb)) # meshgrid of X coordinates
Z = np.outer(rb_sc,np.cos(thb)) # meshgrid of Z coordinates

#---------------------------constants in cgs----------------------------------#
H_molecule = 1.00784        # mass of H in a.m.u.
amu = 1.66e-24              # 1 a.m.u. in g
H_mu = H_molecule * amu     # mass of a single H molecule in g
gamma = 5/3                 # 1 + 2/d.o.f. where d.o.f. of H is taken to be 3
kb = 1.38e-16               # boltzmann constant in cgs

#------------------------calculating temperature------------------------------#
T = U * H_mu * (gamma - 1)/(D * kb)
T = T.T
radii = np.linspace(1, rb_sc_max, 128)

#%%

#relevant equations
# P = (gamma -1)e
# P = k/mu rho T

# u = internal energy/volume
# tau = optical depth
# ne = fraction of ionised hydrogen
# d = mass density



"""
task for next week:
plot log T as function of log(height) and investigate points just before the slope begins
determine point at which radius will begin
"""

#%%
"""
streamlines for interpolated velocity or log(velocity) matrices
correction for valid streamlines included
radial distances scaled

"""

fig, ax = plt.subplots()     


#----------------------fine grid for post-interpolation-----------------------#
radii = np.linspace(1.04, rb_sc_max, 1)
rspan = [radii[0], radii[-1]]
thetas = np.linspace(0, 0.1, 100)*np.pi
tspan = [thetas[0], thetas[-1]]

#----------------------interpolating coarse grid points-----------------------#
f_r = slm.rbs(rb_sc, thb, vrb)
f_t = slm.rbs(rb_sc, thb, vthb)

interp_sol = ivp(slm.dtdy_rbs, tspan, radii, t_eval = thetas, args = (f_r, f_t))
# to do: replace t_eval with some form of rtol or atol

for rads in interp_sol.y:
    slm.plot_cart(rads, interp_sol.t)
    
planet = plt.Circle((0, 0), 1, color=pl_color)
ax.add_patch(planet)

#plt.savefig("images/velocity.pdf", format="pdf")
plt.show()

#%%
"""
contour plot of temperature on cartesian grid

"""

fig, ax = plt.subplots()
plt.contourf(X, Z, vthb, 64, cmap = "BuPu", levels = [-5e6, 0, 5e6])
c = plt.colorbar()
c.set_label("velocity")

planet = plt.Circle((0, 0), rb_sc[0], color=pl_color)
ax.add_patch(planet)
plt.show()
