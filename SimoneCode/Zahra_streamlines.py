import os
import numpy as np
from scipy.integrate import solve_ivp as ivp
import matplotlib.pyplot as plt
import stream_solvers as slm
from scipy.io import loadmat
from scipy.interpolate import interp1d as interp

plt.style.use("cool-style.mplstyle")

#%%
hydro_sim = loadmat("sims/pure_HD.mat")

rb = hydro_sim['r']
rb = rb/rb[0]

thb = hydro_sim['th']


rb = rb.T
rb = rb[0]


vrb = hydro_sim['vr']
vthb = hydro_sim['vth']

D = hydro_sim['D']
U = hydro_sim['U']
ne = hydro_sim['ne']

X = np.outer(rb,np.sin(thb))
Z = np.outer(rb,np.cos(thb))

'''
plt.contourf(X,Z,vrb, 64, cmap = "BuPu")
c = plt.colorbar()
c.set_label(r"\log_{10}(Density) [g/cm3]")
'''

'''
plt.contourf(X,Z,vrb, 64, cmap = "BuPu", #
             levels = [-3e6, -2e6, -1e6, 0]
             )
c = plt.colorbar()
c.set_label("(vrb)")

'''


r_lim = rb[-1]
radii = np.linspace(1.02, r_lim, 500)
rspan = [radii[0], radii[-1]]
#----------------------coarse grid for interpolation--------------------------#
#can adjust coarseness to evaluate accuracy of interpolation
thetas = np.linspace(0, 1, 30)*np.pi
#thetas = np.array([0.25])*np.pi

#----------------------interpolating coarse grid points-----------------------#
f_r = slm.rbs(rb, thb, vrb)
f_t = slm.rbs(rb, thb, vthb)

interp_sol = ivp(slm.dydt_rbs, rspan, thetas, t_eval = radii, args = (f_r, f_t))
slm.plot_mult(interp_sol.t, interp_sol.y, color = "blue", lw = 0.6)

#%%


# P = (gamma -1)e
# P = k/mu rho T

#u = internal energy/volume
#tau = optical depth
#ne = fraction of ionised hydrogen
#d = mass density
#


"""
plot log T as function of log(height) and investigate points just before the slope begins
determine point at which radius will begin
"""
#%%

mu = 1.00784 * 1.66e-24
gamma = 5/3
kb = 1.38e-16

#%%

T = U* mu * (gamma - 1)/(D * kb)
T = T.T
radii = np.linspace(1, r_lim, 128)
"""
for i in range(63):
    plt.plot(rb, np.log(T[i]), color = "red")
for i in range(64, 127):
    plt.plot(rb, np.log(T[i]), color = "blue")

"""
plt.plot(rb, np.log(T[0]), color = "navy")
#f_T = interp(rb, np.log(T[0]), kind = "cubic")
#plt.plot(radii, f_T(radii))



