# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 14:12:07 2021

@author: zaza
"""

import numpy as np
import scipy as sc
from matplotlib import pyplot as plt
from scipy.io import loadmat
import stream_solvers as slm
from scipy.integrate import solve_ivp as ivp

plt.style.use("cool-style.mplstyle")
pl_color = 'blue'

MHD_sim = loadmat("term1/mhd_sim.mat")


rb = MHD_sim['r']
rb = rb.T
rb_sc = rb/abs(rb[0])
rb_sc_max = rb_sc[-1]
thb = MHD_sim['th']
thb = thb.T

vrb = MHD_sim['v1']
vthb = MHD_sim['v2']
D = MHD_sim['rho']
Br = MHD_sim['B1']
Bt = MHD_sim['B2']
Bc = MHD_sim['B3']

'''
X = np.outer(rb,np.sin(thb))
Z = np.outer(rb,np.cos(thb))
'''

X = np.outer(rb_sc,np.sin(thb))
Z = np.outer(rb_sc,np.cos(thb))


## Grid is setup so the z-axis points to the star

plt.contourf(X, Z, np.log10(D), 64, cmap = "Reds")
c = plt.colorbar()
c.set_label(r"$\log_{10}$(Density) [g/cm3]")


#%%

plt.contourf(X, Z, Br, 64, cmap = "BuPu")
c = plt.colorbar()
c.set_label(r"$\log_{10}$(Br) [g/cm3]")

#%%
fig, ax = plt.subplots()     

radii = np.linspace(1.04, rb_sc_max, 500)
rspan = [radii[0], radii[-1]]
thetas = np.linspace(0, 0.75, 10)*np.pi

f_r = slm.rbs(rb_sc, thb, Br)
f_t = slm.rbs(rb_sc, thb, Bt)

interp_sol = ivp(slm.dydt_rbs, rspan, thetas, t_eval = radii, args = (f_r, f_t))

slm.plot_mult(interp_sol.t, interp_sol.y)

planet = plt.Circle((0, 0), 1, color=pl_color)
ax.add_patch(planet)

#plt.savefig("images/velocity.pdf", format="pdf")
plt.show()
