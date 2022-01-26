# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 16:04:00 2022

@author: zaza
"""

import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat
import stream_solvers as slm
from scipy.integrate import solve_ivp as ivp
import scipy as sc


plt.style.use("cool-style.mplstyle")
pl_color = 'blue'

MHD_sim = loadmat("term1/sims/mhd_sim.mat")



rb = MHD_sim['r']
rb = rb.T[0]
rb_sc = rb/abs(rb[0])
rb_sc_max = rb_sc[-1]
thb = MHD_sim['th']
thb = thb.T[0]

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

'''
plt.contourf(X, Z, np.log10(D), 64, cmap = "Reds")
c = plt.colorbar()
c.set_label(r"$\log_{10}$(Density) [g/cm3]")


plt.contourf(X, Z, Br, 64, cmap = "BuPu", levels = [-6, -3, 0, 3, 6])
c = plt.colorbar()
c.set_label(r"$\log_{10}$(Br) [g/cm3]")

'''

fig, ax = plt.subplots()   

r_pl = 1.04
radii = np.linspace(r_pl, rb_sc_max, 500)
rspan = [radii[0], radii[-1]]
#thetas = np.linspace(0, 0.75, 30)*np.pi

f_r = slm.rbs(rb_sc, thb, Br)
f_t = slm.rbs(rb_sc, thb, Bt)

def event(t, y, fr, ft):
    return fr(t, y) 
def event2(t, y, fr, ft):
    return ft(t, y)
event.terminal = True
event.direction = 1


thetas = np.linspace(0, 1, 100)*np.pi
#thetas = np.array([0.8])*np.pi
r_stops = []
t_stops = []

radii = []
for i in range(len(rb_sc)-1):
    subr = np.linspace(rb_sc[i], rb_sc[i+1], 5)
    radii.append(subr[0:4])
radii = np.array(radii)
radii = radii.flatten()
radii = np.array([r for r in radii if r > r_pl])


#radii = radii[0:10]
rspan = [radii[0], radii[-1]]


sols_y = []
sols_t = []
#erroneous initial angle is 1.5549296972313118
thetas = np.array([1.5549296972313118, 1.1])
#thetas = np.array([1])

theta = thetas[1]
t_eval = radii
err = ivp(slm.dydt_rbs, rspan, [theta], t_eval = t_eval,
          args = (f_r, f_t), events = (event), atol = 1e-13, rtol = 1e-8)
if err.status == 1:
    event.direction = -1
    last_ang = err.y_events[0][0][0]
    last_rad = err.t_events[0][0]
    plt.scatter(slm.cart_x(last_rad, last_ang), slm.cart_y(last_rad, last_ang))
    y0 = last_rad
    t_eval = np.linspace(last_ang, last_ang + 0.2, 100)
    tspan = [t_eval[0], t_eval[-1]]
    
    err2 = ivp(slm.dtdy_rbs, tspan, [y0], t_eval = t_eval,
          args = (f_r, f_t), events = (event2), atol = 1e-13, rtol = 1e-8)

slm.plot_cart(err.t, err.y[0])

r = err2.y[0]
theta = err2.t
plt.plot(slm.cart_x(r, theta), slm.cart_y(r, theta))

#slm.plot_cart(-1*err2.y[0], -1*err2.t, color = 'green')
#slm.plot_cart(err2.t, err2.y[0], color = 'red')

planet = plt.Circle((0, 0), 1, color=pl_color)
ax.add_patch(planet)
plt.show()
'''
def f(x, y):
    return x + y

f_Br_tr = slm.rbs(thb, rb_sc, Br)

sol = sc.optimize.root(f, 0.1, args = 0.1, method='hybr')
print(sol.x)

a = f_Br_tr.get_coeffs()

#sol= sc.optimize.root(f_Br_tr, 0.1, args = 0.1, method='hybr')

for r in radii:
    sol = sc.optimize.root(f_r, [0.5], args = r, method='hybr')
    #plt.scatter(sol.x)
'''














    