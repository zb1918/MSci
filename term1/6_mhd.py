# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 14:12:07 2021

@author: zaza
"""
import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat
import stream_solvers as slm
from scipy.integrate import solve_ivp as ivp


plt.style.use("cool-style.mplstyle")
pl_color = 'blue'

MHD_sim = loadmat("term1/mhd_sim.mat")

if os.path.exists('term1/sols/mhd_sol_t.npy'):
    os.remove('term1/sols/mhd_sol_t.npy')
    
if os.path.exists('term1/sols/mhd_sol_y.npy'):
    os.remove('term1/sols/mhd_sol_y.npy')


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

plt.contourf(X, Z, np.log10(D), 64, cmap = "Reds")
c = plt.colorbar()
c.set_label(r"$\log_{10}$(Density) [g/cm3]")


#%%

plt.contourf(X, Z, Br, 64, cmap = "BuPu", levels = [-6, -3, 0, 3, 6])
c = plt.colorbar()
c.set_label(r"$\log_{10}$(Br) [g/cm3]")

#%%
fig, ax = plt.subplots()   
  
r_pl = 1.04
radii = np.linspace(r_pl, rb_sc_max, 500)
rspan = [radii[0], radii[-1]]
#thetas = np.linspace(0, 0.75, 30)*np.pi

f_r = slm.rbs(rb_sc, thb, Br)
f_t = slm.rbs(rb_sc, thb, Bt)

def event(t, y, fr, ft):
    return fr(t, y) 

event.terminal = True


thetas = np.linspace(0, 0.5, 100)*np.pi
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


sols_y = []
sols_t = []
#thetas = np.array([1.5549296972313118])
for theta in thetas:
    print(theta)

    sol_y = np.array([])
    sol_t = np.array([])
    num_events = 0
    
    t_eval = radii.flatten()           
    rspan = [t_eval[0], t_eval[-1]]
    event.direction = np.array(f_r(t_eval[0], theta) / abs(f_r(t_eval[0], theta))) * -1
    event.direction = event.direction.item()
    
    print(event.direction)

    if len(sol_y) > 0:
        print("what is the point of this test")
        t_eval = [r for r in radii if r * event.direction > sol_t[-1] * event.direction]
    t_eval = t_eval[::-1 * int(event.direction)]
    rspan = [t_eval[0], t_eval[-1]]
    print(rspan)
    
    #intital solution until the event is triggered (negative radial velocity)
    sol = ivp(slm.dydt_rbs, rspan, [theta], t_eval = t_eval,
              args = (f_r, f_t), events = (event), atol = 1e-12, rtol = 1e-6)
    
    sol_y = np.append(sol_y, sol.y[0])
    sol_t = np.append(sol_t, sol.t)
    sol_y = sol_y.flatten()
    sol_t = sol_t.flatten()
      
    while sol.status != 0:
        num_events +=1
        last_y = np.array(sol.y_events).item()
        last_t = np.array(sol.t_events).item()
        sol_y = np.append(sol_y, last_y)
        sol_t = np.append(sol_t, last_t)
        
        if len(sol_y) > 0:
            t_eval = [r for r in radii if r * event.direction > last_t * event.direction]
        event.direction *= -1
        t_eval = t_eval[::-1 * int(event.direction)]
        rspan = [t_eval[0], t_eval[-1]]

        sol = ivp(slm.dydt_rbs, rspan, [last_y], t_eval = t_eval,
                  args = (f_r, f_t), events = (event)
                  , atol = 1e-12, rtol = 1e-6
                  )
        sol_y = np.append(sol_y, sol.y[0])
        sol_t = np.append(sol_t, sol.t)
        
        ''' '''
        
        sol_y = sol_y.flatten()
        sol_t = sol_t.flatten()      
    
    if sol_t[-1] != sol_t[0]: # if the streamline doesnt 'close'
    #if 1 == 1:
        slm.plot_cart(sol_t, sol_y, color = "blue", lw = 2)
        plt.scatter(slm.cart_x(r_pl, theta), slm.cart_y(r_pl, theta))
        
        sols_t.append(sol_t)
        sols_y.append(sol_y)
        
        
print('success')
#%%
sols_t = np.array(sols_t)    
sols_y = np.array(sols_y)    

file_t = 'term1/sols/mhd_sol_t.npy'
file_y = 'term1/sols/mhd_sol_y.npy'

np.save(file_t, sols_t, allow_pickle = True)
np.save(file_y, sols_y, allow_pickle = True)


plt.show()

                                                                                                                               
planet = plt.Circle((0, 0), 1, color=pl_color)
ax.add_patch(planet)

#plt.savefig("images/velocity.pdf", format="pdf")
plt.show()
