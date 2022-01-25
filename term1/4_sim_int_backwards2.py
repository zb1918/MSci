"""
simulation streamlines 
include event detection (i.e. when the integration goes back into planet)
and corrects during event detection such that intial conditions are flipped
(radii and rspan)
"""

import os
import os.path
import numpy as np
from scipy.integrate import solve_ivp as ivp
import matplotlib.pyplot as plt
import stream_solvers as slm
from scipy.io import loadmat
from scipy.interpolate import interp1d as interp

plt.style.use("cool-style.mplstyle")
pl_color = 'moccasin'

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

'''
###############################################################################
#--contour plot of radial velocity with negative and positive distinguished---#
plt.contourf(X, Z, vrb, 64, cmap = "BuPu", levels = [-2e8, 0, 2e8])
c = plt.colorbar()
c.set_label("velocity")
###############################################################################
'''

#----------------------interpolating coarse grid points-----------------------#
f_r = slm.rbs(rb_sc, thb, vrb)
f_t = slm.rbs(rb_sc, thb, vthb)


def event(t, y, fr, ft):
    return fr(t, y)

event.terminal = True


thetas = np.linspace(0, 0.5, 50)*np.pi
#thetas = np.array([0.8])*np.pi
r_stops = []
t_stops = []

r_pl = 1.04
radii = []
for i in range(len(rb_sc)-1):
    subr = np.linspace(rb_sc[i], rb_sc[i+1], 5)
    radii.append(subr[0:4])
radii = np.array(radii)
radii = radii.flatten()
radii = np.array([r for r in radii if r > r_pl])


fig, ax = plt.subplots()     

sols_t = []
sols_y = []

if os.path.exists('term1/sols/hyd_sol_t.npy'):
    os.remove('term1/sols/hyd_sol_t.npy')
    
if os.path.exists('term1/sols/hyd_sol_y.npy'):
    os.remove('term1/sols/hyd_sol_y.npy')


for theta in thetas:
    sol_y = np.array([])
    sol_t = np.array([])
    num_events = 0
    
    t_eval = radii.flatten()           
    rspan = [t_eval[0], t_eval[-1]]
    event.direction = np.array(f_r(t_eval[0], theta) / abs(f_r(t_eval[0], theta))) * -1
    event.direction = event.direction.item()
    if len(sol_y) > 0:
        t_eval = [r for r in radii if r * event.direction > sol_t[-1] * event.direction]
    t_eval = t_eval[::-1 * int(event.direction)]
    rspan = [t_eval[0], t_eval[-1]]
    
    
    sol = ivp(slm.dydt_rbs, rspan, [theta], t_eval = t_eval,
              args = (f_r, f_t), events = (event), atol = 1e-12, rtol = 1e-8)
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
                  args = (f_r, f_t), events = (event), atol = 1e-12, rtol = 1e-8)
        sol_y = np.append(sol_y, sol.y[0])
        sol_t = np.append(sol_t, sol.t)
        
        ''' '''
        
        sol_y = sol_y.flatten()
        sol_t = sol_t.flatten()      
    sols_t.append(list(sol_t))    
    sols_y.append(list(sol_y))
    slm.plot_cart(sol_t, sol_y, color = "blue", lw = 2)
    plt.scatter(slm.cart_x(r_pl, theta), slm.cart_y(r_pl, theta))
    
sols_t = np.array(sols_t)    
sols_y = np.array(sols_y)    

file_t = 'term1/sols/hyd_sol_t.npy'
file_y = 'term1/sols/hyd_sol_y.npy'

np.save(file_t, sols_t, allow_pickle = True)
np.save(file_y, sols_y, allow_pickle = True)



planet = plt.Circle((0, 0), 1, color=pl_color)
ax.add_patch(planet)
plt.show()


