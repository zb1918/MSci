"""
casts onto coarse grid and interpolates functions in
stream_solvers.u_r and stream_solvers.u_theta

"""

import numpy as np
from scipy.integrate import solve_ivp as ivp
import matplotlib.pyplot as plt
import stream_solvers as slm
from matplotlib.widgets import Slider


plt.style.use("cool-style.mplstyle")
#plt.subplots(1,1, sharex = True, sharey = True)

#ax1 = plt.subplot(1,1,1)

def u_r(r, theta, a=1):
    return r**-0.5
def u_theta(r, theta, a =1):
    return np.cos(theta)


thetas = np.linspace(0, 2, 80)*np.pi #initial conditions to cover 2pi radians
thetas = thetas[0:-1]

#----------------------setting the radius limits------------------------------#
r_lim = 30
radii = slm.rad(r_lim, 100)(0)
rspan = np.array([radii[0], radii[-1]])

#----------------------coarse grid for interpolation--------------------------#
#can adjust coarseness to evaluate accuracy of interpolation
coarse_r = slm.rad(r_lim, 100)(0)
coarse_t = np.linspace(0, 3, 80)*np.pi

#----------------------interpolating coarse grid points-----------------------#
u_rad_cast = slm.cast(coarse_r, coarse_t, u_r)
u_the_cast = slm.cast(coarse_r, coarse_t, u_theta)
f_r = slm.rbs(coarse_r, coarse_t, u_rad_cast)
f_t = slm.rbs(coarse_r, coarse_t, u_the_cast)

interp_sol = ivp(slm.dydt_rbs, rspan, thetas, t_eval = radii, args = (f_r, f_t))

"""
for ys in interp_sol.y:
    slm.plot_cart(interp_sol.t, ys, "blue", lw = 1.5)

size = r_lim
plt.ylim(-size,size)
plt.xlim(-size,size)
plt.title("interpolating set of points")
"""

#%%
"""
interpolation with different grid/resolution for comparison

"""


#----------------------coarse grid for interpolation--------------------------#
#can adjust coarseness to evaluate accuracy of interpolation
coarse_r = rad(r_lim, 10)
coarse_t = np.linspace(0, 2, 10)*np.pi

#----------------------interpolating coarse grid points-----------------------#
u_rad_cast = slm.cast(coarse_r, coarse_t, u_r)
u_the_cast = slm.cast(coarse_r, coarse_t, u_theta)
f_r = slm.rbs(coarse_r, coarse_t, u_rad_cast)
f_t = slm.rbs(coarse_r, coarse_t, u_the_cast)

interp_sol2 = ivp(slm.dydt_rbs, rspan, thetas, t_eval = radii, args = (f_r, f_t))

"""
for ys in interp_sol2.y:
    slm.plot_cart(interp_sol2.t, ys, "red", lw = 1.5)

size = r_lim
plt.ylim(-size,size)
plt.xlim(-size,size)
plt.title("interpolating set of points")
#plt.show()
"""
#%%
"""
solving streamlines for functions in 
streamlines.u_r and streamlines.u_theta directly

slm.u_r

"""

#ax2 = plt.subplot(1,1,1, sharex = ax1, sharey = ax1)

direct_sol = ivp(slm.dydt, rspan, thetas, t_eval = radii,
                 args = (u_r, u_theta)
                 )


"""
for ys in direct_sol.y:
    slm.plot_cart(direct_sol.t, ys, "orange", lw = 1)
                   
size = r_lim
plt.ylim(-size,size)
plt.xlim(-size,size)
plt.title("solving ODE for velocity functions")
#plt.show()
"""
#%%

#slm.plot_mult(direct_sol.t, direct_sol.y, color = "red", lw = 2)
slm.plot_mult(interp_sol.t, interp_sol.y, color = "blue", lw = 0.6)

#test

"""
for r in np.linspace(0,20,10):
    for t in (np.linspace(1.5, 2, 10)*np.pi):
        plt.scatter(r, t)
        intr = f_r(r,t, grid=False)
        intt = f_t(r,t, grid=False)
        rear = u_r(r,t)
        reat = u_theta(r,t)
        perr = (intr - rear)/rear
        pert = (intt - reat)/reat
        
        plt.text(r, t, str(perr) + "\n" + str(pert))
"""


#128 x 256
#will probably have to interpolate in log space instead for accuracy
#hydro field (nomag)

#plot r,theta plots for ur and utheta
#can downgeade to linear interpolation and compare to cubic
#can even go higherr (5th order spline/4th)
#is thesimulatiom resolution appropriate for the problem?
#

