"""
casts onto coarse grid and interpolates functions in
stream_solvers.u_r and stream_solvers.u_theta

"""

import numpy as np
from scipy.integrate import solve_ivp as ivp
import matplotlib.pyplot as plt
import stream_solvers as slm
plt.style.use("cool-style.mplstyle")

def u_r(r, theta, a=1):
    return 1
def u_theta(r, theta, a =1):
    return np.cos(theta+0.01)

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

#u_rad_cast = np.log(u_rad_cast)
#u_the_cast = np.log(u_the_cast)
f_r = slm.rbs(coarse_r, coarse_t, u_rad_cast)
f_t = slm.rbs(coarse_r, coarse_t, u_the_cast)

interp_sol = ivp(slm.dydt_rbs, rspan, thetas, t_eval = radii, args = (f_r, f_t))
slm.plot_mult(interp_sol.t, interp_sol.y, color = "blue", lw = 0.6)
