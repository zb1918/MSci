"""

"""

import numpy as np
from scipy.integrate import solve_ivp as ivp
import matplotlib.pyplot as plt
import stream_solvers as slm
from matplotlib.widgets import Slider
plt.style.use("cool-style.mplstyle")


#----------------------setting the radius limits------------------------------#
r_lim = 30
radii = slm.rad(r_lim, 100)(0)
rspan = np.array([radii[0], radii[-1]])

thetas = np.linspace(0, 2, 80)*np.pi #initial conditions to cover 2pi radians
thetas = thetas[0:-1]

def int_sol(coarse_r, coarse_t, u_r, u_theta):
    """
    casts onto coarse grid and interpolates. returns integrated interpolated f
    """

    u_rad_cast = slm.cast(coarse_r, coarse_t, u_r)
    u_the_cast = slm.cast(coarse_r, coarse_t, u_theta)
    f_r = slm.rbs(coarse_r, coarse_t, u_rad_cast)
    f_t = slm.rbs(coarse_r, coarse_t, u_the_cast)

    return ivp(slm.dydt_rbs, rspan, thetas, t_eval = radii, args = (f_r, f_t))

#%%
def u_r(r, theta):
    return r
def u_theta(r, theta):
    return -np.cos(theta)*np.sin(theta)
#%%

"""
interpolated onto coarse grid

"""
#----------------------coarse grid for interpolation--------------------------#
#can adjust coarseness to evaluate accuracy of interpolation
coarse_r = slm.rad(r_lim, 100)(0)
coarse_t = np.linspace(-6, 6, 300)*np.pi

interp_sol = int_sol(coarse_r, coarse_t, u_r, u_theta)
#%%

"""
interpolation with different grid/resolution for comparison

"""
#----------------------coarse grid for interpolation--------------------------#
#can adjust coarseness to evaluate accuracy of interpolation
coarse_r = slm.rad(r_lim - 1, 10)(0)
coarse_t = np.linspace(-6, 6, 50)*np.pi

interp_sol2 = int_sol(coarse_r, coarse_t, u_r, u_theta)
#%%
"""
solving streamlines for functions in 
streamlines.u_r and streamlines.u_theta directly

"""

direct_sol = ivp(slm.dydt, rspan, thetas, t_eval = radii,
                 args = (u_r, u_theta)
                 )

#%%

slm.plot_mult(direct_sol.t, direct_sol.y, color = "red", lw = 2)
plt.show()
slm.plot_mult(interp_sol.t, interp_sol.y, color = "blue", lw = 0.6)
plt.show()
slm.plot_mult(interp_sol2.t, interp_sol2.y, color = "blue", lw = 0.6, ls = 'dashed')
plt.show()

#%%
#128 x 256
#will probably have to interpolate in log space instead for accuracy
#hydro field (nomag)

#plot r,theta plots for ur and utheta
#can downgeade to linear interpolation and compare to cubic
#can even go higherr (5th order spline/4th)
#is thesimulatiom resolution appropriate for the problem?
#

