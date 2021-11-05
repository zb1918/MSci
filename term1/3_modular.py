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
    return a
def u_theta(r, theta, a =1):
    return np.cos(theta)


thetas = np.linspace(0, 2, 100)*np.pi #initial conditions to cover 2pi radians

#----------------------setting the radius limits------------------------------#
r_lim = 20
radii = np.linspace(1, r_lim, r_lim*10)
rspan = np.array([radii[0], radii[-1]])

#----------------------coarse grid for interpolation--------------------------#
#can adjust coarseness to evaluate accuracy of interpolation
coarse_r = np.linspace(1, r_lim, r_lim*5)
coarse_t = np.linspace(0, 6, 100)*np.pi

#----------------------interpolating coarse grid points-----------------------#
u_rad_cast = slm.cast(coarse_r, coarse_t, u_r, args = (2))
u_the_cast = slm.cast(coarse_r, coarse_t, u_theta, args = (2))
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
coarse_r = np.linspace(1, r_lim, 10)
coarse_t = np.linspace(0, 2, 10)*np.pi

#----------------------interpolating coarse grid points-----------------------#
u_rad_cast = slm.cast(coarse_r, coarse_t, u_r, args = (2))
u_the_cast = slm.cast(coarse_r, coarse_t, u_theta, args = (2))
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

#slm.plot_mult(direct_sol.t, direct_sol.y, color = "orange", lw = 2)
slm.plot_mult(interp_sol.t, interp_sol.y, color = "blue", lw = 0.6)

ax_slide = plt.axes([.25, .1, .65, .03])
val_a = Slider(ax_slide, "changing value",
               valmin=0.2,valmax=1.0,valinit=.6,valstep=.02)


def update(val):
    current_a = val_a.val

val_a.on_changed(update)
plt.show()



'''
for y in range(len(direct_sol.y)):
    slm.plot_cart(direct_sol.t, direct_sol.y[y], "orange", lw = 2)
    #slm.plot_cart(interp_sol2.t, interp_sol2.y[y], "red", lw = 1.2)
    slm.plot_cart(interp_sol.t, interp_sol.y[y], "blue", lw = 1)

plt.legend()
'''
#%%
'''
fine_r = np.linspace(1, r_lim, 10)
fine_t = np.linspace(0, 2, 10)*np.pi


z = []
for r in range(len(fine_r)):
    z_r = []
    for t in range(len(fine_t)):
        z_rt = f_r(r,t) - slm.u_r(r,t)
        z_r.append(z_rt)
    z.append(z_r)


fig, ax = plt.subplots()
aaa=ax.pcolormesh(fine_r, fine_t, f_r(fine_r, fine_t) - slm.u_r(fine_t,fine_r))
fig.colorbar(aaa)

'''

#%%

def u_r_new(a):
    return lambda r,t: a

def u_t_new(a):
    return lambda r,t: 1

u_r = u_r_new(1)
u_t = u_t_new(1)


direct_sol = ivp(slm.dydt, rspan, thetas, t_eval = radii,
                 args = (u_r, u_t)
                 )

def cart_y(r, theta):
    return r*np.cos(theta)
def cart_x(r,theta):
    return r*np.sin(theta)
def plot_cart(r, theta, colour = "green", lw = 1):
    plt.plot(cart_x(r, theta), cart_y(r, theta), color = colour, lw = lw)

fig = plt.figure()
plt.subplots_adjust(bottom = 0.25)
ax = fig.subplots()

#for ys in range(len(direct_sol.y)):
#    p, = ax.plot(cart_x(direct_sol.t,direct_sol.y[ys]), cart_y(direct_sol.t, direct_sol.y[ys]), lw = 0.6, color = "green")   

ax_slide = plt.axes([.25, .1, .65, .03])
val_a = Slider(ax_slide, "changing value",
               valmin=0.2,valmax=2.0,valinit=1,valstep=.02)


p = []
for ys in range(len(direct_sol.y)):
    p.append(ax.plot(cart_x(direct_sol.t,direct_sol.y[ys]), cart_y(direct_sol.t, direct_sol.y[ys]), lw = 0.6, color = "green")) 


''''''
def update(val):
    
    current_a = val_a.val
    u_r = u_r_new(current_a)
    u_t = u_t_new(current_a)
    
    direct_sol = ivp(slm.dydt, rspan, thetas, t_eval = radii,
                 args = (u_r, u_t)
                 )
    for ys in range(len(direct_sol.y)):
        p[ys][0].set_xdata(cart_x(direct_sol.t, direct_sol.y[ys]))
        p[ys][0].set_ydata(cart_y(direct_sol.t, direct_sol.y[ys]))
        plt.draw()


val_a.on_changed(update)

plt.show()

