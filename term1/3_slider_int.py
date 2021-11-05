import numpy as np
from scipy.integrate import solve_ivp as ivp
import matplotlib.pyplot as plt
import stream_solvers as slm
from matplotlib.widgets import Slider


def u_r_new(a):
    return lambda r,t: 1

def u_t_new(a):
    return lambda r,t: np.cos(a*t)

u_r = u_r_new(1)
u_t = u_t_new(1)

plt.style.use("cool-style.mplstyle")

thetas = np.linspace(0, 2, 100)*np.pi #initial conditions to cover 2pi radians

#----------------------setting the radius limits------------------------------#
r_lim = 20
radii = np.linspace(1, r_lim, r_lim*10)
rspan = np.array([radii[0], radii[-1]])

#----------------------coarse grid for interpolation--------------------------#
#can adjust coarseness to evaluate accuracy of interpolation

def coarse_x(xi, xf, n):
    return lambda f: np.linspace(xi, xf, n)

coarse_r = coarse_x(1, r_lim, 20)
coarse_t = coarse_x(0, 2*np.pi, 100)

coarse_r = coarse_r(0)
coarse_t = coarse_t(0)

#----------------------interpolating coarse grid points-----------------------#
#casting onto coarse grid:
u_rad_cast = slm.cast(coarse_r, coarse_t, u_r)
u_the_cast = slm.cast(coarse_r, coarse_t, u_t)
#interpolation:
f_r = slm.rbs(coarse_r, coarse_t, u_rad_cast)
f_t = slm.rbs(coarse_r, coarse_t, u_the_cast)

interp_sol = ivp(slm.dydt_rbs, rspan, thetas, t_eval = radii, args = (f_r, f_t))


def cart_y(r, theta):
    return r*np.cos(theta)
def cart_x(r,theta):
    return r*np.sin(theta)
def plot_cart(r, theta, colour = "green", lw = 1):
    plt.plot(cart_x(r, theta), cart_y(r, theta), color = colour, lw = lw)

fig = plt.figure()
plt.subplots_adjust(bottom = 0.25)
ax = fig.subplots()

ax_slide = plt.axes([0.2, 0.1, 0.7, 0.05])
val_xi = Slider(ax_slide, "xi",
               valmin=0.2,valmax=5.0,valinit=1,valstep=.02)

ax_slide2 = plt.axes([0.2, 0.05, 0.7, 0.05])
val_xf = Slider(ax_slide2, "xf",
               valmin=0.2,valmax=5.0,valinit=1,valstep=.02)

p = []
for ys in range(len(interp_sol.y)):
    p.append(ax.plot(cart_x(interp_sol.t,interp_sol.y[ys]), 
                     cart_y(interp_sol.t, interp_sol.y[ys]), 
                     lw = 0.6, color = "green")) 

''''''
def update(val):
    
    current_xi = val_xi.val
    current_xf = val_xf.val
    
    coarse_r = coarse_x(current_xi, current_xf, 10)
    coarse_r = coarse_r(0)
    u_rad_cast = slm.cast(coarse_r, coarse_t, u_r)
    f_r = slm.rbs(coarse_r, coarse_t, u_rad_cast)
    interp_sol = ivp(slm.dydt_rbs, rspan, thetas, t_eval = radii, args = (f_r, f_t))

    for ys in range(len(interp_sol.y)):
        p[ys][0].set_xdata(cart_x(interp_sol.t, interp_sol.y[ys]))
        p[ys][0].set_ydata(cart_y(interp_sol.t, interp_sol.y[ys]))
        plt.draw()


val_xi.on_changed(update)
val_xf.on_changed(update)
plt.show()
