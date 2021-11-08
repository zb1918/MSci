import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
from scipy.integrate import solve_ivp as ivp
import matplotlib.pyplot as plt
import stream_solvers as slm
from matplotlib.widgets import Slider


plt.style.use("cool-style.mplstyle")

thetas = np.linspace(0, 2, 50)*np.pi #initial conditions to cover 2pi radians
thetas = thetas[1:-1]

#----------------------setting the radius limits------------------------------#
r_lim = 20
radii = np.linspace(1, r_lim, r_lim*10)
rspan = np.array([radii[0], radii[-1]])

def u_r_new(a):
    return lambda r,t: r**-0.5

def u_t_new(b):
    return lambda r,t: np.cos(b*t)

#generating the functions by passing argument to lambda functions above
u_r = u_r_new(1)
u_t = u_t_new(1)

#solving the ODE
direct_sol = ivp(slm.dydt, rspan, thetas, t_eval = radii, args = (u_r, u_t))

#---------------------------plotting functions--------------------------------#

def cart_y(r, theta):
    return r*np.cos(theta)
def cart_x(r,theta):
    return r*np.sin(theta)
def plot_cart(r, theta, colour = "grey", lw = 1):
    plt.plot(cart_x(r, theta), cart_y(r, theta), color = colour, lw = lw)

fig = plt.figure()
plt.subplots_adjust(bottom = 0.2)
ax = fig.subplots()

#----------------------creating the sliders-----------------------------------#
ax_slide_a, ax_slide_b = plt.axes([0.2, 0.1, 0.7, 0.05]), plt.axes([0.2, 0.05, 0.7, 0.05])

val_a = Slider(ax_slide_a, "a",
               valmin=-5,valmax=5.0,valinit=1,valstep=.002, color = "blue")
val_b = Slider(ax_slide_b, "b",
               valmin=0.05,valmax=5.0,valinit=1,valstep=.002, color = "red")

#initial plot
p = []
for ys in range(len(direct_sol.y)):
    p.append(ax.plot(cart_x(direct_sol.t,direct_sol.y[ys]), 
                     cart_y(direct_sol.t, direct_sol.y[ys]), 
                     lw = 0.6, color = "navy")) 

#------------------------slider update functions------------------------------#
def update_a(val):
    
    #grabs the new value for a
    current_a = val_a.val
    u_r = u_r_new(current_a)
    
    #updates the ivp solver with the new velocity
    direct_sol = ivp(slm.dydt, rspan, thetas, t_eval = radii,
                 args = (u_r, u_t)
                 )
    #plotting
    for ys in range(len(direct_sol.y)):
        p[ys][0].set_xdata(cart_x(direct_sol.t, direct_sol.y[ys]))
        p[ys][0].set_ydata(cart_y(direct_sol.t, direct_sol.y[ys]))
        plt.draw()
        
def update_b(val):
    
    #grabs the new value for b
    current_b = val_b.val
    u_t = u_t_new(current_b)
    
    #updates the ivp solver with the new velocity
    direct_sol = ivp(slm.dydt, rspan, thetas, t_eval = radii,
                 args = (u_r, u_t)
                 )
    
    #plotting
    for ys in range(len(direct_sol.y)):
        p[ys][0].set_xdata(cart_x(direct_sol.t, direct_sol.y[ys]))
        p[ys][0].set_ydata(cart_y(direct_sol.t, direct_sol.y[ys]))
        plt.draw()

#----------------------update on slider movement------------------------------#
val_a.on_changed(update_a)
val_b.on_changed(update_b)

plt.show()
