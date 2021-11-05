import numpy as np
from scipy.integrate import solve_ivp as ivp
import matplotlib.pyplot as plt
import stream_solvers as slm
from matplotlib.widgets import Slider


plt.style.use("cool-style.mplstyle")

thetas = np.linspace(0, 2, 100)*np.pi #initial conditions to cover 2pi radians

#----------------------setting the radius limits------------------------------#
r_lim = 20
radii = np.linspace(1, r_lim, r_lim*10)
rspan = np.array([radii[0], radii[-1]])





def u_r_new(a):
    return lambda r,t: a

def u_t_new(b):
    return lambda r,t: np.cos(b*t)

u_r = u_r_new(1)
u_t = u_t_new(1)


direct_sol = ivp(slm.dydt, rspan, thetas, t_eval = radii, args = (u_r, u_t))

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
val_a = Slider(ax_slide, "a",
               valmin=0.05,valmax=5.0,valinit=1,valstep=.02)

ax_slide2 = plt.axes([0.2, 0.05, 0.7, 0.05])
val_b = Slider(ax_slide2, "b",
               valmin=0.05,valmax=5.0,valinit=1,valstep=.02)

p = []
for ys in range(len(direct_sol.y)):
    p.append(ax.plot(cart_x(direct_sol.t,direct_sol.y[ys]), 
                     cart_y(direct_sol.t, direct_sol.y[ys]), 
                     lw = 0.6, color = "green")) 

''''''
def update_a(val):
    
    current_a = val_a.val
    u_r = u_r_new(current_a)
    
    direct_sol = ivp(slm.dydt, rspan, thetas, t_eval = radii,
                 args = (u_r, u_t)
                 )
    for ys in range(len(direct_sol.y)):
        p[ys][0].set_xdata(cart_x(direct_sol.t, direct_sol.y[ys]))
        p[ys][0].set_ydata(cart_y(direct_sol.t, direct_sol.y[ys]))
        plt.draw()
        
def update_b(val):
    
    current_b = val_b.val
    u_t = u_t_new(current_b)
    
    direct_sol = ivp(slm.dydt, rspan, thetas, t_eval = radii,
                 args = (u_r, u_t)
                 )
    for ys in range(len(direct_sol.y)):
        p[ys][0].set_xdata(cart_x(direct_sol.t, direct_sol.y[ys]))
        p[ys][0].set_ydata(cart_y(direct_sol.t, direct_sol.y[ys]))
        plt.draw()


val_a.on_changed(update_a)
val_b.on_changed(update_b)
plt.show()
