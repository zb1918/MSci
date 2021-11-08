import numpy as np
from scipy.integrate import solve_ivp as ivp
import matplotlib.pyplot as plt
import stream_solvers as slm
from matplotlib.widgets import Slider
plt.style.use("cool-style.mplstyle")

def u_r_new(a):
    return lambda r,t: 1
def u_t_new(a):
    return lambda r,t: np.cos(a*t)
#generating functions with given coefficients
u_r, u_t = u_r_new(1), u_t_new(1)

t_res = 50
####DO NOT EDIT THETAS!!!!!!!###
thetas = np.linspace(0, 2, t_res+1)[0:-1]*np.pi #initial conditions to cover 2pi radians
#########!!!!!!!!!!!!!!#########

#----------------------setting the radius limits------------------------------#
r_lim = 20
radii = slm.rad(r_lim, 100)(0)
rspan = np.array([radii[0], radii[-1]])

#----------------------coarse grid for interpolation--------------------------#
#can adjust coarseness to evaluate accuracy of interpolation
coarse_r, coarse_t = slm.rad(r_lim, 30)(0), np.linspace(0, 3, 100)[0:-1]*np.pi

#----------------------interpolating coarse grid points-----------------------#
#casting onto coarse grid:
u_rad_cast, u_the_cast = slm.cast(coarse_r, coarse_t, u_r), slm.cast(coarse_r, coarse_t, u_t)
#interpolation:
f_r, f_t = slm.rbs(coarse_r, coarse_t, u_rad_cast), slm.rbs(coarse_r, coarse_t, u_the_cast)
#solving the ODE:
interp_sol = ivp(slm.dydt_rbs, rspan, thetas, t_eval = radii, args = (f_r, f_t))
direct_sol = ivp(slm.dydt, rspan, thetas, t_eval = radii, args = (u_r, u_t))

#----------------------creating plots and sliders-----------------------------#
fig = plt.figure()
ax = fig.subplots()
plt.subplots_adjust(bottom = 0.25) #shifts the plot up
ax_slide = plt.axes([0.2, 0.1, 0.7, 0.05])
val_res = Slider(ax_slide, "res", color = "midnightblue", valmin=3, valmax=150, valinit=1, valstep=3)
val_res.label.set_size(15)

#---------------------------------PLOTTING------------------------------------#
#-----------------------------initial plotting--------------------------------#
p, q = [], []

#plot streamlines in x, y basis
for ys in range(len(interp_sol.y)):
    p.append(ax.plot(slm.cart_x(interp_sol.t,interp_sol.y[ys]), slm.cart_y(interp_sol.t, interp_sol.y[ys]), lw = 0.6, color = "navy"))
    #q.append(ax.plot(slm.cart_x(direct_sol.t,direct_sol.y[ys]), slm.cart_y(direct_sol.t, direct_sol.y[ys]), lw = 0.6, color = "maroon"))

'''
#plot streamlines in r, theta basis
plt.xlabel("r")
plt.ylabel(r'$\theta$')
plt.grid()

for ys in range(len(interp_sol.y)):
    p.append(ax.plot(interp_sol.t, interp_sol.y[ys], lw = 0.6, color = "navy"))
    q.append(ax.plot(direct_sol.t, interp_sol.y[ys], lw = 1, color = "maroon"))
'''
def update(val):
    current_res = val_res.val
    radii = slm.rad(r_lim, current_res)(0)
    rspan = [radii[0], radii[-1]]
    interp_sol = ivp(slm.dydt_rbs, rspan, thetas, t_eval = radii, args = (f_r, f_t))
    
    #plot streamlines in x, y basis
    for ys in range(len(interp_sol.y)):
        p[ys][0].set_xdata(slm.cart_x(interp_sol.t, interp_sol.y[ys]))
        p[ys][0].set_ydata(slm.cart_y(interp_sol.t, interp_sol.y[ys]))
        plt.draw()
    '''
    ''''''
    #plot streamlines in r, theta basis
    for ys in range(len(interp_sol.y)): #loop through all intitial values
        #p is for interpolated
        p[ys][0].set_xdata(interp_sol.t)
        p[ys][0].set_ydata(interp_sol.y[ys])
        #q is for function 
        q[ys][0].set_xdata(direct_sol.t)
        q[ys][0].set_ydata(direct_sol.y[ys])
    '''
val_res.on_changed(update)

plt.show()   
       
#%%




