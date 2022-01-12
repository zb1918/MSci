import numpy as np
from scipy.integrate import solve_ivp as ivp
import matplotlib.pyplot as plt
import stream_solvers as slm
from scipy.io import loadmat
from scipy.interpolate import interp1d as interp

plt.style.use("cool-style.mplstyle")
pl_color = 'slategrey'
#%%

"""
extraction of data and physical calculations
"""
hydro_sim = loadmat("sims/pure_HD.mat") #extract pure_HD simulation

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


X = np.outer(rb_sc,np.sin(thb)) # meshgrid of X coordinates
Z = np.outer(rb_sc,np.cos(thb)) # meshgrid of Z coordinates


plt.contourf(X, Z, vrb, 64, cmap = "BuPu", levels = [-2e8, 0, 2e8])
c = plt.colorbar()
c.set_label("velocity")

#----------------------interpolating coarse grid points-----------------------#
f_r = slm.rbs(rb_sc, thb, vrb)
f_t = slm.rbs(rb_sc, thb, vthb)


def event(t, y, fr, ft):
    return fr(t, y)

event.terminal = True
event.direction = -1


#----------------------fine grid for post-interpolation-----------------------#
rspan = [1.05, rb_sc_max-11]
    

thetas = np.linspace(0.2, 0.65, 20)*np.pi
#thetas = np.array([0.38])*np.pi
r_stops = []
t_stops = []

rb_sc = rb_sc[25:]

for theta in thetas:
    print("theta")
    event.direction = -1
    
    radii = []
    for i in range(len(rb_sc)-1):
        subr = np.linspace(rb_sc[i], rb_sc[i+1], 5)
        radii.append(subr[0:4])
    radii = np.array(radii)
    radii = radii.flatten()
    rspan = [radii[0], radii[-1]]
    
    interp_sol = ivp(slm.dydt_rbs, rspan, [theta], t_eval = radii, args = (f_r, f_t), events = (event))#, rtol = 1e-7)
    interp_sol.y = interp_sol.y[0] # doing single solutions at a time 
    
    if interp_sol.t_events[0].size > 0: # if termination has actually happened
        print("TERMINATED")
        r_stop = interp_sol.t_events[0][0]
        t_stop = interp_sol.y_events[0][0][0]
        print(r_stop, interp_sol.t[-1])
        print(t_stop, interp_sol.y[-1])

        interp_sol.t = np.append(interp_sol.t, r_stop)
        interp_sol.y = np.append(interp_sol.y, t_stop)
        
        event.direction = 1
        rev_radii= []
        for r in radii:
            if r < r_stop:
                rev_radii.append(r)
        rev_radii = rev_radii[::-1]
        rspan = [rev_radii[0], rev_radii[-1]]
        
        interp_sol2 = ivp(slm.dydt_rbs, rspan, [interp_sol.y[-1]], t_eval = rev_radii, args = (f_r, f_t), events = (event))
        interp_sol2.y = interp_sol2.y[0]
        interp_sol.y = np.append(interp_sol.y, interp_sol2.y)
        interp_sol.y = interp_sol.y.flatten()
        interp_sol.t = np.append(interp_sol.t, interp_sol2.t)
        interp_sol.t = interp_sol.t.flatten()

    slm.plot_cart(interp_sol.t, interp_sol.y, color = "blue", lw = 2)
    xt = slm.cart_x(interp_sol.t, interp_sol.y)
    yt = slm.cart_y(interp_sol.t, interp_sol.y)
    #plt.scatter(xt, yt, color = "red")
    r_stops.append(r_stop)
    t_stops.append(t_stop)
    
    

'''
for i in range(len(r_stops)):
    if r_stops[i][0].size > 0:
        ax = slm.cart_x(r_stops[i], t_stops[i])
        ay = slm.cart_y(r_stops[i], t_stops[i])
        plt.scatter(ax, ay, color = "red")
'''





#slm.plot_cart(interp_sol.t, interp_sol.y[1], color = "purple", lw = 2)



plt.show()


#%%

# interpolate n3
# solveivp(t,x,y,y')
# now x y and y' are vector
# y= [f1 f2]


# recombination is faster at lower temperatures
# managed by coloumb forces


# helium


# construct cartesian grid around dayside flow
# interpolate for velocity, helium fraction, temperature etc
# lambda of interest -> cross section
# absorption can be calculated per grid cell 
# absorption can be caluclated per grid wedge
# add up absorption including the planet's size
# can test with n3 - e.g. spherical n3
# stop at around 6rp

#%%

# interpolate density in log space
# Temp and ne in linear but discounting the rapid change

