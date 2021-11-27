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
D = hydro_sim['D']
U = hydro_sim['U']
ne = hydro_sim['ne']


X = np.outer(rb_sc,np.sin(thb)) # meshgrid of X coordinates
Z = np.outer(rb_sc,np.cos(thb)) # meshgrid of Z coordinates

radii = np.linspace(1, rb_sc_max, 128)


#relevant equations
# P = (gamma -1)e
# P = k/mu rho T

# u = internal energy/volume
# tau = optical depth
# ne = fraction of ionised hydrogen
# d = mass density



"""
task for next week:
plot log T as function of log(height) and investigate points just before the slope begins
determine point at which radius will begin
"""

"""
streamlines for interpolated velocity or log(velocity) matrices
correction for valid streamlines included
radial distances scaled

"""

fig, ax = plt.subplots()     

#--------------------------logarithms of velocities---------------------------#
vrb_signs = vrb/abs(vrb)
vrb_log = np.log(abs(vrb))
vrb_log_signs = np.multiply(vrb_log, vrb_signs)

vthb_signs = vthb/abs(vthb)
vthb_log = np.log(abs(vthb))
vthb_log_signs = np.multiply(vthb_log, vthb_signs)



#----------------------fine grid for post-interpolation-----------------------#
radii = np.linspace(1.04, rb_sc_max-10, 2000)
rspan = [radii[0], radii[-1]]
#thetas = np.linspace(0, 0.75, 10)*np.pi
thetas = np.array([0.32])*np.pi

#----------------------interpolating coarse grid points-----------------------#
f_r = slm.rbs(rb_sc, thb, vrb)
f_t = slm.rbs(rb_sc, thb, vthb)
"""
# uncomment the following for a logarithmic plot
f_r = slm.rbs(rb_sc, thb, vrb_log_signs)
f_t = slm.rbs(rb_sc, thb, vthb_log_signs)

"""
def event(t, y, fr, ft):
    return fr(t,y)
event.terminal = True
interp_sol = ivp(slm.dydt_rbs, rspan, thetas, t_eval = radii, args = (f_r, f_t), events = [event])

a = interp_sol.t_events
b = interp_sol.y_events

valid_sols = []
valid_sol_ys = []
valid_sol_ts = []

edge_of_valid = False
# night side corrections
n_s_x = 1.0    # radius of planet
n_s_z = 0.0    # correction

for i in range(len(interp_sol.y)): # loop over all streamlines
    valid = True # true if streamline never crosses over to night side
    valid_sol_y = []
    valid_sol_t = []
    
    for j in range(len(interp_sol.y[i])): # loop over points of one streamline
    
        sol_x = interp_sol.t[j]*np.sin(interp_sol.y[i][j])
        sol_z = interp_sol.t[j]*np.cos(interp_sol.y[i][j])
        
        if ((sol_x**2 < n_s_x**2) and (sol_z < n_s_z)) or (sol_x < 0): # if on night side
            valid = False
            edge_of_valid = True
        else:
            valid_sol_y.append(interp_sol.y[i][j])
            valid_sol_t.append(interp_sol.t[j])
            
        if sol_x**2 + sol_z**2 < 1: # if inside planet
            valid = False
            edge_of_valid = True
    valid_sol_ys.append(valid_sol_y)
    valid_sol_ts.append(valid_sol_t)
         

    if (valid == True):# & (edge_of_valid == False):
        #slm.plot_cart(interp_sol.t, interp_sol.y[i])
        #slm.plot_cart(interp_sol.t, -1*interp_sol.y[i]) #reflection across axis
        valid_sols.append(interp_sol.y[i])

if len(valid_sols) > 0:
    print("last valid angle ", round(valid_sols[-1][0], 3), "pi")
    print(len(valid_sols), "/", len(thetas), "dayside solutions")
"""
"""
# all the solutions with deletion
for i in range(len(valid_sol_ys)):
    slm.plot_cart(valid_sol_ts[i], valid_sol_ys[i], ls = None)
    #plt.scatter(slm.cart_x(valid_sol_ts[i], valid_sol_ys[i]), slm.cart_y(valid_sol_ts[i], valid_sol_ys[i]))

# all the solutions:
slm.plot_mult(interp_sol.t, interp_sol.y, color = "blue", lw = 0.6)

planet = plt.Circle((0, 0), 1, color=pl_color)
ax.add_patch(planet)

#plt.savefig("images/velocity.pdf", format="pdf")
plt.show()


"""
contour plot of velocity on cartesian grid

"""
a = interp_sol.t_events
b = interp_sol.y_events

plt.contourf(X, Z, vrb, 64, cmap = "BuPu", levels = [-2e8, 0, 2e8])
c = plt.colorbar()
c.set_label("velocity")
for i in range(len(a)):
    plt.scatter(slm.cart_x(a[i], b[0]), slm.cart_y(a[i], b[0]), color = 'red')
plt.show()

