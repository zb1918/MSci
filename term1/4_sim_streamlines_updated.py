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
hydro_sim = loadmat("term1/sims/pure_HD.mat") #extract pure_HD simulation

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

#---------------------------constants in cgs----------------------------------#
H_molecule = 1.00784        # mass of H in a.m.u.
amu = 1.66e-24              # 1 a.m.u. in g
H_mu = H_molecule * amu     # mass of a single H molecule in g
gamma = 5/3                 # 1 + 2/d.o.f. where d.o.f. of H is taken to be 3
kb = 1.38e-16               # boltzmann constant in cgs

#------------------------calculating temperature------------------------------#
T = U * H_mu * (gamma - 1)/(D * kb)
T = T.T
radii = np.linspace(1, rb_sc_max, 128)

#%%

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

#%%
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
radii = np.linspace(1.04, rb_sc_max, 500)
rspan = [radii[0], radii[-1]]
thetas = np.linspace(0, 0.75, 10)*np.pi
#thetas = np.array([0.4])*np.pi

#----------------------interpolating coarse grid points-----------------------#
f_r = slm.rbs(rb_sc, thb, vrb)
f_t = slm.rbs(rb_sc, thb, vthb)
"""
# uncomment the following for a logarithmic plot
f_r = slm.rbs(rb_sc, thb, vrb_log_signs)
f_t = slm.rbs(rb_sc, thb, vthb_log_signs)

"""

interp_sol = ivp(slm.dydt_rbs, rspan, thetas, t_eval = radii, args = (f_r, f_t))

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

print("last valid angle ", round(valid_sols[-1][0], 3), "pi")
print(len(valid_sols), "/", len(thetas), "dayside solutions")
"""
"""
for i in range(len(valid_sol_ys)):
    slm.plot_cart(valid_sol_ts[i], valid_sol_ys[i], ls = None)
    #plt.scatter(slm.cart_x(valid_sol_ts[i], valid_sol_ys[i]), slm.cart_y(valid_sol_ts[i], valid_sol_ys[i]))
# all the solutions:
#slm.plot_mult(interp_sol.t, interp_sol.y, color = "blue", lw = 0.6)

#slm.plot_mult(valid_sol_ts[0], valid_sol_ys[0][0], color = "red", lw = 0.7)

planet = plt.Circle((0, 0), 1, color=pl_color)
ax.add_patch(planet)

#plt.savefig("images/velocity.pdf", format="pdf")
plt.show()

#%%
"""
USELESS (?)
plot log(Temperature) as a function of radial distance
at different angles (with different colors)

"""
x = np.linspace(0, len(T[0]) -1, len(T[0]))

#rb_sc = np.log(rb_sc)

for i in range(128):
    if i < 20:
        #ove you<333 i   wasgiyur work :999999
        #but i didn't printzahra is azing'
        plt.plot(x, np.log(T[i]), color = "red")
    if (i < 40) & (i >=20): 
        plt.plot(x, np.log(T[i]), color = "orange")
    if (i < 60) & (i >=40):
        plt.plot(x, np.log(T[i]), color = "yellow")
    if (i < 80) & (i >=60):
        plt.plot(x, np.log(T[i]), color = "lawngreen")
    if (i < 100) & (i >=60):
        plt.plot(x, np.log(T[i]), color = "blue")
    if (i < 120) & (i >=100):
        plt.plot(x, np.log(T[i]), color = "purple")
    if i > 120:
        plt.plot(x, np.log(T[i]), color = "black")
        
        
""""""
#%%
"""
contour plot of velocity on cartesian grid

"""
plt.contourf(X, Z, vthb, 64, cmap = "BuPu")
c = plt.colorbar()
c.set_label("velocity")
plt.show()


#attempt at interpolating the temperature on boundary of planet
#plt.plot(rb_sc, np.log(T[0]), color = "navy")
#f_T = interp(rb_sc, np.log(T[0]), kind = "cubic")
#plt.plot(radii, f_T(radii))

#%%
"""
contour plot of temperature on cartesian grid

"""

fig, ax = plt.subplots()
rb_num = 50
thb_num = len(T)
#cut r and theta to desired grid size
rb_short = rb_sc[:rb_num] #original rb is array of array of single elements
thb_short = (thb.T[:thb_num]).T

#cut temperature to desired grid size
T_short = T
T_short = T_short.T[:rb_num]
T_short = T_short.T
T_short = T_short[:thb_num]

X_short = np.outer(rb_short,np.sin(thb_short))
Z_short = np.outer(rb_short,np.cos(thb_short))


plt.contourf(X_short, Z_short, T_short.T, 64, cmap = "BuPu")
c = plt.colorbar()
c.set_label("Temperature (K)")

planet = plt.Circle((0, 0), rb_short[0], color=pl_color)
ax.add_patch(planet)
plt.show()

#%%
"""
contour plot on radial grid of 
log(T) as a function of theta and height
height in units of r0

"""
rb_num = 30
thb_num = len(T)
#cut r and theta to desired grid size
rb_short = rb_sc[:rb_num] #original rb is array of array of single elements
thb_short = (thb.T[:thb_num]).T

#cut temperature to desired grid size
T_short = T
T_short = T_short.T[:rb_num]
T_short = T_short.T
T_short = T_short[:thb_num]

#plt.xkcd()
plt.contourf(rb_short, thb_short/np.pi, np.log(T_short), 64, cmap = "BuPu")
plt.xlabel(r"height ($r_0$)")
plt.ylabel("angle ($\pi$)")
c = plt.colorbar()
c.set_label("ln(Temperature)")
#plt.savefig("images/HD_pure_sim/T_r_theta_full.pdf", format = "pdf")
plt.show()


#%%
"""
contour plot on radial grid of 
log(T) as a function of theta and height
height in units of m

"""

# number of points (radial and angular respectively)
rb_num = 30
thb_num = 70
#cut r and theta to desired grid size
rb_short = rb[:rb_num] #original rb is array of array of single elements
thb_short = (thb.T[:thb_num]).T

#cut temperature to desired grid size
T_short = T
T_short = T_short.T[:rb_num]
T_short = T_short.T
T_short = T_short[:thb_num]

#plt.xkcd()
plt.contourf(rb_short, thb_short/np.pi, np.log(T_short), 64, cmap = "BuPu")
plt.xlabel(r"height ($m$)")
plt.ylabel("angle ($\pi$)")
c = plt.colorbar()
c.set_label("ln(Temperature)")
plt.show()