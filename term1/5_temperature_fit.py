"""
fitting function for temperature
"""

import os
import numpy as np
from scipy.integrate import solve_ivp as ivp
import matplotlib.pyplot as plt
import stream_solvers as slm
from scipy.io import loadmat
from scipy.interpolate import interp1d as interp
from scipy.interpolate import interp2d as interp2
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit





plt.style.use("cool-style.mplstyle")
pl_color = 'slategrey'
fig, ax = plt.subplots(1, 2)


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


# cutting r and theta to desired grid size
rb_num = 100 # first # radii
thb_num = len(T) - 20 # = len(T) # for full size # determines first # angles
rb_short = rb_sc[:rb_num]
thb_short = (thb.T[:thb_num]).T

#cut temperature to desired grid size, to fit dimensions of r and theta
T_short = T
T_short = T_short.T[:rb_num]
T_short = T_short.T
T_short = T_short[:thb_num]

f_T = interp2(rb_short, thb_short/np.pi, np.log(T_short), kind = 'linear')
T_boundary = f_T(1.022, 0.158) # visually determined edge of atmosphere temp

'''
'''
#plt.xkcd()
ax1 = plt.subplot(1, 2, 1)

plt.contourf(thb_short/np.pi, rb_short, np.log(T_short.T), 64, cmap = "BuPu")
plt.ylabel(r"height ($r_0$)")
plt.xlabel("angle ($\pi$)")
c = plt.colorbar()
c.set_label("ln(Temperature)")
#plt.savefig("images/HD_pure_sim/T_r_theta_full.pdf", format = "pdf")

ax2 = plt.subplot(1, 2, 2, sharex = ax1, sharey = ax1)


plt.contourf(thb_short/np.pi, rb_short, f_T(rb_short, thb_short/np.pi).T, 64, cmap = "BuPu")
plt.ylabel(r"height ($r_0$)")
plt.xlabel("angle ($\pi$)")
c = plt.colorbar()
c.set_label("ln(Temperature)")
#plt.show()



rb_crits = []
thb_crits = [] # in units of pi rads

for t in thb_short/np.pi:
    rc = 0
    rc = [r for r in rb_short if f_T(r, t) > T_boundary][0]
    T_c = f_T(rc, t)
    rb_crits.append(rc)
    thb_crits.append(t)

# fitting the constant temperature lines as a function of angle
f_rc = interp(thb_crits, rb_crits, kind = 'linear')
f_rc_filter = savgol_filter(rb_crits, 51, 3)
thetas = np.linspace(thb_crits[0], thb_crits[-1], len(thb_crits) *2)
plt.plot(thetas, f_rc(thetas), color = 'red')
plt.plot(thb_crits, f_rc_filter, color = 'orange')

#%%


#%%
"""
contour plot of temperature on cartesian grid

temperature fitted at critical temperature T_boundary
"""

fig, ax = plt.subplots()


#cut temperature to desired grid size

X_short = np.outer(rb_short,np.sin(thb_short))
Z_short = np.outer(rb_short,np.cos(thb_short))


plt.contourf(X_short, Z_short, np.log(T_short).T, 200, cmap = "BuPu")
c = plt.colorbar()
c.set_label("Temperature (K)")

planet = plt.Circle((0, 0), rb_short[0], color=pl_color)
ax.add_patch(planet)
thb_crits_resc = np.array(thb_crits)*np.pi
plt.plot(slm.cart_x(f_rc_filter, thb_crits_resc), slm.cart_y(f_rc_filter, thb_crits_resc), color = "red")
plt.text(0, 0, "planet zaza <3")
plt.show()


#%%
"""
plotting sigmoids at all angles of planet for temperature
plots T as a function of r for all thetas (overlapped)
 
"""

rc_crits = []
tc_crits = [] # in units of pi rads

fine_rb = np.linspace(rb_short[0], rb_short[-1] - 5, len(rb_short) * 100)

for t in thb_short[0:10]/np.pi:
    rc = 0
    rc = [r for r in rb_short if f_T(r, t) > T_boundary][0]
    T_c = f_T(rc, t)
    # plt.plot(rb_short, f_T(rb_short, t))
    rc_crits.append(rc)
    tc_crits.append(t)
    p0 = np.array([max(f_T(rb_short, t)), np.median(rb_short), 1, min(f_T(rb_short, t))]) # this is an mandatory initial guess

    popt, pcov = curve_fit(slm.sigmoid, rb_short, f_T(rb_short, t), p0, method='lm')
    plt.plot(fine_rb, slm.sigmoid(fine_rb, *popt), 'r-')
    #plt.plot(rb_short, sigmoid(rb_short, *popt), 'b-')
    #for r in rb_short:
    #    plt.scatter(rb_short, f_T(rb_short, t))
    
#%%  
"""
plotting sigmoids at all angles of planet for temperature
plots T as a function of r for all thetas (overlapped)

compacted 

grid is cut to exclude blatant nightside
"""
slm.fit_sigmoid(rb_short, thb_short, T_short)

#%%
"""
plotting sigmoids at all angles of planet for temperature
plots T as a function of r for all thetas (overlapped)

compacted 

full grid
"""

slm.fit_sigmoid(rb_sc, thb, T)

# cut the thb short to like [0:116] in slm.sigmoid if it bugs
