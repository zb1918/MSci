"""
Created on Sat Nov 27 19:11:46 2021
@author: Simone Di Giampasquale
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import interpolate as inter
from scipy.integrate import solve_ivp as ivp
from scipy.optimize import curve_fit

#%%
## Load simulation data ##
hydro_sim = loadmat("pure_HD.mat")

rb = hydro_sim['r']  #1D array of radius for grid
thb = hydro_sim['th']  #1D array of theta for grid

vrb = hydro_sim['vr']    # u_r
vthb = hydro_sim['vth']  # u_theta
D = hydro_sim['D']       # mass density
U = hydro_sim['U']       # internal energy pu volume
Xe = hydro_sim['ne']     # electron fraction ie fraction of ionised hydrogen
Xh = 1-Xe                # fraction of non ionised hydrogen

## Calculate temperature ###
g = 5/3                  # adiabatic gamma for monoatomic gas
P = (g-1)*U              # pressure
mu = 1.00784*1.66e-24    # mean molecular mass
k = 1.380649e-16         # boltzmann constant in cgs units
T = P*mu/(D*k)           # temperature in cgs units

## Scale by radius of the planet and define cartesian and polar grids ##
radii = rb/rb[0]  
r_max = radii[-1][0]

rax,tax = np.meshgrid(radii,thb)
X = np.outer(radii,np.sin(thb))
Z = np.outer(radii,np.cos(thb))

#%%
# solve ODE using interpolated functions ###
def get_rhs(rad,theta,ur_interp,ut_interp):
    '''
    Right hand side of our differential equation dtheta/dr = u_t/(r*u_r)
    using the interpolated velocity functions 
    '''
    #Set up with grid=False to avoid calculations with a single radius matched
    #to more than one angle which we do not need since we are interested in
    #calculating at a point with specific coordinates (r,theta)   
    vel_r = ur_interp(rad,theta,grid=False)
    vel_t = ut_interp(rad,theta,grid=False)
    
    f_r = vel_t/(vel_r*rad)
    return f_r

## INTERPOLATE VELOCITIES ##
get_vr = inter.RectBivariateSpline(radii.T[0], thb.T[0], vrb)
get_vt = inter.RectBivariateSpline(radii.T[0], thb.T[0], vthb)

#%%
## Define starting radii as funtction of angles ##
def r_min(theta):
    rad_min = (theta/14.5473)**2 + 1.01659
    return rad_min

## Integrate forwards in radius and stop when velocity is negative ##
def ur_zero(radius,theta,u_r,u_th):  # define for events in sole_ivp
    vel_r = u_r(radius,theta)
    return vel_r
    
ur_zero.terminal = True   # to stop the integration when u_r goes to zero
#ur_zero.direction = -1    # going from positive to negative u_r

thetas = np.arange(0,1.84,0.05)
theta_eval = np.array([thetas]).T

Radii = []
Angles = []
for i in range(len(theta_eval)):
    r_start = r_min(theta_eval[i][0])
    interval_r = (r_start,r_max)
    r_eval = np.linspace(r_start, r_max, 700)
    
    sol = ivp(get_rhs, interval_r, theta_eval[i], args=(get_vr, get_vt),\
              t_eval=r_eval, events=[ur_zero])
    
    Radii.append(sol.t)
    Angles.append(sol.y)

for i in range(30):
    plt.plot(Radii[i]*np.sin(Angles[i][0]),Radii[i]*np.cos(Angles[i][0]),'r','.')
    
plt.contourf(X,Z,vrb/np.abs(vrb),2)         #plot the sign of v_r
circle1 = plt.Circle((0, 0), 1, color='k')
plt.gca().add_patch(circle1)
plt.xlim(0,1.5)
plt.ylim(-1.5,1.5)
plt.show()

#%%
#define a finer grid inserting points in between the original radius array
#the grid will stay finer near the planet
#however cannot change initial radius this way  

fine_r = np.copy(radii)
space = np.diff(radii.T[0])
for i in range(len(radii)-1):
    fine_r = np.insert(fine_r,4*i+1,radii.T[0][i]+(space[i]/4))
    fine_r = np.insert(fine_r,4*i+2,radii.T[0][i]+(2*space[i]/4))
    fine_r = np.insert(fine_r,4*i+3,radii.T[0][i]+(3*space[i]/4))   
      
Radii_fine = []
Angles_fine = []
#need to use a loop otherwise events command doesn't work
for i in range(len(theta_eval)):
    sol_fine = ivp(get_rhs,(fine_r[0],fine_r[-1]), theta_eval[i], args=(get_vr, get_vt),\
              t_eval=fine_r, events=[ur_zero])
    
    Radii_fine.append(sol_fine.t)
    Angles_fine.append(sol_fine.y)

for i in range(len(Radii_fine)):
    plt.plot(Radii_fine[i]*np.sin(Angles_fine[i][0]),\
             Radii_fine[i]*np.cos(Angles_fine[i][0]),'r','.')

        
plt.contourf(X,Z,vrb/np.abs(vrb),2)         #plot the sign of v_r
circle1 = plt.Circle((0, 0), 1, color='k')
plt.gca().add_patch(circle1)
plt.xlim(0,1.5)
plt.ylim(-1.5,1.5)
plt.show()

#%%
# try eliminating values from fine_r that are less than the radius we want to start with
# to both use a more refined grid close to the planet and different starting radii

Radii_fine_filter = []
Angles_fine_filter = []

r_terminal = []
th_terminal = []


for i in range(len(theta_eval)):
    r_start = r_min(theta_eval[i][0])
    interval_r = (r_start,r_max)
    #slice the array to start from the first value higher than minimum r
    fine_r_filter = np.copy(fine_r)
    j = 0
    while fine_r[j] < r_start:
        j += 1            
    fine_r_filter = fine_r_filter[j:] 
    
    sol_filt = ivp(get_rhs, interval_r, theta_eval[i], args=(get_vr, get_vt),\
              t_eval=fine_r_filter, events=[ur_zero])
    
    Radii_fine_filter.append(sol_filt.t)
    Angles_fine_filter.append(sol_filt.y)
    
    if len(sol_filt.t_events[0]) != 0:
        r_terminal.append(sol_filt.t_events[0])
        th_terminal.append(sol_filt.y_events[0])


for i in range(len(Radii_fine_filter)):
    stream_x = Radii_fine_filter[i]*np.sin(Angles_fine_filter[i][0])
    stream_z = Radii_fine_filter[i]*np.cos(Angles_fine_filter[i][0])
    plt.plot(stream_x, stream_z,'r','.')

plt.contourf(X,Z,vrb/np.abs(vrb),2)         #plot the sign of v_r
circle1 = plt.Circle((0, 0), 1, color='k')
plt.gca().add_patch(circle1)
plt.xlim(0,1.5)
plt.ylim(-1.5,1.5)
plt.show()

#%%
## Intergrate backwards by switching the interval
Radii_back = []
Angles_back = []

for i in range(len(r_terminal)):
    interval_r_back = (r_terminal[i][0],1.04)
    r_eval_back = np.linspace(r_terminal[i][0],1.04,50)
    
    sol_back = ivp(get_rhs, interval_r_back, th_terminal[i][0], args=(get_vr, get_vt),\
                   max_step = 0.001, events=[ur_zero])
    
    Radii_back.append(sol_back.t)
    Angles_back.append(sol_back.y)
    
for i in range(len(Radii_back)):
    stream_x_back = Radii_back[i]*np.sin(Angles_back[i][0])
    stream_z_back = Radii_back[i]*np.cos(Angles_back[i][0])
    plt.plot(stream_x_back, stream_z_back,'k','.')
    
