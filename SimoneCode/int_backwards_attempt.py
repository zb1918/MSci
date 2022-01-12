"""
Created on Sat Nov 27 19:11:46 2021
@author: Simone Di Giampasquale
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
from scipy import interpolate as inter
from scipy.integrate import solve_ivp as ivp
from scipy.optimize import curve_fit

#%%
## Load simulation data ##
hydro_sim = loadmat("pure_HD.mat")

rb = hydro_sim['r']      #1D array of radius for grid
thb = hydro_sim['th']    #1D array of theta for grid

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

thetas = np.arange(0,1.8,0.005)
theta_eval = np.array([thetas]).T

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
    rad_min = (theta/14.7)**2 + 1.025
    return rad_min

## Integrate forwards in radius and stop when velocity is negative ##
def ur_zero(radius,theta,u_r,u_th):  # define for events in sole_ivp
    vel_r = u_r(radius,theta,grid=False)
    return vel_r

## Stop integration in the night side when u_t is negative as well ##
def ut_zero(theta,radius,u_r,u_th):
    vel_t = u_th(radius,theta,grid=False)
    return vel_t

ur_zero.terminal = True   # to stop the integration when u_r goes to zero
ut_zero.terminal = True   # to stop the integration when u_t goes to zero
#ur_zero.direction = -1    # going from positive to negative u_r
#%%
#solve streamlines
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

#%%
#solve streamlines for the finer grid but with same radius     
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
             Radii_fine[i]*np.cos(Angles_fine[i][0]),'r','.',lw=0.5)

        
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
    
    print(ur_zero(interval_r[0],theta_eval[i],get_vr,get_vt))
    print(i)
    
    sol_filt = ivp(get_rhs, interval_r, theta_eval[i], args=(get_vr, get_vt),\
              t_eval=fine_r_filter, events=[ur_zero])
        
    
    Radii_fine_filter.append(sol_filt.t)
    Angles_fine_filter.append(sol_filt.y)
    
    if len(sol_filt.t_events[0]) != 0:
        r_terminal.append(sol_filt.t_events[0])
        th_terminal.append(sol_filt.y_events[0])
        
#%%

for i in range(len(Radii_fine_filter)):
    stream_x = Radii_fine_filter[i]*np.sin(Angles_fine_filter[i][0])
    stream_z = Radii_fine_filter[i]*np.cos(Angles_fine_filter[i][0])
    plt.plot(stream_x, stream_z,'r','.',lw = 0.5)

plt.contourf(X,Z,vrb/np.abs(vrb),2)         #plot the sign of v_r
circle1 = plt.Circle((0, 0), 1, color='k')
plt.gca().add_patch(circle1)
plt.xlim(0,1.5)
plt.ylim(-1.5,1.5)
plt.show()

#%%
## Intergrate backwards by switching the interval
## DON'T USE THIS, USE THE NEXT CELL WHICH INTEGRATES IN THETA ##
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

#%%
## Integrate in the night side by switching to theta as variable ##

def get_rhs_inv(theta,rad,ur_interp,ut_interp):
    '''
    Right hand side of our differential equation dr/dtheta = (u_r*r)/u_t
    using the interpolated velocity functions 
    '''
    #Set up with grid=False to avoid calculations with a single radius matched
    #to more than one angle which we do not need since we are interested in
    #calculating at a point with specific coordinates (r,theta)   
    vel_r = ur_interp(rad,theta,grid=False)
    vel_t = ut_interp(rad,theta,grid=False)
    
    f_r = (vel_r*rad)/(vel_t)
    return f_r

Angles_night = []
Radii_night = []

for i in range(len(th_terminal)):
    interval_th = (th_terminal[i][0],np.pi)
    th_eval = np.linspace(th_terminal[i][0],2.8,50)
    
    sol_inv = ivp(get_rhs_inv, interval_th, r_terminal[i], args=(get_vr, get_vt),\
                   max_step = 0.01, events=[ut_zero])
        
    Angles_night.append(sol_inv.t)
    Radii_night.append(sol_inv.y)
    
for i in range(len(Angles_night)):
    stream_x_night = Radii_night[i][0]*np.sin(Angles_night[i])
    stream_z_night = Radii_night[i][0]*np.cos(Angles_night[i])
    plt.plot(stream_x_night, stream_z_night,'y','.',lw = 0.5)
    
#%%
# TRY TO MERGE THE STREAMLINES
a = 0
stream_r = []     # r coords of streamlines
stream_th = []    # theta coords of streamlines
for i in range(len(Radii_fine_filter)):
    if Radii_fine_filter[i][-1] != 14.645348426646176:
        arr_r = np.concatenate((Radii_fine_filter[i],Radii_night[a][0]))
        arr_th = np.concatenate((Angles_fine_filter[i][0],Angles_night[a]))
        
        stream_r.append(arr_r)
        stream_th.append(arr_th)
        
        a += 1
        
    else:
        stream_r.append(Radii_fine_filter[i])
        stream_th.append(Angles_fine_filter[i][0])
        
for i in range(len(stream_r)):
    stream_x_all = stream_r[i]*np.sin(stream_th[i])
    stream_z_all = stream_r[i]*np.cos(stream_th[i])
    plt.plot(stream_x_all, stream_z_all,'r','.',lw = 0.5)
    
circle1 = plt.Circle((0, 0), 1, color='k')
plt.gca().add_patch(circle1)


#%%
# interpolate values along a streamline

### TASK 1: SOLVE EQUATIONS FOR n3 ###
# turn time derivative in spatial derivative using velocity
# initial condition: steady state since atmosphere has slow movement
# assume optically thin (check approximation later)
# think how to use Dn/Dt for eqn, should make easier
# need to solve coupled equations --> use matrix/vector like notation


### TASK 2: FIND ABSORPTION ON LINE OF SIGHT ###
# construct a cartesian grid of observable dayside around flow
# interpolate in the cart grid for all variables
# once known the wl we can calculate cross section to starlight
# once known cross section we can calculate reduction in flux

#%%
                ### SOLVE IN LOGSPACE OF VELOCITY ###

###--------------Define the log of absolute value of u_r, u_th------------- ###
###------------------Define the sign matrix of u_r, u_th--------------------###

log_ur = np.log10(np.abs(vrb)) 
log_ut = np.log10(np.abs(vthb))

sign_ur = np.sign(vrb)
sign_ut = np.sign(vthb)

###-----------------Interpolate BOTH the log and the sign-------------------###

get_log_ur = inter.RectBivariateSpline(radii.T[0], thb.T[0], log_ur)
get_log_ut = inter.RectBivariateSpline(radii.T[0], thb.T[0], log_ut)

get_sign_ur = inter.RectBivariateSpline(radii.T[0], thb.T[0], sign_ur, kx=1, ky=1)
get_sign_ut = inter.RectBivariateSpline(radii.T[0], thb.T[0], sign_ut, kx=1, ky=1)

###############################################################################
###------------Define the RHS using log(velocity) integrating in r----------###

def get_rhs_log(rad,theta,log_ur_int,log_ut_int,sign_r,sign_t):
    '''
    Right hand side of our differential equation dtheta/dr = u_t/(r*u_r)
    using the interpolated log(velocity) and sign of velocities 
    '''
    #Set up with grid=False   
    log_vel_r = log_ur_int(rad,theta,grid=False)
    log_vel_t = log_ut_int(rad,theta,grid=False)
    
    sign_vr = np.sign(sign_r(rad,theta,grid=False))
    sign_vt = np.sign(sign_t(rad,theta,grid=False))
    
    vel_r = sign_vr*(10**log_vel_r)
    vel_t = sign_vt*(10**log_vel_t)
    
    f_r = vel_t/(vel_r*rad)
    
    return f_r

###------------Define the RHS using log(velocity) integrating in th---------###

def get_rhs_log_inv(theta,rad,log_ur_int,log_ut_int,sign_r,sign_t):
    '''
    Right hand side of our differential equation dr/dtheta = (r*u_r)/u_t
    using the interpolated log(velocity) and sign of velocities 
    '''
    #Set up with grid=False   
    log_vel_r = log_ur_int(rad,theta,grid=False)
    log_vel_t = log_ut_int(rad,theta,grid=False)
    
    sign_vr = np.sign(sign_r(rad,theta,grid=False))
    sign_vt = np.sign(sign_t(rad,theta,grid=False))
    
    vel_r = sign_vr*(10**log_vel_r)
    vel_t = sign_vt*(10**log_vel_t)
    
    f_r_inv = (vel_r*rad)/vel_t
    
    return f_r_inv

#%%
###-------FIND THE STREAMLINES BY SOLVING THE ODE WITH LOG VELOCITIES-------###
###------USE THE FINER GRID AND THE DIFFERENT STARTING RADII AS BEFORE------###

## Integrate forwards in radius and stop when velocity is negative ##
def ur_zero_log(radius,theta,log_u_r,log_u_th,sign_r,sign_t):  # define for events in sole_ivp
    log_vel_r = log_u_r(radius,theta,grid=False)
    sign_vr = np.sign(sign_r(radius,theta,grid=False))
    vel_r = sign_vr*(10**log_vel_r)
    return vel_r

ur_zero_log.terminal=True


Radii_log = []
Angles_log = []

r_terminal_log = []
th_terminal_log = []


for i in range(len(theta_eval)):
    r_start = r_min(theta_eval[i][0])
    interval_r = (r_start,r_max)
    #slice the array to start from the first value higher than minimum r
    fine_r_filter = np.copy(fine_r)
    j = 0
    while fine_r[j] < r_start:
        j += 1            
    fine_r_filter = fine_r_filter[j:]
    
    print(i)
    
    sol_log = ivp(get_rhs_log, interval_r, theta_eval[i], args=(get_log_ur, get_log_ut,\
                  get_sign_ur,get_sign_ut), t_eval=fine_r_filter, events=[ur_zero_log])
        
    Radii_log.append(sol_log.t)
    Angles_log.append(sol_log.y)
    
    if len(sol_log.t_events[0]) != 0:
        r_terminal_log.append(sol_log.t_events[0])
        th_terminal_log.append(sol_log.y_events[0])

#%%
'''  
###############################################################################
####           ALL THIS PART IS USELESS NOW AS I MANUALLY ELIMINATE        ####
####               PART OF THE STREAMLINES IN THE CODE BELOW               ####
###############################################################################

###---------FIND STREAMLINES THAT TERMINATED BY INTEGRATING IN THETA--------###

def ut_zero_log(theta,radius,log_u_r,log_u_th,sign_r,sign_t):  # define for events in sole_ivp
    log_vel_t = log_u_th(radius,theta,grid=False)
    sign_vt = np.sign(sign_t(radius,theta,grid=False))
    vel_t = sign_vt*(10**log_vel_t)
    return vel_t

ut_zero_log.terminal=True

Angles_night_log = []
Radii_night_log = []

for i in range(len(th_terminal_log)):
    interval_th = (th_terminal_log[i][0],3)
    th_eval = np.linspace(th_terminal_log[i][0],3,50)
    
    sol_inv_log = ivp(get_rhs_log_inv, interval_th, r_terminal_log[i],\
                      args=(get_log_ur, get_log_ut, get_sign_ur, get_sign_ut),\
                      max_step = 0.01, events=[ut_zero_log])
        
    Angles_night_log.append(sol_inv_log.t)
    Radii_night_log.append(sol_inv_log.y)

### PLOT STREAMLINES FROM r INTEGRATION

for i in range(len(Radii_log)):
    stream_x_log = Radii_log[i]*np.sin(Angles_log[i][0])
    stream_z_log = Radii_log[i]*np.cos(Angles_log[i][0])
    plt.plot(stream_x_log, stream_z_log,'r','.',lw = 0.5)

### PLOT STREAMLINES FROM th INTEGRATION

for i in range(len(Angles_night_log)):
    stream_x_night_log = Radii_night_log[i][0]*np.sin(Angles_night_log[i])
    stream_z_night_log = Radii_night_log[i][0]*np.cos(Angles_night_log[i])
    plt.plot(stream_x_night_log, stream_z_night_log,'y','.',lw = 0.5)
    
plt.contourf(X,Z,np.sign(vrb),2)         #plot the sign of v_r
circle1 = plt.Circle((0, 0), 1, color='k')
plt.gca().add_patch(circle1)
plt.xlim(0,1.5)
plt.ylim(-1.5,1.5)
plt.show()
'''
#%%
### REMOVE THE PARTS OF STREAMLINES THAT GO BACKWARDS

for i in range(len(Angles_log)-1):
    for j in range(len(Angles_log[i+1][0])-1):
        a0 = Angles_log[i+1][0][0]
        a_j = Angles_log[i+1][0][j]
        a_j1 = Angles_log[i+1][0][j+1]
       
        if a0>np.pi/4 and a_j > a_j1:
            a = Angles_log[i+1][0][:j]
            a.shape = (1,len(Angles_log[i+1][0][:j]))  # to have a row vector
            Angles_log[i+1] = a
            
            r = Radii_log[i+1][:j]
            Radii_log[i+1] = r
            break
        
        else:
            pass

#%%
### PLOT STREAMLINES FROM INTEGRATION IN r ###

for i in range(len(Radii_log)):
    stream_x_log = Radii_log[i]*np.sin(Angles_log[i][0])
    stream_z_log = Radii_log[i]*np.cos(Angles_log[i][0])
    plt.plot(stream_x_log, stream_z_log,'r','.',lw = 0.5)

plt.contourf(X,Z,np.sign(vrb),2)
circle1 = plt.Circle((0, 0), 1, color='k')
plt.gca().add_patch(circle1)
plt.xlim(0,1.5)
plt.ylim(-1.5,1.5)
plt.show()

#%%
### RE-INTEGRATE IN THE NIGHT SIDE STARTING FROM THE NEW ENDPOINTS
def ut_zero_log(theta,radius,log_u_r,log_u_th,sign_r,sign_t):  # define for events in sole_ivp
    log_vel_t = log_u_th(radius,theta,grid=False)
    sign_vt = np.sign(sign_t(radius,theta,grid=False))
    vel_t = sign_vt*(10**log_vel_t)
    return vel_t

ut_zero_log.terminal=True

#create the starting point arrays from the endpoints of the other integration
th_terminal_log = []
r_terminal_log = []

for i in range(len(Radii_log)):
    if len(Radii_log[i])!=0 and Radii_log[i][-1] < 10:
        th_terminal_log.append(Angles_log[i][0][-1])
        r_terminal_log.append(Radii_log[i][-1])
    else:
        pass   

#solve the ivp ode

Angles_night_log = []
Radii_night_log = []

for i in range(len(th_terminal_log)):
    interval_th = (th_terminal_log[i],3)
    th_eval = np.linspace(th_terminal_log[i],3,50)
    
    sol_inv_log = ivp(get_rhs_log_inv, interval_th, [r_terminal_log[i]],\
                      args=(get_log_ur, get_log_ut, get_sign_ur, get_sign_ut),\
                      max_step = 0.01, events=[ut_zero_log])
        
    Angles_night_log.append(sol_inv_log.t)
    Radii_night_log.append(sol_inv_log.y)


#%%
### MERGE THE STREAMLINES ###
c = 0
stream_r = []     # r coords of streamlines
stream_th = []    # theta coords of streamlines
for i in range(len(Radii_log)-1):
    if len(Radii_log[i]) != 0 and Radii_log[i][-1] < 10:
        arr_r = np.concatenate((Radii_log[i],Radii_night_log[c][0]))
        arr_th = np.concatenate((Angles_log[i][0],Angles_night_log[c]))
        
        stream_r.append(arr_r)
        stream_th.append(arr_th)
        
        c += 1
        
    else:
        stream_r.append(Radii_log[i])
        stream_th.append(Angles_log[i][0])


del stream_r[-2:]
del stream_th[-2:]

#%%
### TRY ELIMINATE STREAMLINES THAT OVERLAP ###

stream_r_fix = []
stream_th_fix = []

for i in range(len(stream_r)-1):
    if stream_r[i][-1] <= stream_r[i][-2]:  #streamlines that go back
        if stream_r[i][-1] < stream_r[i+1][-1]: 
            # if two streamlines that go back overlap once (not always the case)
            # then the final radius of the ith streamline will be lower than the (i+1)th
            pass
        else:
            stream_r_fix.append(stream_r[i])
            stream_th_fix.append(stream_th[i])
            
    else:                                     # streamlines that don't go back
        if stream_th[i][-1] > stream_th[i+1][-1]:
            # if two streamlines that don't go back overlap once (not always the case)
            # then the final theta of the ith streamline will be greater than the (i+1)th
            pass
            
        else:
            stream_r_fix.append(stream_r[i])
            stream_th_fix.append(stream_th[i])               
                
#%%

for i in range(len(stream_r_fix)):
    plt.plot(stream_r_fix[i], stream_th_fix[i],'r','.',lw = 0.5)
   
# create pandas dataframes to save the lists as cvs #
r_df = pd.DataFrame(stream_r_fix)
th_df = pd.DataFrame(stream_th_fix)

r_df.to_csv('stream_r_005.csv', index=False, header=False)
th_df.to_csv('stream_th_005.csv', index=False, header=False)    

#%%
### PLOT ALL THE STREAMLINES ###      
  
for i in range(len(stream_r)):
    stream_x_all = stream_r[i]*np.sin(stream_th[i])
    stream_z_all = stream_r[i]*np.cos(stream_th[i])
    plt.plot(stream_x_all, stream_z_all,'r','.',lw = 0.5)

plt.contourf(X,Z,np.sign(vrb),2)
circle1 = plt.Circle((0, 0), 1, color='k')
plt.gca().add_patch(circle1)

    