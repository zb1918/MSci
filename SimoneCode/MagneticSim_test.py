"""
Created on Sun Dec 12 21:46:23 2021
@author: Simone Di Giampasquale
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from scipy.io import loadmat
from scipy import interpolate as inter
from scipy.integrate import solve_ivp as ivp
from scipy.spatial import Delaunay

#%%
# loads dictionary of MHD variables
# r - is b grid for r
# th - is the b grid for th
# v1 - is the vr velocity ALREADY averaged to the b grid
# v2 is the vth velocity ALREADY averaged to the b grid
# rho is the mass density 
# Xe is the electron fraction
# U is the energy density
# B1 is the Br magnetic field ALREADY averaged onto the b grid
# B2 is the Bth magnetid field ALREADY averaged onto the b grid

MHD_sim = loadmat("Bp3G_sim.mat")

rb = MHD_sim['r']       # radius 1D array
thb = MHD_sim['th']     # theta 1D array
vrb = MHD_sim['v1']     # u_r
vthb = MHD_sim['v2']    # u_theta
D = MHD_sim['rho']      # mass density
Br = MHD_sim['B1']      # B_r
Bth = MHD_sim['B2']     # B_theta
U = MHD_sim['U']        # internal energy pu volume
Xe = MHD_sim['Xe']      # electron fraction ie fraction of ionised hydrogen
Xh = 1-Xe               # fraction of non ionised hydrogen
F = 1e6                 # flux in erg/s/cm^2

B = np.sqrt((Br**2)+(Bth**2))   # absolute value of B field

## Calculate temperature ##
g = 5/3                  # adiabatic gamma for monoatomic gas
P = (g-1)*U              # pressure
mu = 1.00784*1.66e-24    # mean molecular mass
k = 1.380649e-16         # boltzmann constant in cgs units
T = P*mu/(D*k)           # temperature in cgs units

## Scale radii by planet radius ##
radii = rb/rb[0]
r_max = radii[-1][0]

rax,tax = np.meshgrid(radii,thb)  # radius, theta grids
X = np.outer(radii, np.sin(thb))  # x grid --> X AXIS POINTS TOWARDS THE STAR
Z = np.outer(radii, np.cos(thb))  # z grid --> Z AXIS PERPENDICULAR TO THE ORBIT

thetas = np.arange(0,np.pi,0.01)
theta_eval = np.array([thetas]).T

#%%
                  ### SOLVE IN LOGSPACE OF B COMPONENTS ###
###--------------Define the log of absolute value of B_r, B_th------------- ###
###------------------Define the sign matrix of B_r, B_th--------------------###

log_Br = np.log10(np.abs(Br)) 
log_Bt = np.log10(np.abs(Bth))

sign_Br = np.sign(Br)
sign_Bt = np.sign(Bth)

###-----------------Interpolate BOTH the log and the sign-------------------###

get_log_Br = inter.RectBivariateSpline(radii.T[0], thb.T[0], log_Br)
get_log_Bt = inter.RectBivariateSpline(radii.T[0], thb.T[0], log_Bt)

get_sign_Br = inter.RectBivariateSpline(radii.T[0], thb.T[0], sign_Br, kx=1, ky=1)
get_sign_Bt = inter.RectBivariateSpline(radii.T[0], thb.T[0], sign_Bt, kx=1, ky=1)

###############################################################################
###------------Define the RHS using log(velocity) integrating in r----------###

def get_rhs_log(rad,theta,log_Br_int,log_Bt_int,sign_r,sign_t):
    '''
    Right hand side of our differential equation dtheta/dr = B_t/(r*B_r)
    using the interpolated log(B comp) and sign of B comps 
    '''
    #Set up with grid=False   
    log_B_r = log_Br_int(rad,theta,grid=False)
    log_B_t = log_Bt_int(rad,theta,grid=False)
    
    sign_B_r = np.sign(sign_r(rad,theta,grid=False))
    sign_B_t = np.sign(sign_t(rad,theta,grid=False))
    
    B_r = sign_B_r*(10**log_B_r)
    B_t = sign_B_t*(10**log_B_t)
    
    f_r = B_t/(B_r*rad)
    
    return f_r

###------------Define the RHS using log(velocity) integrating in th---------###

def get_rhs_log_inv(theta,rad,log_Br_int,log_Bt_int,sign_r,sign_t):
    '''
    Right hand side of our differential equation dr/dtheta = (r*B_r)/B_t
    using the interpolated log(B comp) and sign of B comps 
    '''
    #Set up with grid=False   
    log_B_r = log_Br_int(rad,theta,grid=False)
    log_B_t = log_Bt_int(rad,theta,grid=False)
    
    sign_B_r = np.sign(sign_r(rad,theta,grid=False))
    sign_B_t = np.sign(sign_t(rad,theta,grid=False))
    
    B_r = sign_B_r*(10**log_B_r)
    B_t = sign_B_t*(10**log_B_t)
    
    f_r_inv = (B_r*rad)/B_t
    
    return f_r_inv

#%%
fine_r = np.copy(radii)
space = np.diff(radii.T[0])
for i in range(len(radii)-1):
    fine_r = np.insert(fine_r,4*i+1,radii.T[0][i]+(space[i]/4))
    fine_r = np.insert(fine_r,4*i+2,radii.T[0][i]+(2*space[i]/4))
    fine_r = np.insert(fine_r,4*i+3,radii.T[0][i]+(3*space[i]/4)) 

## Integrate forwards in radius and stop when Br is negative (closed lines) ##
def Br_zero_log(radius,theta,log_B_r,log_B_th,sign_r,sign_t):  # define for events in sole_ivp
    log_B_r = log_B_r(radius,theta,grid=False)
    sign_B_r = np.sign(sign_r(radius,theta,grid=False))
    B_r = sign_B_r*(10**log_B_r)
    return B_r
Br_zero_log.terminal=True

Radii_log = []
Angles_log = []

r_terminal_log = []
th_terminal_log = []

for i in range(len(theta_eval)):
    r_start = 1.04
    interval_r = (r_start,r_max)
    #slice the array to start from the first value higher than minimum r
    fine_r_filter = np.copy(fine_r)
    j = 0
    while fine_r[j] < r_start:
        j += 1            
    fine_r_filter = fine_r_filter[j:]
    
    #print(i)
    
    sol_log = ivp(get_rhs_log, interval_r, theta_eval[i], args=(get_log_Br, get_log_Bt,\
                  get_sign_Br,get_sign_Bt), t_eval=fine_r_filter, events=[Br_zero_log])
        
    Radii_log.append(sol_log.t)
    Angles_log.append(sol_log.y)
    
    if len(sol_log.t_events[0]) != 0:
        r_terminal_log.append(sol_log.t_events[0])
        th_terminal_log.append(sol_log.y_events[0])

#%%
for i in range(len(Angles_log)-1):
    for j in range(len(Angles_log[i+1][0])-1):
        a0 = Angles_log[i+1][0][0]
        a_j = Angles_log[i+1][0][j]
        a_j1 = Angles_log[i+1][0][j+1]
       
        if a0>np.pi/6 and a_j > a_j1:
            a = Angles_log[i+1][0][:j]
            a.shape = (1,len(Angles_log[i+1][0][:j]))  # to have a row vector
            Angles_log[i+1] = a
            
            r = Radii_log[i+1][:j]
            Radii_log[i+1] = r
            break
        
        else:
            pass
        
#%%
### separate open streamlines from closed streamlines ###
open_r = Radii_log[:50]
open_th = Angles_log[:50]

closed_r = Radii_log[50:]
closed_th = Angles_log[50:]

### plot open streamlines ###
for i in range(49):
    stream_x_log = Radii_log[i]*np.sin(Angles_log[i][0])
    stream_z_log = Radii_log[i]*np.cos(Angles_log[i][0])
    plt.plot(stream_x_log, stream_z_log,'r','.',lw = 0.5)

circle1 = plt.Circle((0, 0), 1, color='k')
plt.gca().add_patch(circle1)
plt.show()

#%%
### FIND HELIUM FRACTION ON THE OPEN STREAMLINES ###
### Define a list of steps dr, dtheta to find dl ###
dr = []
dth = []
for i in range(len(open_r)):
    diff_r = np.diff(open_r[i])
    diff_th = np.diff(open_th[i])
    z = np.array([0])
    
    dr_i = np.append(z,np.abs(diff_r))
    dth_i = np.append(z,np.abs(diff_th))
    
    dr.append(dr_i)
    dth.append(dth_i)
    
open_dl = []
for i in range(len(dr)):
    r = open_r[i]
    th = open_th[i]
    
    dl_i = np.sqrt(((r*dth[i])**2)+((dr[i])**2))
    open_dl.append(dl_i)
    
### Define the cumulative sum of dl's ###
open_l = []
for i in range(len(open_dl)):
    l_i = np.append(z,np.cumsum(open_dl[i][1:]))
    open_l.append(l_i)


### Interpolate the streamline to find r and theta depending on l ###
### create a list of interpolators st we have one for each streamline ###
get_r = []
get_th = []
for i in range(len(open_l)):
    #plt.plot(stream_l[i],stream_r[i],'r')
    int_r = inter.interp1d(open_l[i]*rb[0], open_r[i]*rb[0])
    int_th = inter.interp1d(open_l[i]*rb[0], open_th[i])
        
    get_r.append(int_r)
    get_th.append(int_th)
    
#%%
### DEFINE PHYSICAL QUANTITIES WE NEED ###
n = D/(1.00784*1.66e-24)
ne = n*Xe 
n0 = n*Xh 
u = np.sqrt((vrb**2)+(vthb**2))
log_u = np.log10(np.abs(u))

## Atomic factors ###
A31 = 1.272e-4                  # radiative de-excitation rate 2S tri -> 1S
q13a = 4.5e-20                  # collisional excitation rate 1S -> 2S tri
q31a = 2.6e-8                   # collisional excitation rate 2S tri -> 2S sing
q31b = 4.0e-9                   # collisional excitation rate 2S tri -> 2P sing
Q31 = 5e-10                     # collisional de-exciration rate 2S tri -> 1S

### Physical quantites of the system ###
F1 = ((10e3)/(24.6*1.602e-12))*(7.82e-18)  # photoionisation rate of 1S
F3 = ((10e3)/(4.8*1.602e-12))*(4.84e-18)   # photoionisation rate of 2S triplet

def a1(r,t):                               # recombination rate of 1S 
    if get_T(r,t,grid=False) <= 1.5e3:
        a1 = 2.17e-13
    else:
        a1 = 1.54e-13
    return a1

def a3(r,t):                               # recombination rate of 2S triplet 
    if get_T(r,t,grid=False) <= 1.5e3:
        a3 = 1.97e-14
    else:
        a3 = 1.49e-14 
    return a3

### INTERPOLATE THINGS WE NEED ###
get_T = inter.RectBivariateSpline(radii.T[0],thb.T[0],T,kx=1,ky=1) 
get_ne = inter.RectBivariateSpline(radii.T[0],thb.T[0],ne,kx=1,ky=1)
get_n0 = inter.RectBivariateSpline(radii.T[0],thb.T[0],n0,kx=1,ky=1)
get_log_u = inter.RectBivariateSpline(radii.T[0],thb.T[0],log_u)
#linear since sharp increase and non-sharp part is in the night side mostly

#%%
### DEFINE THE RHS OF THE EQN ###
def get_rhs(l,f,i):           # need to find an expression for r and theta from l
    '''
    RHS of the QM equation for helium fractions in terms of distance along the streamline
    l = distance along the streamline
    f = helium fraction, 1D array with values f1 and f3
    i = index of the streamline we are considering
    '''
    r = get_r[i](l)/rb[0]
    t = get_th[i](l)
    n_e = get_ne(r,t,grid=False)
    n_0 = get_n0(r,t,grid=False)
    a_1 = a1(r,t)
    a_3 = a3(r,t)
    
    vel = (10**get_log_u(r,t,grid=False))
    
    A = (-n_e*a_1-F1-n_e*q13a)
    B = (-n_e*a_1+A31+n_e*q31a+n_e*q31b+n_0*Q31)
    C = (-n_e*a_3+n_e*q13a)
    D = (-n_e*a_3-A31-F3-n_e*q31a-n_e*q31b-n_0*Q31)
    
    g1 = (n_e*a_1 + A*f[0] + B*f[1])/vel
    g2 = (n_e*a_3 + C*f[0] + D*f[1])/vel
    
    g = np.array([g1,g2])
    return g

f1 = []
open_f3 = []
l_sol = []

for i in range(len(open_l)):
        
    l0 = (open_l[i][0],open_l[i][-1])*rb[0]  
     
    f0 = np.array([1,1e-10])
    sol_f = ivp(get_rhs, l0, f0, method='LSODA',args=[i],t_eval=open_l[i]*rb[0],\
                atol=1e-12,rtol=1e-6)
    #print(i)
    
    f1.append(sol_f.y[0])
    open_f3.append(sol_f.y[1])
    l_sol.append(sol_f.t)

#%%
### FIND DATA POINTS IN r-theta FOR OPEN STREAMLINES ###
r_sol = []
th_sol = []

for i in range(len(l_sol)): 
    r_sol.append(get_r[i](l_sol[i]))
    th_sol.append(get_th[i](l_sol[i]))

# input of Delauney are coordinate points in the form (x,z) since we want cartesian grid
open_points = []
for i in range(len(r_sol)):
    for j in range(len(r_sol[i])):
        rad = r_sol[i][j]/rb[0][0]
        ang = th_sol[i][0][j]
        
        coord = [rad*np.sin(ang),rad*np.cos(ang)]
        open_points.append(coord)

## put values of f3 in a single list s.t. they correspond to their coordinates
## take the LOGARITHM 
open_logf3_list = []
for i in range(len(open_f3)):
    for j in range(len(open_f3[i])):
        frac = np.log10(open_f3[i][j])
        open_logf3_list.append(frac)

open_logf3 = np.asarray(open_logf3_list)

#%%
### USE STEADY STATE SOLN FOR CLOSED STREAMLINES ###
closed_points = []
cl_ne = np.array([])
cl_n0 = np.array([])
cl_a1 = np.array([])
cl_a3 = np.array([])

for i in range(len(closed_r)-2):
    for j in range(len(closed_r[i])):
        rad = closed_r[i][j]
        ang = closed_th[i][0][j]
        coord = [rad*np.sin(ang),rad*np.cos(ang)]
        
        closed_points.append(coord)
        cl_ne=np.append(cl_ne,[get_ne(rad,ang,grid=False)])
        cl_n0=np.append(cl_n0,[get_n0(rad,ang,grid=False)])
        cl_a1=np.append(cl_a1,[a1(rad,ang)])
        cl_a3=np.append(cl_a3,[a3(rad,ang)]) 
        
cl_A = (-cl_ne*cl_a1-F1-cl_ne*q13a)
cl_B = (-cl_ne*cl_a1+A31+cl_ne*q31a+cl_ne*q31b+cl_n0*Q31)
cl_C = (-cl_ne*cl_a3+cl_ne*q13a)
cl_D = (-cl_ne*cl_a3-A31-F3-cl_ne*q31a-cl_ne*q31b-cl_n0*Q31) 
cl_X1 = cl_ne*cl_a1
cl_X3 = cl_ne*cl_a3

closed_logf3 = np.log10(((cl_C*cl_X1)-(cl_A*cl_X3))/((cl_A*cl_D)-(cl_B*cl_C)))

#%%
### MERGE THE OPEN AND CLOSED STREAMLINES VALUES FOR TRIANGULATION
points_list = open_points.copy()
for i in range(len(closed_points)):    
    points_list.append(closed_points[i])

points = np.asarray(points_list)
tri = Delaunay(points)   # does the triangulation

logf3_values = np.append(open_logf3,closed_logf3)

get_logf3 = inter.LinearNDInterpolator(tri,logf3_values,fill_value=np.nan)

#%%
plt.contourf(X,Z,10**get_logf3(X,Z),200)
circle1 = plt.Circle((0, 0), 1, color='k')
plt.gca().add_patch(circle1)
plt.colorbar()

#%%
### GENERATE A GRID FOR LINE OF SIGHT ###
x = np.linspace(0,r_max,500)
y = np.linspace(1.03,8,500)
x_grid, z_grid = np.meshgrid(x,y)

## Interpolate what we need on a cartesian grid ##
logD = np.log10(D)
get_logD = inter.RectBivariateSpline(radii.T[0],thb.T[0],logD)

### CALCULATE COLUMN DENSITY ###
dx = np.diff(x)[0]
dy = np.diff(y)[0]

XHe = 0.3                        # mass fraction of hydrogen (ISM)
mHe = 4.002602*1.66e-24          # mass of Helium

N = []                           #column density at different impact parameters (x values)
for i in range(len(y)):          # calculate N_j at constant z (ie y) values along x
    f3_line = 10**get_logf3(x,y[i])
    r_i = np.sqrt(x**2+y[i]**2)
    th_i = np.arctan2(x,y[i])
    D_line = 10**get_logD(r_i,th_i,grid=False)
    
    N_j = np.nansum(XHe*D_line*f3_line*dx*rb[0]/mHe)
    N.append(N_j)

plt.plot(y,np.asarray(N))
plt.yscale('log')

import AbsorptionLine_Hydro as Ab
plt.plot(Ab.x,np.asarray(Ab.N))
plt.yscale('log')

#%%
plt.contourf(x_grid,z_grid,10**get_logf3(x_grid,z_grid),200)