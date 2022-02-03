"""
Created on Wed Jan 05 19:01:40 2022
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
### WILL INTERPOLATE ALL QUANTITIES
### USE LINEAR INTERPOLATION FOR QUANTITIES THAT VARY MUCH
### USE RECTBIVARIATESPLINE FOR THE OTHERS, POSSIBLY IN LOGSPACE
### RESULT ALONG STREAMLINES WILL GO ON AN IRREGULAR GRID, LOOK INTO IRREGULAR GRID INTERPOLATION
#%%
## Load simulation data ##
hydro_sim = loadmat("pure_HD.mat")

rb = hydro_sim['r']              #1D array of radius for grid
thb = hydro_sim['th']            #1D array of theta for grid
vrb = hydro_sim['vr']            # u_r
vthb = hydro_sim['vth']          # u_theta
u = np.sqrt((vrb**2)+(vthb**2))  # |u|
D = hydro_sim['D']               # mass density
n = D/(1.00784*1.66e-24)         # number density
U = hydro_sim['U']               # internal energy pu volume
Xe = hydro_sim['ne']             # electron fraction ie fraction of ionised hydrogen
ne = n*Xe                        # electron number density
Xh = 1-Xe                        # fraction of non ionised hydrogen
n0 = n*Xh                        # neutral hydrogen number density

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

## Atomic factors ###
A31 = 1.272e-4                  # radiative de-excitation rate 2S tri -> 1S
q13a = 4.5e-20                  # collisional excitation rate 1S -> 2S tri
q31a = 2.6e-8                   # collisional excitation rate 2S tri -> 2S sing
q31b = 4.0e-9                   # collisional excitation rate 2S tri -> 2P sing
Q31 = 5e-10                     # collisional de-exciration rate 2S tri -> 1S

### Physical quantites of the system ###
F1 = ((10e3)/(24.6*1.602e-12))*(7.82e-18)  # photoionisation rate of 1S
F3 = ((10e3)/(4.8*1.602e-12))*(5.48e-18)   # photoionisation rate of 2S triplet

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

#%%
### INTERPOLATE THINGS WE NEED ###

get_T = inter.RectBivariateSpline(radii.T[0],thb.T[0],T,kx=1,ky=1) 
get_ne = inter.RectBivariateSpline(radii.T[0],thb.T[0],ne,kx=1,ky=1)
get_n0 = inter.RectBivariateSpline(radii.T[0],thb.T[0],n0,kx=1,ky=1)
#linear since sharp increase and non-sharp part is in the night side mostly

### INTERPOLATE VELOCITIES IN LOGSPACE ###
log_u = np.log10(np.abs(u))
log_ur = np.log10(np.abs(vrb)) 
log_ut = np.log10(np.abs(vthb))

sign_ur = np.sign(vrb)
sign_ut = np.sign(vthb)

get_log_u = inter.RectBivariateSpline(radii.T[0],thb.T[0],log_u)
get_log_ur = inter.RectBivariateSpline(radii.T[0],thb.T[0],log_ur)
get_log_ut = inter.RectBivariateSpline(radii.T[0],thb.T[0],log_ut)
get_sign_ur = inter.RectBivariateSpline(radii.T[0],thb.T[0],sign_ur,kx=1,ky=1)
get_sign_ut = inter.RectBivariateSpline(radii.T[0],thb.T[0],sign_ut,kx=1,ky=1)

#%%
### IMPORT STREAMLINES FROM THE PREVIOUS THING ###

### Import as pandas dataframes ###
r_df = pd.read_csv(r'C:\Users\simon\Documents\GitHub\MSci\SimoneCode\stream_r_001.csv',header=None)
th_df = pd.read_csv(r'C:\Users\simon\Documents\GitHub\MSci\SimoneCode\stream_th_001.csv',header=None)

### Make a list from the dataframes ###
stream_r = []
stream_th = []
for i in range(len(r_df)):
    stream_r.append(np.asarray(r_df.T[i]))
    stream_th.append(np.asarray(th_df.T[i]))

### Define a list of steps dr, dtheta to find dl ##
dr = []
dth = []
for i in range(len(stream_r)):
    diff_r = np.diff(stream_r[i])
    diff_th = np.diff(stream_th[i])
    z = np.array([0])
    
    dr_i = np.append(z,np.abs(diff_r))
    dth_i = np.append(z,np.abs(diff_th))
    
    dr.append(dr_i)
    dth.append(dth_i)
    
stream_dl = []
for i in range(len(dr)):
    r = stream_r[i]
    th = stream_th[i]
    
    dl_i = np.sqrt(((r*dth[i])**2)+((dr[i])**2))
    stream_dl.append(dl_i)
    
### Define the cumulative sum of dl's ###
stream_l = []
for i in range(len(stream_dl)):
    l_i = np.append(z,np.cumsum(stream_dl[i][1:]))
    stream_l.append(l_i)


### Interpolate the streamline to find r and theta depending on l ###
### create a list of interpolators st we have one for each streamline ###
get_r = []
get_th = []
for i in range(len(stream_l)):
    #plt.plot(stream_l[i],stream_r[i],'r')
    int_r = inter.interp1d(stream_l[i]*rb[0], stream_r[i]*rb[0])
    int_th = inter.interp1d(stream_l[i]*rb[0], stream_th[i])
        
    get_r.append(int_r)
    get_th.append(int_th)
    
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
f3 = []
l_sol = []

for i in range(len(stream_l)):
    j = 1
    while math.isnan(stream_l[i][-j]) == True:
        j+=1
        
    l0 = (stream_l[i][0],stream_l[i][-j])*rb[0]  
     
    f0 = np.array([1,1e-10])
    sol_f = ivp(get_rhs, l0, f0, method='LSODA',args=[i],t_eval=stream_l[i]*rb[0],\
                atol=1e-12,rtol=1e-6)
    #print(i)
    
    f1.append(sol_f.y[0])
    f3.append(sol_f.y[1])
    l_sol.append(sol_f.t)

#%%
'''
num=1110  ## here fraction starts to fuck off

fig1 = plt.figure()
for i in range(num,1314):
    plt.plot(l_sol[i],f3[i])
    
fig2 = plt.figure()
for i in range(num):
    plt.plot(l_sol[i],f3[i])
    
fig3  = plt.figure()
for i in range(len(l_sol)):
    if i < num:
        #plt.plot(get_r[i](l_sol[i])*np.sin(get_th[i](l_sol[i]))/rb[0][0],\
         #        get_r[i](l_sol[i])*np.cos(get_th[i](l_sol[i]))/rb[0][0],'r')
        pass
    else:
        plt.plot(get_r[i](l_sol[i])*np.sin(get_th[i](l_sol[i]))/rb[0][0],\
                 get_r[i](l_sol[i])*np.cos(get_th[i](l_sol[i]))/rb[0][0],'b')
            
plt.contourf(X,Z,T,200)
circle1 = plt.Circle((0, 0), 1, color='k')
plt.gca().add_patch(circle1)
plt.colorbar()
'''
#%%
### FIND DATA POINTS IN r-theta AND USE DELAUNEY TRIANGULATION ###
r_sol = []
th_sol = []

for i in range(len(l_sol)): 
    r_sol.append(get_r[i](l_sol[i]))
    th_sol.append(get_th[i](l_sol[i]))

# input of Delauney are coordinate points in the form (x,z) since we want cartesian grid
points_list = []
for i in range(len(r_sol)):
    for j in range(len(r_sol[i])):
        rad = r_sol[i][j]/rb[0][0]
        ang = th_sol[i][j]
        
        coord = [rad*np.sin(ang),rad*np.cos(ang)]
        points_list.append(coord)

points = np.asarray(points_list)

## put values of f3 in a single list s.t. they correspond to their coordinates
## take the LOGARITHM 
logf3_list = []
for i in range(len(f3)):
    for j in range(len(f3[i])):
        frac = np.log10(f3[i][j])
        logf3_list.append(frac)
logf3_values = np.asarray(logf3_list)

np.save('f3_coords',points)
np.save('logf3_values',logf3_values)

tri = Delaunay(points)   # does the triangulation
get_logf3 = inter.LinearNDInterpolator(tri,logf3_values)

#%%
#for i in range(len(r_sol)-1):
 #   plt.plot(r_sol[i]*np.sin(th_sol[i])/rb[0][0],r_sol[i]*np.cos(th_sol[i])/rb[0][0],'r')
    
plt.contourf(X,Z,10**get_logf3(X,Z),200)
circle1 = plt.Circle((0, 0), 1, color='k')
plt.gca().add_patch(circle1)
plt.colorbar()

#%%
x = np.linspace(1.03,8,200)
y = np.linspace(-14,14,500)

x_grid, y_grid = np.meshgrid(x,y)

fig1 = plt.figure()
plt.contourf(x_grid,y_grid,10**get_logf3(x_grid,y_grid),200)
plt.colorbar()

#%%
#################################################################################################
###
### TREAT AS A TWO LEVEL ATOM 2S-2P WITH TRANSITION AT 10830ANG
### TRANSITION WIDTH GIVEN BY UNCERTAINTY BUT SHOULD BE LOW SINCE TIMELIFE IS LOW
### MAXWELL-BOLTZMANN INTEGRATED PROJECTED ALONG LINE OF SIGHT GIVES GAUSSIAN IN THE REST FRAME
### USE HELIUM MASS -> VELOCITIES SHOULD BE ~4 LOWER THAN AVERAGE THERMAL FOR H
### GAUSSAIN PROFILE IS RED/BLUE SHIFTED BY BULK VELOCITY
### COMPARE WIDTH, SHIFT AND HEIGHT OF THE LINE
### 
##################################################################################################