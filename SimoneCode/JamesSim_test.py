"""
Created on Sat Nov 13 20:49:33 2021
@author: Simone Di Giampasquale
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import interpolate as inter
from scipy.integrate import solve_ivp as ivp
from scipy.optimize import curve_fit
#%matplotlib inline

#%%
# loads dictionary of hydro variables
# r - is b grid for r
# th - is the b grid for th
# vr - is the velocity ALREADY averaged to the b grid
# vth is the velocity ALREADY averaged to the b grid
# D is the mass density 

hydro_sim = loadmat("pure_HD.mat")

#%%

rb = hydro_sim['r']  #1D array of radius for grid
thb = hydro_sim['th']  #1D array of theta for grid

vrb = hydro_sim['vr']  #u_r
vthb = hydro_sim['vth']  #u_theta
D = hydro_sim['D']  #mass density
U = hydro_sim['U']  #internal energy pu volume
Xe = hydro_sim['ne']  #electron fraction ie fraction of ionised hydrogen
Xh = 1-Xe  #fraction of non ionised hydrogen


#scale radius by planet radius
radii = rb/rb[0]  
rax,tax = np.meshgrid(radii,thb)
#set radius grid in log
r_log = np.log10(radii)

X = np.outer(radii,np.sin(thb))
Z = np.outer(radii,np.cos(thb))

#%%
## Grid is setup so the z-axis points to the star

plt.contourf(X,Z,np.log10(D),100)  #plot colourmap of densty
c = plt.colorbar()
c.set_label(r"\log_{10}(Density) [g/cm3]")

#%%
#plot velocity on a semi-log grid
rax_log, tax = np.meshgrid(r_log,thb)

f1 = plt.figure()
plt.contourf(rax_log,tax,vthb,100)
plt.xlabel('log10(r)')
plt.ylabel('theta')
c = plt.colorbar()


#%%
#set velocities in logscale??? why?? why not??

#define sign matrices to track the sign for after taking the log of the absolute value
sign_vth = vthb/np.absolute(vthb)
sign_vr = vrb/np.absolute(vrb)

vr_log = sign_vr*np.log10(np.absolute(vrb))
vth_log = sign_vth*np.log10(np.absolute(vthb))

### need to interpolate the signs as well for when we interpolate

plt.contourf(rax_log,tax,vth_log,500)
plt.xlabel('log10(r/r_P)')
plt.ylabel('theta')
c = plt.colorbar()

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

#%%
#interpolate u_r, u_t on a logscale grid in r

get_ur_log = inter.RectBivariateSpline(r_log,thb,vrb.T)
get_ut_log = inter.RectBivariateSpline(r_log,thb,vthb.T)

#plot on a finer grid
fine_r_log = np.linspace(r_log[0],r_log[127],200)
fine_th = np.linspace(0,1,200)*np.pi

fine_rax_log, fine_tax = np.meshgrid(fine_r_log,fine_th)
int_uth_log = get_ut_log(fine_rax_log,fine_tax, grid = False)

f2 = plt.figure()
plt.contourf(fine_rax_log,fine_tax, int_uth_log,100)
plt.xlabel('log10(r/r_P)')
plt.ylabel('theta')
c = plt.colorbar()

### SOLVE IN r LOGSPACE??? ###

R_log = (r_log[0][0],r_log[-1][0])
Th = np.linspace(0,1,50)*np.pi 

rlim_log = r_log[-1][0]
r_eval_log = np.linspace(0,rlim_log,200)

#sol_log = ivp(get_rhs, R_log, Th, args=(get_ur_log,get_ut_log), t_eval=r_eval_log)

#%%
### NOT IN LOGSPACE!!! ###

get_vr = inter.RectBivariateSpline(radii.T[0], thb.T[0], vrb)
get_vt = inter.RectBivariateSpline(radii.T[0], thb.T[0], vthb)

R = (1.02,radii[-1][0])
Th = np.linspace(0,0.5,30)*np.pi 
rlim = radii[-1][0]
r_eval = np.linspace(1.02,rlim,500)

sol = ivp(get_rhs, R, Th, args=(get_vr,get_vt), t_eval=r_eval)

Radii = sol.t
Angles = sol.y

for i in range(len(Angles)-1):
    x_R = Radii*np.sin(Angles[i])
    y_R = Radii*np.cos(Angles[i])
    
    plt.plot(x_R, y_R, color='k',lw = 0.5)

plt.xlabel('r/r_P')
plt.ylabel('r/r_P')

#%%

#solving methods:
# (1) going in from every cell in the grid not in the shadowed region
# (2) going out from ~1.02 (??) starting at angles not in the shadowed region 
#     (different starting angle as radii grow)
#


#PLOT TEMPERATURE AS A FUNCTION OF (RADIUS-R_P) AND CHECK WHERE THE COLD REGION ENDS
#START THE STREAMLINES FROM A BIT BEFORE THAT, CHECKING IN THE NEIGHBOURHOOD 
#IF THAT DOES NOT CHANGE MUCH

#%%
### find temperature profile

g = 5/3  #gamma for monoatomic gas
P = (g-1)*U  #pressure
mu = 1.00784*1.66e-24  #mean molecular mass
k = 1.380649e-16 #boltzmann constant in cgs units
T = P*mu/(D*k)  #temperature in cgs units

#plt.plot(r_log, np.log10(T.T[0]))
#plt.plot(radii,T.T[50])
#plt.xlim(0,0.01)

plt.contourf(rax,tax,np.log10(T).T,100)
plt.xlim(1,1.1)
#plt.ylim(0,1)
c=plt.colorbar()

#%%
#1D interpolation of the temperature at theta=0
tck = inter.splrep(radii.T[0], T.T[0])
fine_r = np.linspace(1,1.05,200)
T_new = inter.splev(fine_r,tck, der = 0)

#plt.plot(fine_r,T_new)

#2D interpolation of T in T LOGSPACE
get_T_log = inter.RectBivariateSpline(radii.T[0],thb.T[0],np.log10(T))

fine_rax, fine_tax = np.meshgrid(fine_r,fine_th)
T_int_log = get_T_log(fine_rax,fine_tax, grid = False)
plt.contourf(fine_rax,fine_tax,(T_int_log),200)
X_fine = np.outer(fine_r,np.sin(fine_th))
Z_fine = np.outer(fine_r,np.cos(fine_th))
#plt.contourf(X_fine,Z_fine,T_int_log.T,100)
#c = plt.colorbar()

for i in range(199):
    plt.plot(fine_r,T_int_log[i],color = 'k')
for j in range(127):
    plt.plot(radii.T[0],np.log10(T.T[j]),color='b')
    plt.xlim(1,1.05)

#%%
#set a different initial radius for each temperature chunk

init_R = []
for i in range(len(thb)): 
    if (i>=0) and (i<=18) :
        R_in = 1.017        
    elif (i>=19) and (i<=40):
        R_in = 1.019        
    elif (i>=41) and (i<=51):
        R_in = 1.022        
    elif (i>=52) and (i<=57):
        R_in = 1.024        
    elif (i>=58) and (i<=61):
        R_in = 1.026
    elif (i>=62) and (i<=65):
        R_in = 1.029
    elif (i>=66) and (i<=71):
        R_in = 1.031
    else:
        R_in = 1.033        
    init_R.append(R_in)
    
#%%
### define 8 different angulars ranges with different starting radii ###
### for each temperature chunk ###

init_r = np.array([1.017,1.019,1.022,1.024,1.026,1.029,1.031,1.033])
init_th = np.array([0,0.5,1,1.285,1.43,1.53,1.63,1.77])

sol_rad = []
sol_ang = []

### solve the ODE by looping through the different ranges and append the reuslts ### 
for i in range(len(init_r)-1):
    R = (init_r[i],rlim)
    thetas = np.arange(init_th[i],init_th[i+1],0.05)
    reval = np.linspace(init_r[i],rlim,500)
    
    sol = ivp(get_rhs, R, thetas, args=(get_vr,get_vt), t_eval = reval)
    sol_rad.append(sol.t)
    sol_ang.append(sol.y)

### turn solutions into single arrays ###
sol_rad = np.vstack(sol_rad)
sol_ang = np.vstack(sol_ang)

#plot solutions
for j in range(7):
    s = 5*(j-1)
    steps = np.arange(s,s+5,1)
    for i in steps :
        x_R = sol_rad[j]*np.sin(sol_ang[i])
        y_R = sol_rad[j]*np.cos(sol_ang[i])
        
        plt.plot(x_R, y_R, color='r',lw = 0.5)

circle1 = plt.Circle((0, 0), 1, color='k')
plt.gca().add_patch(circle1)

plt.xlabel('r/r_P')
plt.ylabel('r/r_P')

#%%

def Th_r_min(rad,a,b):
    '''
    Piecewise function for angle as function of minimum radius to start at
    square root like at small radii
    
    '''
    f = []
    for i in range(len(rad)):
        if rad[i] <= 1.033:
            f_i = a*np.sqrt(np.abs(rad[i]-b))
        if rad[i] > 1.033:
            f_i = np.exp(rad[i]**4)
        f.append(f_i)
    fun = np.asarray(f)
    return fun

centr_th = np.array([0.25,0.75,1.143,1.358,1.48,1.58,1.7,1.77])
param, cov = curve_fit(Th_r_min, init_r, centr_th, p0=(14,1.0165))

plt.plot(init_r,init_th,'.')
plt.plot(init_R,thb)
plt.plot(fine_r,Th_r_min(fine_r,param[0],param[1]))

#define minimum radius as a function of theta, 
#accidentally did it the wrong way round above

#def r_min(theta,a,b):
    
    