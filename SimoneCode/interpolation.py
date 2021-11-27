"""
Created on Sat Oct 23 14:25:56 2021
@author: Simone Di Giampasquale
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate as inter
from scipy.integrate import solve_ivp as ivp
from mpl_toolkits.mplot3d import Axes3D


#%%
'''
define two r-theta grids
'''

r_max = 30
n_step = 300
r_offset = (r_max/n_step)/2 
t_offset = (np.pi/(3*n_step))/2

r1 = np.linspace(1, r_max, n_step)
t1 = np.linspace(0, 10, 6*n_step)*np.pi

r2 = r1 + r_offset
t2 = t1 + t_offset

rax1, tax1 = np.meshgrid(r1, t1)  #grid for u_theta
rax2, tax2 = np.meshgrid(r2, t2)  #grid for u_r offset in both directions wrt to grid for u_r

#plt.plot(rax1,tax1,'.', rax2, tax2, 'x', color='k')
#plt.show()

def cart2pol(x, y):
    '''
    Cartesians to sherical polars conversion
    '''
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(x,y)
    return(rho, phi)

def pol2cart(rho, phi):
    '''
    Spherical polars to cartesians conversion
    '''
    x = rho * np.sin(phi)
    y = rho * np.cos(phi)
    return(x, y)

xax1,yax1 = pol2cart(rax1, tax1)
xax2,yax2 = pol2cart(rax2, tax2)

#plt.plot(xax1,yax1,'.')
#plt.plot(xax2,yax2,'o')

#%%

def test_vr(r,t):
    #vr = (r**-0.5)*np.exp((t**2)/2)
    #if type(r) == float: 
    #    vr = 1
    #else:
    #    vr = np.full(r.shape,1)
    vr = r**-0.5
    return vr

def test_vt(r,t):
    #vt = np.exp(r)
    vt = np.cos(t)
    #vt = np.sin(t)
    #if type(r) == float: 
    #    vt = 1
    #else:
    #    vt = np.full(r.shape,1)
    #vt = t
    #vt = r**-0.25
    return vt

u_r = test_vr(rax1,tax1)
u_t = test_vt(rax1,tax1)

get_ur = inter.RectBivariateSpline(r1, t1, u_r.T)  #interpolator for u_r
get_ut = inter.RectBivariateSpline(r1, t1, u_t.T)  #interpolator for u_theta


#%%

R = (1, r_max)
TH = np.linspace(0,2,100)*np.pi
r_eval = np.linspace(1,30,200) 

### solve ODE with analytic functions ###
def rhs_r_inv(rad,theta,v_r,v_t):
    '''
    right hand side of our differential equation
    dtheta/dr = u_t/(r*u_r)
    '''
    vel_r = v_r(rad,theta)
    vel_t = v_t(rad,theta)
    
    f_r = vel_t/(vel_r*rad)
        
    return f_r

#plot streamlines from known function
sol_R = ivp(rhs_r_inv, R, TH, args=(test_vr,test_vt), t_eval=r_eval)

Radii = sol_R.t
Angles = sol_R.y

for i in range(len(Angles)-1):
    x_R = Radii*np.sin(Angles[i])
    y_R = Radii*np.cos(Angles[i])
    
    plt.plot(x_R, y_R, color='r')
    

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

sol_interp_R = ivp(get_rhs, R, TH, args=(get_ur,get_ut), t_eval=r_eval)

#plot streamlines from interpolated function
Radii_int = sol_interp_R.t
Angles_int = sol_interp_R.y

for i in range(len(Angles_int)-1):
    x_R_int = Radii_int*np.sin(Angles_int[i])
    y_R_int = Radii_int*np.cos(Angles_int[i])
    
    plt.plot(x_R_int, y_R_int,color='b', lw = 0.5)
    
#plt.xlim(-3,3)
#plt.ylim(-3,3)
plt.show()

#%%

#overlap known function and interpolator on a r-theta plane
#for both u_r and u_theta

fine_r = np.linspace(1,30,200)
fine_t = np.linspace(0,2,200)*np.pi

R_grid, TH_grid = np.meshgrid(fine_r,fine_t)

### u_theta ###
fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
ax.plot_surface(R_grid, TH_grid, get_ut(R_grid,TH_grid,grid=False))
ax.plot_surface(R_grid, TH_grid, test_vt(R_grid,TH_grid),color = 'k')
plt.xlabel('r')
plt.ylabel('theta')
plt.show()

### u_r ###
fig1 = plt.figure()
ax = Axes3D(fig1, auto_add_to_figure=False)
fig1.add_axes(ax)
ax.plot_surface(R_grid, TH_grid, get_ur(R_grid,TH_grid,grid=False))
ax.plot_surface(R_grid, TH_grid, test_vr(R_grid,TH_grid),color = 'k')
plt.xlabel('r')
plt.ylabel('theta')
plt.show()

#%%

#plot relative difference between known function and interpolator

### u_theta ###
ut_rel_diff = 100*(test_vt(R_grid,TH_grid) - get_ut(R_grid,TH_grid,grid=False))/test_vt(R_grid,TH_grid)

fig3 = plt.figure()
ax = Axes3D(fig3, auto_add_to_figure=False)
fig3.add_axes(ax)
ax.plot_surface(R_grid, TH_grid, ut_rel_diff)
plt.xlabel('r')
plt.ylabel('theta')
plt.show()

### u_r ###
ur_rel_diff = 100*(test_vr(R_grid,TH_grid) - get_ur(R_grid,TH_grid,grid=False))/test_vr(R_grid,TH_grid)

fig4 = plt.figure()
ax = Axes3D(fig4, auto_add_to_figure=False)
fig4.add_axes(ax)
ax.plot_surface(R_grid, TH_grid, ur_rel_diff, color = 'k')
plt.xlabel('r')
plt.ylabel('theta')
plt.show()


#%%

#plot streamlines in r-theta plane from the known function and the interpolator

for i in range(len(Angles)-1):
    plt.plot(Radii, Angles[i], color='r')
    
for i in range(len(Angles_int)-1):
    plt.plot(Radii_int, Angles_int[i], color='k', lw = 0.5)
    
#plt.xlim(0,3)
plt.show()
