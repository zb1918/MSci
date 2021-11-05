"""
Created on Sat Oct 9 11:45:16 2021
@author: Simone Di Giampasquale
"""

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp as ivp

x1 = np.arange(-10, 11, 0.5, dtype=float)
y1 = x1

xax, yax = np.meshgrid(x1, y1)

#u_x = xax
#u_y = yax
#plt.quiver(x,y,u_x,u_y)
#plt.show()

def v1_x(x,y):
    vx = x
    return vx

def v1_y(x,y):
    vy = y
    return vy

def rhs(x,y,u_x,u_y):
    vel_x = u_x(x,y)
    vel_y = u_y(x,y)
    
    f = vel_y/vel_x
        
    return f

interval = (x1[0], x1[len(x1)-1])

sol = ivp(rhs, interval, np.arange(-100,1,1), args=(v1_x,v1_y), max_step = 0.5)

a = sol.t
b = sol.y

for i in range(len(b)-1):
    plt.plot(a, b[i])
    
plt.show()

'''
fig = plt.figure()
ax0 = fig.add_subplot(1,1,1)

one = np.full((len(x1),len(x1)), 1)
v_x = xax
v_y = yax


ax0.streamplot(xax, yax, v_x, v_y)
plt.show()
'''

#%%

def u_r(R,T):
    return 1

def u_t(R,T):
    return 1

def rhs_r(theta,rad):
    
    vel_r = 1
    vel_t = 1
    
    f_r = rad*vel_r/vel_t
        
    return f_r

r = np.sqrt((x1**2)+(y1**2))
th = (0, 2*np.pi)

sol_r = ivp(rhs_r, th, r, max_step= np.pi/24)

radii = sol_r.y
angles = sol_r.t

for i in range(len(radii)-1):
    x_r = radii[i]*np.cos(angles)
    y_r = radii[i]*np.sin(angles)
    
    plt.plot(x_r, y_r)
    
plt.xlim(-4,4)
plt.ylim(-4,4)    
#%%

def rhs_r_inv(rad,theta):
    
    vel_r = 1
    vel_t = 1
    
    f_r = vel_t/(vel_r*rad)
        
    return f_r

R = (1,10)
TH = np.linspace(0,2*np.pi, 48)

sol_R = ivp(rhs_r_inv, R, TH, max_step= 0.1)
#sol  = ivp(f,    t_span, y0, t_eval)
Radii = sol_R.t
Angles = sol_R.y

for i in range(len(Angles)-1):
    x_R = Radii*np.sin(Angles[i])
    y_R = Radii*np.cos(Angles[i])
    
    plt.plot(x_R, y_R)
    
#%%