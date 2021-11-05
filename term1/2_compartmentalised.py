
"""
testing 
streamlines of interpolation from discrete grid points of known velocity functions 
against 
streamlines of known velocity functions
"""

import numpy as np
from scipy.integrate import solve_ivp as ivp
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline as spline
from mpl_toolkits.mplot3d import Axes3D


plt.style.use("cool-style.mplstyle")

def u_r(r, theta):
    return 1

def u_theta(r, theta):
    return 1

def radial(r, theta, f_r, f_t):
     return f_t(r, theta)/(f_r(r, theta)*r)




thetas = np.linspace(0, 2, 100)*np.pi

r_lim = 20
radii = np.linspace(1, r_lim, r_lim*10)
rspan = np.array([radii[0], radii[-1]])



sol = ivp(radial, rspan, thetas, t_eval = radii, args = (u_r, u_theta))


def cart_y(r, theta):
    return r*np.cos(theta)
def cart_x(r,theta):
    return r*np.sin(theta)
def plot_cart(r, theta, colour, lw = 2):
    plt.plot(cart_x(r, theta), cart_y(r, theta), color = colour, lw = lw)
     
for ys in sol.y:
    plot_cart(sol.t, ys, "red", lw = 1.5)
               
    
size = r_lim
''''''
plt.ylim(-size,size)
plt.xlim(-size,size)

plt.show()


#%%

#build grid of r against theta                                          X

#find u_r and u_theta at grid edge centres                              X

#interpolate in 2D for rough functional forms for u_r and u_theta       .

#compare!


grid_r = np.linspace(1, 20, 5)
grid_t = np.linspace(0, 2, 5) * np.pi



X,Y = np.meshgrid(grid_r, grid_t)
grid_t_centres = np.array(grid_t + (grid_t[1] - grid_t[0])/2)[:len(grid_t)-1]
grid_r_centres = np.array(grid_r + (grid_r[1] - grid_r[0])/2)[:len(grid_r)-1]

centres_x, centres_y = np.meshgrid(grid_r_centres, grid_t_centres)
centres_r, grid_t_y = np.meshgrid(grid_r_centres, grid_t)
grid_r_x, centres_t = np.meshgrid(grid_r, grid_t_centres)


plt.scatter(centres_r, grid_t_y)
plt.scatter(grid_r_x, centres_t)
#plt.scatter(centres_x, centres_y)

v_rad_discrete = []
v_the_discrete = []

for const_r in grid_r:
    v_rad_0 = []
    for const_t in grid_t_centres:
        v_rad = round(u_r(const_r, const_t), 2)
        v_rad_0.append(v_rad)
        plt.text(const_r, const_t, "u_r is " + str(v_rad))
    v_rad_discrete.append(v_rad_0)
        

for const_t in grid_t:
    v_the_0 = []
    for const_r in grid_r_centres:
        v_theta = round(u_theta(const_r, const_t), 2)
        v_the_0.append(v_theta)
        #plt.text(const_r, const_t, "u_theta is " + str(v_theta))
    v_the_discrete.append(v_the_0)
     
plt.plot(X,Y, color = "grey", lw = 0.5)
for n in range(len(X)):
    plt.plot(X[n], Y[n], color = "grey", lw = 0.5)
       
    
plt.xlabel("r")
plt.ylabel("theta")
plt.show()


#%%
fx = spline(grid_r, grid_t_centres, v_rad_discrete)
#fy = spline(grid_t, grid_r_centres, v_the_discrete)

x2 = np.linspace(1, 20, 200)
y2 = np.linspace(0, 6, 200)
X2, Y2 = np.meshgrid(x2, y2)
Z2 = fx(x2, y2)

plt.plot(X2, Z2)
plt.show()

plt.plot(Y2, Z2)
plt.show()

sol = ivp(radial, rspan, thetas, t_eval = radii, args = (fx, u_theta))



