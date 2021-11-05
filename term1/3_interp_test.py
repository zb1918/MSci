import numpy as np
from scipy.integrate import solve_ivp as ivp
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline as spline

plt.style.use("cool-style.mplstyle")
plt.subplots(1,2, sharex = True, sharey = True)

ax1 = plt.subplot(1,2,1)
def u_r(r, theta):
    return r**2

def u_theta(r, theta):
    return 1

def radial(r, theta, f_r, f_t):
     return f_t(r, theta)/(f_r(r, theta)*r)



thetas = np.linspace(0, 2, 100)*np.pi

r_lim = 20
radii = np.linspace(1, r_lim, r_lim*10)
rspan = np.array([radii[0], radii[-1]])

coarse_r = np.linspace(1, 4, 5)
coarse_t = np.linspace(0, 2, 10)*np.pi

v_rad_discrete = []
v_the_discrete = []

for r in coarse_r:
    v_rad_0 = []
    v_the_0 = []
    for t in coarse_t:
        v_rad_0.append(u_r(r, t))
        v_the_0.append(u_theta(r, t))
    v_rad_discrete.append(v_rad_0)
    v_the_discrete.append(v_the_0)
    
""" """

fx = spline(coarse_r, coarse_t, v_rad_discrete)
fy = spline(coarse_r, coarse_t, v_the_discrete)

sol = ivp(radial, rspan, thetas, t_eval = radii, args = (fx, fy))


def cart_y(r, theta):
    return r*np.cos(theta)
def cart_x(r,theta):
    return r*np.sin(theta)
def plot_cart(r, theta, colour, lw = 2):
    plt.plot(cart_x(r, theta), cart_y(r, theta), color = colour, lw = lw)
   

for ys in sol.y:
    plot_cart(sol.t, ys, "blue", lw = 1.5)
               
    
size = r_lim

plt.ylim(-size,size)
plt.xlim(-size,size)
plt.title("interpolating set of points")



#plt.show()
#ax2 = plt.subplot(1,2,2, sharex = ax1, sharey = ax1)


sol = ivp(radial, rspan, thetas, t_eval = radii, args = (u_r, u_theta))

     
for ys in sol.y:
    plot_cart(sol.t, ys, "red", lw = 1.5)
               
    
size = r_lim
''''''
plt.ylim(-size,size)
plt.xlim(-size,size)
plt.title("solving ODE for velocity functions")
plt.show()

''''''