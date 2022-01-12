"""
Created on Sat Dec 04 21:38:15 2021
@author: Simone Di Giampasquale
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import interpolate as inter
from scipy.integrate import solve_ivp as ivp

#%%

### SOLVE COUPLED ODES OF THE FORM dy/dt = f(t,y) WITH y, f VECTORS ###

### try solve  dx/dt = x - xy/2       with x(0)=1 y(0)=2
###            dy/dt = -3y/4 + xy/4

# (1) DEFINE RHS FUNCTION THAT TAKES AN ARRAY AS y VARIABLE
def f(t,y):
    y1 = y[0]  # corresponds to x above
    y2 = y[1]  # corresponds to y above
    
    rhs1 = y1 - 0.5*y1*y2
    rhs2 = -0.75*y2 + 0.25*y1*y2
    
    rhs = np.array([rhs1,rhs2])
    return rhs

# (2) DEFINE INITIAL CONDITIONS y0 AS AN ARRAY IN THE VARIABLE SAME ORDER
y0 = np.array([1,2])

# (3) DEFINE t_span AND t_eval IN THE SAME WAY AS BEFORE
time = (0,20)
times = np.linspace(0,20,200)

# (4) SOLVE USING solve_ivp WHERE THE .y OUTPUT WILL BE AN ARRAY FOR EACH VARIABLE
sol = ivp(f, time, y0, t_eval=times)

t = sol.t
x_sol = sol.y[0]
y_sol = sol.y[1]

# (5) PLOT x AND y VS t
plt.figure()
plt.plot(t,x_sol,'k')
plt.plot(t,y_sol,'r')

plt.show()

#%%
### Set up eqn for helium fraction in simplified scenario

T = 1e4                                    # Temperature
ne = 1e-16/(1.00784*1.66e-24)              # electron number density
n0 = 0.001*ne                              # neutral hydrogen number density
F1 = ((10e3)/(24.6*1.602e-12))*(7.82e-18)  # photoionisation rate of 1S
F3 = ((10e3)/(4.8*1.602e-12))*(7.82e-18)   # photoionisation rate of 2S triplet
u = 1e6                                    # absolute value of velocity
a1 = 1.54e-13                              # recombination rate of 1S 
a3 = 1.49e-14                              # recombination rate of 2S triplet
A31 = 1.272e-4                             # radiative de-excitation rate 2S tri -> 1S
q13a = 4.5e-20                             # collisional excitation rate 1S -> 2S tri
q31a = 2.6e-8                              # collisional excitation rate 2S tri -> 2S sing
q31b = 4.0e-9                              # collisional excitation rate 2S tri -> 2P sing
Q31 = 5e-10                                # collisional de-exciration rate 2S tri -> 1S

A = (-ne*a1-F1-ne*q13a)
B = (-ne*a1+A31+ne*q31a+ne*q31b+n0*Q31)
C = (-ne*a3+ne*q13a)
D = (-ne*a3-A31-F3-ne*q31a-ne*q31b-n0*Q31)

def g(l,frac):
    n1 = frac[0]
    n3 = frac[1]
    
    g1 = (ne*a1 + A*n1 + B*n3)/u
    g2 = (ne*a3 + C*n1 + D*n3)/u
    
    rhs = np.array([g1,g2])
    return rhs

f0 = np.array([1,0])
l0 = (1e10,5e10)
l_eval = np.linspace(l0[0],l0[-1],200)

f_sol = ivp(g, l0, f0, method='LSODA', atol=1e-10)

l = f_sol.t
n1_sol = f_sol.y[0]
n3_sol = f_sol.y[1]

plt.figure()
#plt.plot(l,f1_sol,'k')
plt.plot(l,n3_sol,'b')

plt.show()