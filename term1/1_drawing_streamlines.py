from matplotlib import pyplot as plt
import numpy as np



r = np.linspace(1, 2, 100)
theta = np.linspace(0, 0.2, 100)*np.pi

circle_r = np.linspace(1, 1, 100)
full_theta = np.linspace(0, 2, 100)*np.pi



def cart_x(r, theta):
    return r*np.cos(theta)
def cart_y(r,theta):
    return r*np.sin(theta)

def plot_cart(r, theta):
    plt.plot(cart_x(r, theta), cart_y(r, theta))

offsets =np.linspace(0, 2*np.pi, 200)

for offset in offsets:
    plot_cart(r,theta+offset)


plot_cart(circle_r, full_theta)

plt.ylim(-2,2)
plt.xlim(-2,2)
plt.show()


#%%
