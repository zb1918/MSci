import numpy as np
import matplotlib.pyplot as plt

grid_r = np.linspace(1, 20, 5)
grid_t = np.linspace(0, 2, 5) * np.pi



X,Y = np.meshgrid(grid_r, grid_t)
grid_t_centres = np.array(grid_t + (grid_t[1] - grid_t[0])/2)[:len(grid_t)-1]
grid_r_centres = np.array(grid_r + (grid_r[1] - grid_r[0])/2)[:len(grid_r)-1]

centres_x, centres_y = np.meshgrid(grid_r_centres, grid_t_centres)
centres_r, grid_t_y = np.meshgrid(grid_r_centres, grid_t)
grid_r_x, centres_t = np.meshgrid(grid_r, grid_t_centres)



for i in range(len(grid_t_centres)):
    plt.axhline(centres_t[i][0], color = "red")
    plt.axhline(grid_t_y[i][0], color = "blue")
    
for j in range(len(grid_r_centres)):
    plt.axvline(grid_r_x[0][j], color = "red")
    plt.axvline(centres_r[0][j], color = "blue")
    
        
plt.scatter(centres_r, grid_t_y, label="r centres", color ="blue")
plt.scatter(grid_r_x, centres_t, label="t centres", color ="red")
plt.scatter(centres_x, centres_y, label="all centres", color = "black")

plt.legend()
