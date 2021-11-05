import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

plt.style.use("cool-style.mplstyle")

 
x = np.linspace(0, 10, 200)

def y(x, a):
    return np.sin(x * a)

fig = plt.figure()
plt.subplots_adjust(bottom = .25)
ax = fig.subplots()

p, = ax.plot(x, y(x, a=2))

ax_slide = plt.axes([.25, .1, .65, .03])
val_a = Slider(ax_slide, "changing value",
               valmin=0.2,valmax=1.0,valinit=.6,valstep=.02)


def update(val):
    current_a = val_a.val
    p.set_ydata(y(x, current_a))
    fig.canvas.draw()

val_a.on_changed(update)
plt.show()

