import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

### ==== Space Garbage: now with more garbage! ====
### Change the masses of the TWO ball bearings m1 and m2 ###
G = 1.0
M = 10.0
L = 2.0
m1 = 1.0
m2 = 1.0
z_offset = 1  # increase to offset the balls

def rod_acc(x, y, z):
    r2 = x*x + y*y + z*z
    d = r2 * np.sqrt(r2 + 0.25*L*L)
    ax = -G*M*x/d
    ay = -G*M*y/d
    az = -G*M*z/d
    return ax, ay, az

### ==== system of 4 ODE ====
def f(s, t):
    (x1,vx1, y1,vy1, z1,vz1,
     x2,vx2, y2,vy2, z2,vz2) = s

    # gravity from rod
    axr1, ayr1, azr1 = rod_acc(x1,y1,z1)
    axr2, ayr2, azr2 = rod_acc(x2,y2,z2)

    # ball separation
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    r2 = dx*dx + dy*dy + dz*dz
    r3 = (r2 + 1e-6)**1.5

    # F 2on1
    ax12 =  G*m2*dx / r3
    ay12 =  G*m2*dy / r3
    az12 =  G*m2*dz / r3

    # F 1on2
    ax21 = -G*m1*dx / r3
    ay21 = -G*m1*dy / r3
    az21 = -G*m1*dz / r3

    # Acceleration
    ax1 = axr1 + ax12
    ay1 = ayr1 + ay12
    az1 = azr1 + az12

    ax2 = axr2 + ax21
    ay2 = ayr2 + ay21
    az2 = azr2 + az21

    return np.array([
        vx1, ax1, vy1, ay1, vz1, az1,
        vx2, ax2, vy2, ay2, vz2, az2
    ])

t0, tf = 0.0, 12.0
n = 3000
h = (tf - t0)/n

s = np.array([
    1.0,  0.0,  0.2,  1.1,  0.0, 0.2,  # ball 1
   -1.0,  0.0, -0.3, -1.0,  z_offset, -0.3 # ball 2
])

x1s=[]; y1s=[]; z1s=[]
x2s=[]; y2s=[]; z2s=[]
t = t0

# Runge Kutta 4
for _ in range(n):
    x1s.append(s[0]); y1s.append(s[2]); z1s.append(s[4])
    x2s.append(s[6]); y2s.append(s[8]); z2s.append(s[10])
    k1 = h*f(s,t)
    k2 = h*f(s+0.5*k1, t+0.5*h)
    k3 = h*f(s+0.5*k2, t+0.5*h)
    k4 = h*f(s+k3,     t+h)
    s += (k1+2*k2+2*k3+k4)/6
    t += h

x1s=np.array(x1s); y1s=np.array(y1s); z1s=np.array(z1s)
x2s=np.array(x2s); y2s=np.array(y2s); z2s=np.array(z2s)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

rod_z = np.linspace(-L/2, L/2, 20)
ax.plot(np.zeros_like(rod_z), np.zeros_like(rod_z), rod_z, color='yellow', lw=3)

traj1, = ax.plot(x1s, y1s, z1s, alpha=0.4)
ball1, = ax.plot([x1s[0]],[y1s[0]],[z1s[0]], 'o', color='red')

traj2, = ax.plot(x2s, y2s, z2s, alpha=0.4, color='cyan')
ball2, = ax.plot([x2s[0]],[y2s[0]],[z2s[0]], 'o', color='cyan')

ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
ax.set_box_aspect([1,1,1])

show2 = [True]

def toggle(event):
    show2[0] = not show2[0]
    traj2.set_visible(show2[0])
    ball2.set_visible(show2[0])
    plt.draw()

button_ax = plt.axes([0.8,0.05,0.12,0.05])
btn = Button(button_ax, "Ball 2")
btn.on_clicked(toggle)

def update(i):
    ball1.set_data(x1s[i], y1s[i])
    ball1.set_3d_properties(z1s[i])
    if show2[0]:
        ball2.set_data(x2s[i], y2s[i])
        ball2.set_3d_properties(z2s[i])
    return ball1, ball2

ani = FuncAnimation(fig, update, frames=len(x1s), interval=20, blit=False)

plt.show()
