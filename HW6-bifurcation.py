import numpy as np
import matplotlib.pyplot as plt

# ===== An Addendum to HW6-SpaceGarbage that runs the code 400 times and generates a bifurcation diagram according to the mass ratio
# ===== of two ball bearings in orbit around a rod. Run the code to generate the diagram, should take ~1min

G = 1.0
M = 10.0
L = 2.0

def rod_acc(x,y,z):
    r2 = x*x + y*y + z*z
    d = r2 * np.sqrt(r2 + 0.25*L*L)
    return -G*M*x/d, -G*M*y/d, -G*M*z/d

def f(s, m1, m2):
    x1,vx1,y1,vy1,z1,vz1, x2,vx2,y2,vy2,z2,vz2 = s
    axr1,ayr1,azr1 = rod_acc(x1,y1,z1)
    axr2,ayr2,azr2 = rod_acc(x2,y2,z2)

    dx,dy,dz = x2-x1, y2-y1, z2-z1
    r2 = dx*dx + dy*dy + dz*dz
    r3 = (r2+1e-6)**1.5

    ax12 =  G*m2*dx/r3
    ay12 =  G*m2*dy/r3
    az12 =  G*m2*dz/r3
    ax21 = -G*m1*dx/r3
    ay21 = -G*m1*dy/r3
    az21 = -G*m1*dz/r3

    return np.array([
        vx1, axr1+ax12,  vy1, ayr1+ay12,  vz1, azr1+az12,
        vx2, axr2+ax21,  vy2, ayr2+ay21,  vz2, azr2+az21
    ])

def rk4_step(s, h, m1, m2):
    k1 = h*f(s, m1, m2)
    k2 = h*f(s+0.5*k1, m1, m2)
    k3 = h*f(s+0.5*k2, m1, m2)
    k4 = h*f(s+k3,     m1, m2)
    return s + (k1+2*k2+2*k3+k4)/6

def escape_time(m1, m2, T_max=20.0, n_steps=6000, R_esc=5.0):
    h = T_max/n_steps
    s = np.array([
        1.0, 0.0,  0.2,  1.1,  0.0, 0.2,    # ball 1
       -1.0, 0.0, -0.3, -1.0,  0.5,-0.3])   # ball 2 (z offset)
    t = 0.0
    for _ in range(n_steps):
        x1,y1,z1 = s[0], s[2], s[4]
        x2,y2,z2 = s[6], s[8], s[10]
        if x1*x1+y1*y1+z1*z1 > R_esc**2 or x2*x2+y2*y2+z2*z2 > R_esc**2:
            return t
        s = rk4_step(s, h, m1, m2)
        t += h
    return T_max   # “no escape” within T_max

ratios = np.linspace(0.1, 10.0, 400)
times = []

for r in ratios:
    m1 = 1.0
    m2 = r*m1
    times.append(escape_time(m1, m2))

times = np.array(times)

plt.figure(figsize=(8,4))
plt.scatter(ratios, times, s=5)
plt.xlabel("mass ratio  m2/m1")
plt.ylabel("escape time  t_escape")
plt.title("Escape-time bifurcation diagram")
plt.ylim(0, times.max()*1.05)
plt.show()
