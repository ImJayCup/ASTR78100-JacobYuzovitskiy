from astropy import constants
from astropy import units as u
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from math import cos, sin, pi
from matplotlib.patches import Ellipse, Circle
import numpy as np  # NEW
plt.rcParams['hatch.color'] = 'green'

R = 3.844e8 * u.meter
m_moon = 7.348e22 * u.kilogram
omega = 2.662e-6 / u.second

### earth moon distance is modeled by d(t) = R*(1-0.0549*cos(2*pi*t/27.32))
def moonDist(t):
    return R * (1 - 0.0549 * np.cos(2 * np.pi * t / 27.32))

t0 = 0

def P(r):
    return (constants.G*constants.M_earth)/(r**2) \
         - (constants.G*m_moon)/((moonDist(t0) - r)**2) \
         - (omega**2)*r

def Pp(r):
    return (-2*constants.G*constants.M_earth)/(r**3) \
         - (2*constants.G*m_moon)/((moonDist(t0) - r)**3) \
         - (omega**2)

# Newton's Method
def newton_r(r0, max_iter=50, tol=1e-9*u.meter):
    r = r0.to(u.meter)
    for _ in range(max_iter):
        f = P(r)
        fp = Pp(r)
        if abs(fp) == 0 * (1/u.second**2):
            raise ZeroDivisionError("Derivative vanished; try a different initial guess.")
        step = f / fp
        r_new = r - step
        if not (0 * u.meter < r_new < R):
            raise ValueError(f"Iterate left domain (0, R): r_new={r_new}")
        if abs(step) < tol:
            return r_new
        r = r_new
    return r

r0 = 3.2e8 * u.meter
root = newton_r(r0)
print(root)

### Plot
fig, axs = plt.subplots(ncols=1, nrows=1)
axs.set_aspect(1.0)
axs.set_facecolor("black")
fig.subplots_adjust(bottom=0.25)

axfreq = fig.add_axes([0.25, 0.1, 0.65, 0.03])
freq_slider = Slider(
    ax=axfreq,
    label='Time',
    valmin=0,
    valmax=27.32,
    valinit=t0,
)

# REPLACE: draw orbit as the same parametric curve used for the Moon's motion
tt = np.linspace(0, 27.32, 720)
theta_arr = 2*np.pi*tt/27.32
rr = moonDist(tt).to_value(u.meter)
x_orb = rr*np.cos(theta_arr)
y_orb = rr*np.sin(theta_arr)
orbit_line, = axs.plot(x_orb, y_orb, lw=0.5, color="yellow")  # replaces Ellipse(...)

### Earth
axs.add_patch(Ellipse((0,0), 12756*3000, 12756*3000, hatch="*"))

### Moon
moon = axs.add_patch(Ellipse((R.value,0), 3474.8*3000, 3474.8*3000, color="grey"))

### Lagrange pt
lag = axs.add_patch(Circle((0,0), root.value, fill=False, ec="white", lw=1, ls="--"))

# View window so entire path is visible
axs.set_xlim([-R.value, R.value])
axs.set_ylim([-R.value, R.value])

axs.set_title(f"Lagrange Point located at {newton_r(r0).to(u.km):.2f}")

def update(val):
    global t0
    t0 = val
    theta = 2*pi*t0/27.32
    d = moonDist(t0).value
    moon.center = (d * cos(theta), d * sin(theta))
    try:
        new_root = newton_r(r0)
        lag.set_radius(new_root.value)
        axs.set_title(f"Lagrange Point located at {new_root.to(u.km):.2f}")
    except Exception as e:
        axs.set_title(f"Lagrange computation error: {e}")
    fig.canvas.draw_idle()

freq_slider.on_changed(update)
update(t0)

plt.show()