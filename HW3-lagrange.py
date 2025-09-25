#Show that the distance from the earth to the L1 Lagrange point satisfies
# (G*M)/R - (G*m)/(R-r)**2 = omega**2 * r

from astropy import constants
from astropy import units as u
from matplotlib import pyplot as plt
from math import cos, pi
from matplotlib.patches import Ellipse,Circle
plt.rcParams['hatch.color'] = 'green'

R = 3.844e8 * u.meter
m_moon = 7.348e22 * u.kilogram
omega = 2.662e-6 / u.second

### earth moon distance is modeled by d(t) = R*(1-0.0549*cos(2*pi*t/27.32))
def moonDist(t):
    return R*(1-0.0549*cos(2*pi*t/27.32))

def P(r):
    return (constants.G*constants.M_earth)/(r**2) \
         - (constants.G*m_moon)/((R - r)**2) \
         - (omega**2)*r

def Pp(r):
    return (-2*constants.G*constants.M_earth)/(r**3) \
         - (2*constants.G*m_moon)/((R - r)**3) \
         - (omega**2)

# Newton's Method
def newton_r(r0, max_iter=50, tol=1e-9*u.meter):
    r = r0.to(u.meter)
    for _ in range(max_iter):
        f = P(r)
        fp = Pp(r)
        # guard against division by zero / tiny derivative
        if abs(fp) == 0 * (1/u.second**2):
            raise ZeroDivisionError("Derivative vanished; try a different initial guess.")
        step = f / fp  # has units of meters
        r_new = r - step

        # only allow 0 < r < R
        if not (0 * u.meter < r_new < R):
            raise ValueError(f"Iterate left domain (0, R): r_new={r_new}")

        if abs(step) < tol:
            return r_new
        r = r_new
    return r  # last iterate if not converged

r0 = 3.2e8 * u.meter
root = newton_r(r0)
print(root)

fig,axs = plt.subplots(ncols=1,nrows=1)
axs.set_aspect(1.0)
axs.set_facecolor("black")
axs.add_patch(Ellipse((0,0),R.value*2,383800000*2,fill=False,ec="yellow",lw=0.5))
axs.add_patch(Ellipse((0,0),12756*1000,12756*1000,hatch="*"))
axs.add_patch(Ellipse((R.value,0),3474.8*1000,3474.8*1000,color="grey"))
axs.add_patch(Circle((0,0),root.value,fill=False,ec="white",lw=1,ls="--"))
axs.set_xlim([0,R.value])
axs.set_ylim([-R.value/2,R.value/2])
plt.show()