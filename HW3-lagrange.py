#Show that the distance from the earth to the L1 Lagrange point satisfies
# (G*M)/R - (G*m)/(R-r)**2 = omega**2 * r

from astropy import constants
from astropy import units as u

R = 3.844e8 * u.meter
m_moon = 7.348e22 * u.kilogram
omega = 2.662e-6 / u.second

def P(r):
    # r is a Quantity with length units
    return (constants.G*constants.M_earth)/(r**2) \
         - (constants.G*m_moon)/((R - r)**2) \
         - (omega**2)*r

def Pp(r):
    # derivative wrt r
    return (-2*constants.G*constants.M_earth)/(r**3) \
         - (2*constants.G*m_moon)/((R - r)**3) \
         - (omega**2)

# Newton solver with convergence & domain-safety
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

        # stay in the physical domain: 0 < r < R
        if not (0 * u.meter < r_new < R):
            raise ValueError(f"Iterate left domain (0, R): r_new={r_new}")

        if abs(step) < tol:
            return r_new
        r = r_new
    return r  # last iterate if not converged

# Try a sensible initial guess near L1
r0 = 3.2e8 * u.meter
root = newton_r(r0)
print(root)
