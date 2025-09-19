#Show that the distance from the earth to the L1 Lagrange point satisfies
# (G*M)/R - (G*m)/(R-r)**2 = omega**2 * r

from astropy import constants
def f(r):
    return (constants.G.value*constants.EMConstant.value)/3.844e8 - (constants.G.value*7.348e22)/(3.844e8 - r)**2 - 2.662e-6**2*r

