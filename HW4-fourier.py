from astropy.io import fits
hdul = fits.open("tic0010891640.fits")
times = hdul[1].data['times']
fluxes = hdul[1].data['fluxes']
ferrs = hdul[1].data['ferrs']

from matplotlib import pyplot as plt
plt.scatter(times,fluxes)
plt.xlim([1683,1690])
#plt.show()

#for every entry in fluxes, if the next entry is more than one timestep away,
#interpolate the last two points to guess the next point



from numpy import zeros,exp,pi
def dft(y):
    N = len(y)
    c = zeros(N//2+1,complex)
    for k in range(N//2+1):
        for n in range(N):
            c[k] += y[n]*exp(-2j*pi*k*n/N)
    return c

#c = dft(fluxes[:200])
#plt.plot(abs(c))
#plt.show()
