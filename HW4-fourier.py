from astropy.io import fits
hdul = fits.open("tic0010891640.fits")
times = hdul[1].data['times']
fluxes = hdul[1].data['fluxes']
ferrs = hdul[1].data['ferrs']

from matplotlib import pyplot as plt
plt.scatter(times,fluxes)
plt.xlim([1683,1690])
plt.show()

#for every entry in fluxes, if the next entry is more than one timestep away,
#interpolate the last two points to guess the next point

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Create uniform grid
x_uniform = np.linspace(times[:100].min(), times[:100].max())

# Interpolate
interp_func = interp1d(times[:100], fluxes[:100], kind='cubic')
y_uniform = interp_func(x_uniform)

# Plot to check
plt.plot(times[:100], fluxes[:100], 'o', label='Original')
plt.plot(x_uniform, y_uniform, '-', label='Interpolated')
plt.legend()
plt.show()

# Now y_uniform is ready for FFT
N = len(y_uniform)
dt = x_uniform[1] - x_uniform[0]  # Assuming uniform spacing
fft_result = np.fft.fft(y_uniform)
frequencies = np.fft.fftfreq(N, dt)

# Removing the 0.0 frequency component
fft_result[0] = 0
plt.plot(frequencies[:N//2], np.abs(fft_result)[:N//2])  # Plot only positive frequencies
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.title('FFT of Interpolated Signal (0 freq removed)')
plt.show()

