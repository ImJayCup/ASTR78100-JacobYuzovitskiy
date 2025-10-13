from astropy.io import fits
hdul = fits.open("tic0010891640.fits")
times = hdul[1].data['times']
fluxes = hdul[1].data['fluxes']
ferrs = hdul[1].data['ferrs']

from matplotlib import pyplot as plt

# Number of points to use
n_pts = 250

# Interpolate to uniform spacing for FFT
import numpy as np
from scipy.interpolate import interp1d
x_uniform = np.linspace(times[:n_pts].min(), times[:n_pts].max(), num=len(times[:n_pts]))

# Interpolate
interp_func = interp1d(times[:n_pts], fluxes[:n_pts], kind='cubic')
y_uniform = interp_func(x_uniform)

# Now y_uniform is ready for FFT
N = len(y_uniform)
dt = x_uniform[1] - x_uniform[0]  # Assuming uniform spacing
fft_result = np.fft.fft(y_uniform)
frequencies = np.fft.fftfreq(N, dt)

# Rebuild signal from first 10 Fourier components
reconstructed_signal = np.zeros_like(y_uniform).astype('complex128')
for i in range(1, 11):
    reconstructed_signal += (fft_result[i] * np.exp(2j * np.pi * frequencies[i] * x_uniform / N) +
                             fft_result[-i] * np.exp(-2j * np.pi * frequencies[i] * x_uniform / N)) / N
reconstructed_signal = reconstructed_signal.real  # Take the real part

# Removing the 0.0 frequency component
fft_result[0] = 0
fig, axs = plt.subplots(4, 1, figsize=(10, 8))  # no sharex

# time-domain panels
axs[0].scatter(times, fluxes, s=5)
axs[0].set_xlim(1683, 1690)

axs[1].plot(times[:n_pts], fluxes[:n_pts], 'o', label='Original')
axs[1].plot(x_uniform, y_uniform, '-', label='Interpolated')
axs[1].set_xlim(1683, 1690)
axs[1].legend()

# frequency-domain panel
axs[2].plot(frequencies[:N//2], np.abs(fft_result)[:N//2])
axs[2].set_xlabel('Frequency')

# plot reconstructed signal
axs[3].plot(x_uniform, reconstructed_signal, '-', label='Reconstructed Signal')

plt.tight_layout()
plt.show()