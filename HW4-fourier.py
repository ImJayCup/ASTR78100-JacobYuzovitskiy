# --- USAGE: Run HW-fourier.py to generate the fourier analysis and animation for tic0010891640.fits ---

# Open FITS file and extract data
from astropy.io import fits
hdul = fits.open("tic0010891640.fits")
times = hdul[1].data['times']
fluxes = hdul[1].data['fluxes']
ferrs = hdul[1].data['ferrs']

from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

# Detect large gaps in time-series data
dt = np.diff(times)
gap_threshold = 10 * np.nanmedian(dt)  # adjust if needed
gap_idx = np.where(dt > gap_threshold)[0]

# segment boundaries
starts = np.r_[0, gap_idx + 1]
ends = np.r_[gap_idx, len(times) - 1]

# choose the largest cluster by number of points
seg_lengths = ends - starts + 1
best_seg = np.argmax(seg_lengths)
i0, i1 = int(starts[best_seg]), int(ends[best_seg])

# use the largest cluster for analysis
t_seg = times[i0:i1+1]
y_seg = fluxes[i0:i1+1]

# --- Interpolate to uniform spacing for FFT (uses whole chosen segment) ---
x_uniform = np.linspace(t_seg.min(), t_seg.max(), num=len(t_seg))

# Interpolate (fallback to linear if not enough points for cubic)
kind = 'cubic' if len(t_seg) >= 4 else 'linear' #Did not know you could do this before!
interp_func = interp1d(t_seg, y_seg, kind=kind)
y_uniform = interp_func(x_uniform)

# Plug y_uniform into the FFT
N = len(y_uniform)
dt_uniform = x_uniform[1] - x_uniform[0]  # uniform spacing
fft_result = np.fft.fft(y_uniform)
frequencies = np.fft.fftfreq(N, dt_uniform)

# ----- IFFT Reconstruction -----
filtered_fft = np.zeros_like(fft_result)
filtered_fft[0] = fft_result[0]  # keep mean

# Keep the first 10 positive-frequency bins
# (skip k=0)
K = min(10, N//2 - 1)
for k in range(1, K + 1):
    filtered_fft[k]   = fft_result[k]
    filtered_fft[-k]  = fft_result[-k]

# Reconstruct in time domain from the filtered spectrum
reconstructed_signal = np.fft.ifft(filtered_fft).real

# Remove first signal only for plotting
fft_plot = fft_result.copy()
fft_plot[0] = 0

fig, axs = plt.subplots(4, 1, figsize=(10, 8))  # no sharex

# time-domain panels
axs[0].scatter(times, fluxes, s=5)
axs[0].set_xlim(t_seg.min(), t_seg.max())
axs[0].set_ylabel('Flux')
axs[0].set_title('Original Time Series with Gaps')

axs[1].plot(t_seg, y_seg, 'o', label='Original (cluster)')
axs[1].plot(x_uniform, y_uniform, '-', label='Interpolated')
axs[1].set_xlim(t_seg.min(), t_seg.max())
axs[1].legend()
axs[1].set_ylabel('Flux')
axs[1].set_title('Segmented and Interpolated Time Series')

# frequency-domain panel (half-spectrum)
axs[2].plot(frequencies[:N//2], np.abs(fft_plot)[:N//2])
axs[2].set_xlabel('Frequency')
axs[2].set_ylabel('Amplitude')
axs[2].set_title('FFT Amplitude Spectrum')

# plot reconstructed signal
axs[3].plot(x_uniform, reconstructed_signal, '-', label='Reconstructed (10 comps)')
axs[3].legend()
axs[3].set_xlabel('Time')
axs[3].set_ylabel('Flux')
axs[3].set_title('Reconstructed Time Series from Filtered FFT')

plt.tight_layout()
plt.show()

#Animation
from matplotlib import animation
from matplotlib.patches import Circle

#normalizing flux - wasn't working before I did this
flux = reconstructed_signal.astype(float)
fmin = np.min(flux)
fmax = np.max(flux)
if np.isclose(fmax - fmin, 0):
    fmax = fmin + 1e-9
f_norm = (flux - fmin) / (fmax - fmin)
f_norm = np.clip(f_norm, 0.0, 1.0)


# When f_norm is small (deep eclipse), separation should be small.
R = 1.0                 # star radius 
sep_min = 0.05          # near-total eclipse
sep_max = 3.0           # well separated (no overlap)
seps = sep_min + f_norm * (sep_max - sep_min)


# Downsampling
stride = 1
seps = seps[::stride]
t_anim = x_uniform[::stride]
flux_anim = flux[::stride]
f_norm_anim = f_norm[::stride]

# --- Figure with two panels: top animation, bottom light curve ---
fig = plt.figure(figsize=(8, 8))
gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.25)

ax_anim = fig.add_subplot(gs[0])
ax_lc   = fig.add_subplot(gs[1])

# ----- Top: eclipsing binary patches -----
ax_anim.set_aspect('equal', adjustable='box')
pad = sep_max/2 + R + 0.2
ax_anim.set_xlim(-pad, pad)
ax_anim.set_ylim(-1.5*R, 1.5*R)
ax_anim.set_xticks([])
ax_anim.set_yticks([])
ax_anim.set_title("Eclipsing Binary (driven by reconstructed IFFT)")

# Two stars as circles; we’ll move their centers each frame
star1 = Circle((-seps[0]/2, 0.0), R, fc='yellow', ec='k', lw=0.8, zorder=1)
star2 = Circle(( seps[0]/2, 0.0), R, fc='orange',    ec='k', lw=0.8, zorder=2)
ax_anim.add_patch(star1)
ax_anim.add_patch(star2)

# ----- Bottom: light curve with moving marker -----
ax_lc.plot(t_anim, flux_anim, '-', lw=1.0)
marker_lc, = ax_lc.plot([t_anim[0]], [flux_anim[0]], 'o')
ax_lc.set_xlabel("Time")
ax_lc.set_ylabel("Flux (reconstructed)")
ax_lc.grid(True, alpha=0.25)

# ----- Init & Update functions for FuncAnimation -----
def init():
    # place objects at first frame
    s0 = seps[0]
    star1.center = (-s0/2, 0.0)
    star2.center = ( s0/2, 0.0)
    marker_lc.set_data([t_anim[0]], [flux_anim[0]])
    return star1, star2, marker_lc

def update(i):
    s = seps[i]
    # Move centers symmetrically
    star1.center = (-s/2, 0.0)
    star2.center = ( s/2, 0.0)

    # Update light-curve marker
    marker_lc.set_data([t_anim[i]], [flux_anim[i]])

    # Optional: subtitle with time & flux
    fig.suptitle(f"t = {t_anim[i]:.5f}   flux = {flux_anim[i]:.5f}", y=0.98, fontsize=10)
    return star1, star2, marker_lc

# Create animation
fps = 360
interval_ms = 1000 / fps
ani = animation.FuncAnimation(
    fig, update, frames=len(seps), init_func=init,
    interval=interval_ms, blit=True
)

plt.show()
