"""
Monte Carlo: photon random walk in the Sun (Sun-only case)
- Electron density: n_e(r) = 2.5e26 * exp(-r / (0.096 R_sun))  [cm^-3], valid to 0.9 R_sun
- Mean free path:   l(r)   = 1 / (n_e * sigma_T)               [cm]
- Step length:      s ~ Exponential(mean=l)
- Direction:        isotropic; mu = cos(theta) ~ U[-1,1], phi ~ U[0,2pi)
- Escape when      |r| >= 0.9 R_sun

This file contains:
1) Batch calculation for escape path, steps, and time.
2) A time-lapse animation of a single photon path (3D projected to xy).
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import PillowWriter

# ======================== Constants (cgs) ========================
C = 2.99792458e10         # speed of light [cm/s]
SIGMA_T = 6.652e-25       # Thomson cross section [cm^2]
R_SUN = 696_340.0 * 1e5   # [cm]
R_EXIT = 0.9 * R_SUN      # escape radius [cm]

# ---- DEMO MODE: scale up mean free path so the photon escapes quickly (non-physical) ----
USE_DEMO_SCALING = True
MFP_DEMO = 1e10  # try 1e10–1e14; larger = faster escape on screen

# ======================== Physics profiles =======================
def n_e(r_cm: float) -> float:
    """Electron number density [cm^-3], valid up to 0.9 R_sun."""
    return 2.5e26 * math.exp(-r_cm / (0.096 * R_SUN))

def mean_free_path(r_cm: float) -> float:
    """Mean free path [cm]. DEMO scaling (if enabled) makes the animation fast but non-physical."""
    mfp = 1.0 / (n_e(r_cm) * SIGMA_T)
    if USE_DEMO_SCALING:
        mfp *= MFP_DEMO
    return mfp
# ======================== Radial sim (for stats) =================
@dataclass
class PhotonResult:
    steps: int
    total_path_cm: float
    escape_time_s: float
    escaped: bool

def draw_step_length(l_cm: float) -> float:
    u = 1.0 - random.random()  # in (0,1]
    return -l_cm * math.log(u)

def draw_mu_isotropic() -> float:
    return 2.0 * random.random() - 1.0

def simulate_single_photon_radial(max_steps: int = 20_000_000) -> PhotonResult:
    """
    1D radial transport with isotropic mu. Reflect at r=0. Escape when r >= R_EXIT.
    """
    r = 0.0
    total_path = 0.0
    steps = 0

    while r < R_EXIT and steps < max_steps:
        l = mean_free_path(r)
        s = draw_step_length(l)
        mu = draw_mu_isotropic()
        r_new = r + s * mu
        if r_new < 0.0:
            r_new = -r_new
        total_path += s
        r = r_new
        steps += 1

    t = total_path / C
    return PhotonResult(steps=steps, total_path_cm=total_path, escape_time_s=t, escaped=(r >= R_EXIT))

def simulate_many(n: int, seed: int | None = None, max_steps: int = 20_000_000):
    if seed is not None:
        random.seed(seed)
    out = []
    for _ in range(n):
        out.append(simulate_single_photon_radial(max_steps=max_steps))
    return out

def summarize(results: list[PhotonResult]) -> dict:
    import statistics as stats
    esc = [r for r in results if r.escaped]
    if not esc:
        return {"photons": len(results), "escaped": 0}
    steps = [r.steps for r in esc]
    times = [r.escape_time_s for r in esc]
    paths = [r.total_path_cm for r in esc]
    to_years = lambda s: s / (60.0 * 60.0 * 24.0 * 365.25)
    return {
        "photons": len(results),
        "escaped": len(esc),
        "mean_steps": stats.mean(steps),
        "median_steps": stats.median(steps),
        "max_steps": max(steps),
        "mean_time_years": stats.mean(map(to_years, times)),
        "median_time_years": stats.median(map(to_years, times)),
        "mean_path_over_Rsun": stats.mean(p / R_SUN for p in paths),
    }

# ======================== 3D path sim (for animation) ============
def simulate_photon_path_3d(max_steps: int, seed: int):
    """
    Full 3D walk for animation. Returns x,y,z arrays [cm] and cumulative time [years].
    """
    rng = np.random.default_rng(seed)
    x = y = z = 0.0
    r = 0.0
    xs = [x]; ys = [y]; zs = [z]
    cum_path = [0.0]
    steps = 0

    while r < R_EXIT and steps < max_steps:
        l = mean_free_path(r)
        s = rng.exponential(l)
        mu = rng.uniform(-1.0, 1.0)
        phi = rng.uniform(0.0, 2.0 * np.pi)
        sint = math.sqrt(max(0.0, 1.0 - mu * mu))
        ux = sint * math.cos(phi); uy = sint * math.sin(phi); uz = mu

        x += s * ux; y += s * uy; z += s * uz
        r = math.sqrt(x*x + y*y + z*z)

        xs.append(x); ys.append(y); zs.append(z)
        cum_path.append(cum_path[-1] + s)
        steps += 1

    xs = np.array(xs); ys = np.array(ys); zs = np.array(zs)
    cum_time_years = (np.array(cum_path) / C) / (60.0 * 60.0 * 24.0 * 365.25)
    return xs, ys, zs, cum_time_years, steps, (r >= R_EXIT)

def make_animation(
    out_path: str = "photon_escape.gif",
    seed: int = 19,
    max_steps: int = 250_000,     # longer timespan
    max_frames: int = 600,        # smoother movie, auto downsampled
    fps: int = 24,
    TIME_WARP: float = 1e6,       # label shows years * TIME_WARP, so time "passes" much faster
):
    xs, ys, zs, t_years, steps, escaped = simulate_photon_path_3d(max_steps=max_steps, seed=seed)

    # Downsample to a manageable number of frames
    stride = max(1, len(xs) // max_frames)
    xs = xs[::stride]; ys = ys[::stride]; zs = zs[::stride]; t_years = t_years[::stride]

    # Units in R_sun
    xs_u = xs / R_SUN; ys_u = ys / R_SUN
    r_exit_u = R_EXIT / R_SUN

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_aspect('equal', adjustable='box')
    lim = 1.05 * r_exit_u
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel('x / R_sun'); ax.set_ylabel('y / R_sun')
    title = f'Photon Simulation  |  steps sim: {steps:,}  escaped: {escaped}'
    ax.set_title(title)

    theta = np.linspace(0, 2*np.pi, 400)
    ax.plot(r_exit_u*np.cos(theta), r_exit_u*np.sin(theta))  # escape circle

    (line,) = ax.plot([], [])
    (point,) = ax.plot([], [], marker='o', markersize=3)
    status = ax.text(0.02, 0.98, '', transform=ax.transAxes, va='top')

    def init():
        line.set_data([], []); point.set_data([], []); status.set_text('')
        return line, point, status

    def update(i):
        line.set_data(xs_u[:i+1], ys_u[:i+1])
        point.set_data(xs_u[i], ys_u[i])
        r_now = math.sqrt(xs_u[i]**2 + ys_u[i]**2)
        t_fast = t_years[i] * TIME_WARP
        status.set_text(f'frame {i}  r = {r_now:.3f} $R_{{sun}}$   t ≈ {t_fast:.2e} years')
        return line, point, status

    ani = animation.FuncAnimation(fig, update, frames=len(xs_u), init_func=init, interval=1000/fps, blit=True)
    ani.save(out_path, writer=PillowWriter(fps=fps))
    print(f"Saved {out_path}  |  frames={len(xs_u)}  stride={stride}  steps_simulated={steps}  escaped={escaped}")

# ======================== main ========================
def main():
    # ---- Batch stats
    N_PHOTONS = 5
    RNG_SEED = 123
    MAX_STEPS_STATS = 20_000_000   # safety cap

    results = simulate_many(N_PHOTONS, seed=RNG_SEED, max_steps=MAX_STEPS_STATS)
    summary = summarize(results)
    print("Summary (Sun-only Monte Carlo, escape at 0.9 R_sun ):")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # ---- Animation
    make_animation(
        out_path="photon_escape.gif",
        seed=19,
        max_steps=250_000,   
        max_frames=600,      
        fps=24,
        TIME_WARP=1e6,       # years shown are multiplied by this factor
    )

if __name__ == "__main__":
    main()