import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # fast off-screen rendering for GIFs
import io
import imageio.v2 as imageio
import math  # for erf, exp, sqrt, pi

# ---- pretty plotting defaults ----
plt.style.use("dark_background")
plt.rcParams.update({
    "figure.facecolor": "#05060a",
    "axes.facecolor":   "#05060a",
    "savefig.facecolor":"#05060a",
    "axes.edgecolor":   "#cfd3ff",
    "axes.labelcolor":  "#e6e6e6",
    "xtick.color":      "#cfd3ff",
    "ytick.color":      "#cfd3ff",
    "grid.color":       "#cfd3ff",
    "grid.alpha":       0.10,
    "axes.grid":        True,
})
STAR_MARKER = (5, 1)   # 5-point "star" marker

# ----------------- PHYSICAL UNITS  -----------------

G_SI = 6.67430e-11          # m^3 kg^-1 s^-2
PC   = 3.085677581e16       # m
KPC  = 1e3 * PC
MSUN = 1.98847e30           # kg
MYR  = 1e6 * 365.25 * 24 * 3600  # s

# Choose the scaling:
L_UNIT = 10 * PC            # 1 code length = 10 pc  -> 1000 = 10 kpc
M_UNIT = 1e8 * MSUN         # 1 code mass  = 1e8 Msun -> M_gal=200 => 2e10 Msun

# Derive the time unit so that G_code=1 matches physical gravity:
T_UNIT = math.sqrt(L_UNIT**3 / (G_SI * M_UNIT))  # seconds
V_UNIT = L_UNIT / T_UNIT                          # m/s

def to_kpc(x_code):  # works on scalars or numpy arrays
    return x_code * (L_UNIT / KPC)

def to_myr(t_code):
    return t_code * (T_UNIT / MYR)

def to_kms(v_code):
    return v_code * (V_UNIT / 1e3)

print(f"[Units] 1L = {L_UNIT/PC:.0f} pc, 1M = {M_UNIT/MSUN:.1e} Msun, "
      f"1T = {T_UNIT/MYR:.4f} Myr, 1V = {V_UNIT/1e3:.1f} km/s")

# Random RNG
rng = np.random.default_rng(0)

G = 1.0          # Gravitational constant (scaled)
EPS = 10      # Softening 


# ----------------- BASIC STAR + TREE (Barnes–Hut) ----------------- #

class Star:
    def __init__(self, pos=None, vel=None, mass=1.0):
        if pos is None:
            pos = rng.random(2)
        if vel is None:
            vel = np.zeros(2)

        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.mass = float(mass)

    def __repr__(self):
        return f"Star(pos={self.pos}, mass={self.mass:.2f})"


class Tree:
    """
    Barnes–Hut quadtree in 2D.
    Instance of Tree() contains Node() which contain Star()

    """

    def __init__(self, stars, theta=0.5):
        self.theta = theta

        xs = np.array([s.pos[0] for s in stars])
        ys = np.array([s.pos[1] for s in stars])

        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()

        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        half_size = 0.5 * max(xmax - xmin, ymax - ymin)
        if half_size == 0:
            half_size = 1.0

        self.xlim = (cx - half_size, cx + half_size)
        self.ylim = (cy - half_size, cy + half_size)

        self.root = Tree.Node(self.xlim, self.ylim)

        for s in stars:
            self.root.insert(s)

    class Node:
        """
        A quadtree node, representing a square region.
        """
        def __init__(self, xlim, ylim, depth=0):
            self.xlim = xlim
            self.ylim = ylim
            self.depth = depth

            self.children = [None, None, None, None]  # [SW, SE, NW, NE]
            self.star = None

            self.mass = 0.0
            self.com = np.zeros(2)

            self.is_leaf = True

        def _update_com(self, new_star):

            ### Update function for mass of a node
            ### Every time you insert a star into a node, run _update_com()
            if self.mass == 0.0:
                self.mass = new_star.mass
                self.com = new_star.pos.copy()
            else:
                total_mass = self.mass + new_star.mass
                self.com = (self.com * self.mass + new_star.pos * new_star.mass) / total_mass
                self.mass = total_mass

        def _subdivide(self):

            ### Subdivides a node into four child nodes 

            x0, x1 = self.xlim
            y0, y1 = self.ylim
            xm = 0.5 * (x0 + x1)
            ym = 0.5 * (y0 + y1)

            # Children: SW, SE, NW, NE
            self.children[0] = Tree.Node((x0, xm), (y0, ym), depth=self.depth + 1)  # SW
            self.children[1] = Tree.Node((xm, x1), (y0, ym), depth=self.depth + 1)  # SE
            self.children[2] = Tree.Node((x0, xm), (ym, y1), depth=self.depth + 1)  # NW
            self.children[3] = Tree.Node((xm, x1), (ym, y1), depth=self.depth + 1)  # NE

            self.is_leaf = False

        def _which_child(self, pos):

            ### This is just a function that returns the quadrant a star is in based on position. Important for
            ### inserting stars in the right node.

            x0, x1 = self.xlim
            y0, y1 = self.ylim
            xm = 0.5 * (x0 + x1)
            ym = 0.5 * (y0 + y1)
            x, y = pos

            if x <= xm and y <= ym:
                return 0  # SW
            elif x > xm and y <= ym:
                return 1  # SE
            elif x <= xm and y > ym:
                return 2  # NW
            else:
                return 3  # NE

        def insert(self, star):
            self._update_com(star)

            if self.is_leaf:
                if self.star is None:
                    self.star = star
                else:
                    existing = self.star
                    self.star = None
                    self._subdivide()

                    idx_old = self._which_child(existing.pos)
                    self.children[idx_old].insert(existing)

                    idx_new = self._which_child(star.pos)
                    self.children[idx_new].insert(star)
            else:
                idx = self._which_child(star.pos)
                self.children[idx].insert(star)

        def force_on(self, star, theta, G, eps):

            # Empty node
            if self.mass == 0.0:
                return np.zeros(2)

            # Don't self-interact
            if self.is_leaf and self.star is star:
                return np.zeros(2)

            dx = self.com[0] - star.pos[0]
            dy = self.com[1] - star.pos[1]
            r2 = dx*dx + dy*dy + eps*eps
            r = np.sqrt(r2)

            size = max(self.xlim[1] - self.xlim[0],
                       self.ylim[1] - self.ylim[0])

            # Far enough or leaf -> approximate as single mass
            if self.is_leaf or (size / r) < theta:
                F_mag = G * star.mass * self.mass / (r2 * r)  # = G m1 m2 / r^3
                return F_mag * np.array([dx, dy])

            # Otherwise recurse
            force = np.zeros(2)
            for child in self.children:
                if child is not None and child.mass > 0.0:
                    force += child.force_on(star, theta, G, eps)
            return force

    def force_on(self, star, G=G, eps=EPS):
        return self.root.force_on(star, self.theta, G, eps)

def step_leapfrog(stars, dt, theta=0.5, G=G, eps=EPS):
    ### Solves the second-order system dx/dt = v, dv/dt = a
    ### leap-frog integration instead of Euler

    # kick (half)
    tree = Tree(stars, theta=theta)
    acc = [tree.force_on(s, G=G, eps=eps) / s.mass for s in stars]
    for s, a in zip(stars, acc):
        s.vel += 0.5 * dt * a

    # drift
    for s in stars:
        s.pos += dt * s.vel

    # kick (half) with new forces
    tree = Tree(stars, theta=theta)
    acc2 = [tree.force_on(s, G=G, eps=eps) / s.mass for s in stars]
    for s, a in zip(stars, acc2):
        s.vel += 0.5 * dt * a

# ----------------- SPIRAL GALAXY INITIAL CONDITIONS ----------------- #

def make_spiral_galaxy(
    n_stars,
    center=np.array([0.0, 0.0]),
    com_velocity=np.array([0.0, 0.0]),
    radius=1.0,
    rot_dir=1,
    n_arms=2,
    arm_spread=0.25,   # radians: smaller = thinner arms
    pitch=2.0,         # bigger = more winding
    Rd=None            # disk scale length
):
    
    ### Concentration of central disk. Higher Rd spreads the galaxy out more
    stars = []
    if Rd is None:
        Rd = radius / 1.5

    M_gal = 400.0      # total "mass"
    core = 5.0         

    ### Samples uniform distribution, then applies logarithmic transform to create spirals
    for _ in range(n_stars):
        # Exponential disk-ish radius (clipped)
        u = rng.random()
        r = -Rd * np.log(1 - u)
        r = min(r, radius)

        # Pick which arm, then place star near that arm
        arm = rng.integers(0, n_arms)
        base = 2*np.pi * arm / n_arms

        # Log-spiral angle: phi ~ base + pitch * ln(r)
        phi = base + rot_dir * pitch * np.log((r + core) / core)
        phi += rng.normal(scale=arm_spread)

        pos = center + np.array([r*np.cos(phi), r*np.sin(phi)])

        # Simple circular-ish velocity around center
        v_circ = np.sqrt(G * M_gal / (r + core))
        tang = rot_dir * np.array([-np.sin(phi), np.cos(phi)])
        vel = com_velocity + v_circ * tang

        # small random dispersion
        sigma = 0.2*v_circ
        vel += sigma * rng.normal(size=2)

        stars.append(Star(pos=pos, vel=vel, mass=M_gal/n_stars))

    return stars




# ----------------- MAIN: TWO GALAXIES MERGING ----------------- #

if __name__ == "__main__":
    
    #Number of stars per galaxy, Galaxy radii
    N_GAL1= 200
    N_GAL2= 200
    GAL1_RAD = 2000
    GAL2_RAD = 2000

    #Timestep, number of steps, opening angle
    dt = 0.2
    n_steps = 16000
    theta = 1

    # Galaxy separation and approach velocity
    offset = 2500
    v_approach = 0.2

    # Galaxy 1: left, moving right
    gal1_center = np.array([-offset, 0.0])
    gal1_vel = np.array([v_approach, 0.0])
    gal1 = make_spiral_galaxy(
        N_GAL1,
        center=gal1_center,
        com_velocity=gal1_vel,
        radius=GAL1_RAD,
        rot_dir=+1,
    )

    # Galaxy 2: right, moving left
    gal2_center = np.array([offset, 0.0])
    gal2_vel = np.array([-v_approach, 0.0])
    gal2 = make_spiral_galaxy(
        N_GAL2,
        center=gal2_center,
        com_velocity=gal2_vel,
        radius=GAL2_RAD,
        rot_dir=-1,
        pitch=4
    )

    stars = gal1 + gal2
    N_TOTAL = len(stars)

    print(f"Simulating {N_TOTAL} stars for {n_steps} steps...")

    # Evolve system
    split = len(gal1)          # works even if N_GAL1 != N_GAL2
    FRAME_STRIDE = 10

    '''
    import time
    start = time.time()
    for t in range(n_steps):
            step_leapfrog(stars, dt=dt, theta=theta)
    end = time.time()
    print("The Time to Run is: ")
    print(end-start)
    '''

    split = len(gal1)
    FRAME_STRIDE = 10

    # --- one-time figure setup ---
    plt.style.use("dark_background")
    STAR_MARKER = (5, 1)  # 5-point star

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_facecolor("#05060a")
    fig.patch.set_facecolor("#05060a")

    lim = to_kpc(offset * 2.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("x [kpc]")
    ax.set_ylabel("y [kpc]")

    # Create artists ONCE (empty at start)
    sc1_glow = ax.scatter([], [], s=30, marker=STAR_MARKER, alpha=0.08, linewidths=0)
    sc2_glow = ax.scatter([], [], s=30, marker=STAR_MARKER, alpha=0.08, linewidths=0)
    sc1      = ax.scatter([], [], s=10, marker=STAR_MARKER, alpha=0.9,  linewidths=0)
    sc2      = ax.scatter([], [], s=10, marker=STAR_MARKER, alpha=0.9,  linewidths=0)

    ax.grid(True, alpha=0.10)
    for spine in ax.spines.values():
        spine.set_alpha(0.25)

    import time
    start = time.time()
    with imageio.get_writer("galaxy_merge.gif", mode="I", fps=20) as writer:
        for t in range(n_steps):
            step_leapfrog(stars, dt=dt, theta=theta)

            if t % FRAME_STRIDE == 0:
                t_myr = to_myr(t * dt)

                xs = np.fromiter((s.pos[0] for s in stars), dtype=float, count=len(stars))
                ys = np.fromiter((s.pos[1] for s in stars), dtype=float, count=len(stars))

                xs1, ys1 = xs[:split], ys[:split]
                xs2, ys2 = xs[split:], ys[split:]

                # convert to kpc
                p1 = np.column_stack((to_kpc(xs1), to_kpc(ys1)))
                p2 = np.column_stack((to_kpc(xs2), to_kpc(ys2)))

                # update artist data (fast)
                sc1_glow.set_offsets(p1)
                sc2_glow.set_offsets(p2)
                sc1.set_offsets(p1)
                sc2.set_offsets(p2)

                ax.set_title(f"Galaxy merger  •  t = {t_myr:.2f} Myr", pad=10)

                # render to an array (no PNG roundtrip)
                fig.canvas.draw()
                w, h = fig.canvas.get_width_height()
                frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)

                writer.append_data(frame)

    plt.close(fig)
    print("Saved galaxy_merge.gif")
    end = time.time()
    print(f"Compile time is {end-start}")
    


    # ----------------- PLOT FINAL STATE ----------------- #

    t_myr = to_myr(t * dt)

    xs = np.array([s.pos[0] for s in stars])
    ys = np.array([s.pos[1] for s in stars])

    xs1, ys1 = xs[:N_GAL1], ys[:N_GAL1]
    xs2, ys2 = xs[N_GAL1:], ys[N_GAL1:]

    # convert to kpc for plotting
    xs1_kpc, ys1_kpc = to_kpc(xs1), to_kpc(ys1)
    xs2_kpc, ys2_kpc = to_kpc(xs2), to_kpc(ys2)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(xs1_kpc, ys1_kpc, s=1, alpha=0.7)
    ax.scatter(xs2_kpc, ys2_kpc, s=1, alpha=0.7)

    lim = to_kpc(offset * 2.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", "box")
    ax.set_title(f"t = {t_myr:.2f} Myr")
    ax.set_xlabel("x [kpc]")
    ax.set_ylabel("y [kpc]")

    plt.show()

    

