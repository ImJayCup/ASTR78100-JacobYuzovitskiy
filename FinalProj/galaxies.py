import numpy as np
import matplotlib.pyplot as plt
import io
import imageio.v2 as imageio
import math  # for erf, exp, sqrt, pi

# --- dynamical friction parameters ---
RHO_DF = 0.05      # background density (arbitrary units)
SIGMA_DF = 0.7     # velocity dispersion of background
LN_LAMBDA = 3.0    # Coulomb logarithm

# --- extra velocity damping (per unit time) ---
NU_DAMP = 0.03     # try 0.02–0.05 and tune


# Random RNG
rng = np.random.default_rng(0)

G = 1.0          # Gravitational constant (scaled)
EPS = 1e-3       # Softening to avoid singularities


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
            if self.mass == 0.0:
                self.mass = new_star.mass
                self.com = new_star.pos.copy()
            else:
                total_mass = self.mass + new_star.mass
                self.com = (self.com * self.mass + new_star.pos * new_star.mass) / total_mass
                self.mass = total_mass

        def _subdivide(self):
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
    
def chandra_df_accel(M, v_vec, rho=RHO_DF, sigma=SIGMA_DF, lnΛ=LN_LAMBDA):
    """
    Chandrasekhar dynamical friction acceleration on a massive object of mass M
    moving with velocity v_vec through a background of density rho and velocity
    dispersion sigma.

    Returns a vector a_df with the same shape as v_vec.
    """
    V = np.linalg.norm(v_vec)
    if V < 1e-8:
        return np.zeros(2)

    X = V / (math.sqrt(2.0) * sigma)
    # Dimensionless factor [erf(X) - 2X e^{-X^2} / sqrt(pi)]
    f_X = math.erf(X) - (2.0 * X / math.sqrt(math.pi)) * math.exp(-X * X)

    # Chandrasekhar formula: a_df ∝ -v / V^3
    coeff = -4.0 * math.pi * (G ** 2) * M * rho * lnΛ * f_X / (V ** 3)
    return coeff * v_vec


def step_barnes_hut(stars, dt=0.01, theta=0.5, G=G, eps=EPS):
    tree = Tree(stars, theta=theta)

    forces = []
    for s in stars:
        f = tree.force_on(s, G=G, eps=eps)
        forces.append(f)

    for s, f in zip(stars, forces):
        a = f / s.mass
        s.vel += a * dt
        s.pos += s.vel * dt

# ----------------- SPIRAL GALAXY INITIAL CONDITIONS ----------------- #

def make_spiral_galaxy(
    n_stars,
    center=np.array([0.0, 0.0]),
    com_velocity=np.array([0.0, 0.0]),
    radius=1.0,
    rot_dir=1,
):
    stars = []
    # use a smaller effective mass – this just sets the rotation curve
    M_gal = 10.0          # instead of n_stars
    V_SCALE = 1         # scale factor for rotation speed

    for i in range(n_stars):
        r = radius * np.sqrt(rng.random())
        phi = 2.0 * np.pi * rng.random()
        twist = 6.0 * r
        phi_spiral = phi + rot_dir * twist

        x = r * np.cos(phi_spiral)
        y = r * np.sin(phi_spiral)
        pos = np.array([x, y]) + center

        # much gentler circular velocity
        v_circ = V_SCALE * np.sqrt(G * M_gal / (r + 0.3))
        tan = rot_dir * np.array([-np.sin(phi_spiral), np.cos(phi_spiral)])
        vel = v_circ * tan + com_velocity

        vel += 0.05 * rng.normal(size=2)

        stars.append(Star(pos=pos, vel=vel, mass=1.0))

    return stars



# ----------------- MAIN: TWO GALAXIES MERGING ----------------- #

if __name__ == "__main__":
    # Number of stars per galaxy
    N_PER_GAL = 100

    # Galaxy separation and approach velocity
    offset = 600.0
    v_approach = 0.1

    # Galaxy 1: left, moving right
    gal1_center = np.array([-offset, 0.0])
    gal1_vel = np.array([v_approach, 0.0])
    gal1 = make_spiral_galaxy(
        N_PER_GAL,
        center=gal1_center,
        com_velocity=gal1_vel,
        radius=500,
        rot_dir=+1,
    )

    # Galaxy 2: right, moving left
    gal2_center = np.array([offset, 0.0])
    gal2_vel = np.array([-v_approach, 0.0])
    gal2 = make_spiral_galaxy(
        N_PER_GAL,
        center=gal2_center,
        com_velocity=gal2_vel,
        radius=500,
        rot_dir=-1,
    )

    stars = gal1 + gal2
    N_TOTAL = len(stars)

    # Simulation parameters
    dt = 0.5
    n_steps = 2000
    theta = 0.5

    print(f"Simulating {N_TOTAL} stars for {n_steps} steps...")

    # Evolve system
    frames = []
    FRAME_STRIDE = 10   # save every 10th step

    for t in range(n_steps):
        step_barnes_hut(stars, dt=dt, theta=theta)

        if t % FRAME_STRIDE == 0:
            xs = np.array([s.pos[0] for s in stars])
            ys = np.array([s.pos[1] for s in stars])

            xs1, ys1 = xs[:N_PER_GAL], ys[:N_PER_GAL]
            xs2, ys2 = xs[N_PER_GAL:], ys[N_PER_GAL:]

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(xs1, ys1, s=1, alpha=0.7)
            ax.scatter(xs2, ys2, s=1, alpha=0.7)

            ax.set_xlim(-offset*2.5, offset*2.5)
            ax.set_ylim(-offset*2.5, offset*2.5)
            ax.set_aspect("equal", "box")
            ax.set_title(f"t = {t}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")

            # ---- save figure to in-memory buffer and read as image ----
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100) 
            buf.seek(0)
            frame = imageio.imread(buf)
            buf.close()
            frames.append(frame)
            plt.close(fig)

            plt.close(fig)



    # ----------------- PLOT FINAL STATE ----------------- #

    xs = np.array([s.pos[0] for s in stars])
    ys = np.array([s.pos[1] for s in stars])

    # First N_PER_GAL -> galaxy 1, rest -> galaxy 2
    xs1, ys1 = xs[:N_PER_GAL], ys[:N_PER_GAL]
    xs2, ys2 = xs[N_PER_GAL:], ys[N_PER_GAL:]

    plt.figure(figsize=(6, 6))
    plt.scatter(xs1, ys1, s=1, alpha=0.7, label="Galaxy 1")
    plt.scatter(xs2, ys2, s=1, alpha=0.7, label="Galaxy 2")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Barnes–Hut: Merging Spiral Galaxies (final snapshot)")
    plt.legend(markerscale=5)
    plt.gca().set_aspect("equal", "box")
    plt.tight_layout()
    plt.show()
    
    imageio.mimsave("galaxy_merge.gif", frames, fps=20)
    print("Saved galaxy_merge.gif")

    

