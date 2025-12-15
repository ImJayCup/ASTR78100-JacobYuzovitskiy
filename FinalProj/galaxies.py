from networkx import radius
import numpy as np
import matplotlib.pyplot as plt
import io
import imageio.v2 as imageio
import math  # for erf, exp, sqrt, pi

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

def step_leapfrog(stars, dt, theta=0.5, G=G, eps=EPS):
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
    pitch=4.0,         # bigger = more winding
    Rd=None            # disk scale length
):
    stars = []
    if Rd is None:
        Rd = radius / 3.0

    M_gal = 200.0      # tune this
    core = 5.0         # avoid huge speeds at center

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
    # Number of stars per galaxy
    N_PER_GAL = 50

    # Galaxy separation and approach velocity
    offset = 1000
    v_approach = 0.5

    # Galaxy 1: left, moving right
    gal1_center = np.array([-offset, 0.0])
    gal1_vel = np.array([v_approach, 0.0])
    gal1 = make_spiral_galaxy(
        N_PER_GAL,
        center=gal1_center,
        com_velocity=gal1_vel,
        radius=1000,
        rot_dir=+1,
    )

    # Galaxy 2: right, moving left
    gal2_center = np.array([offset, 0.0])
    gal2_vel = np.array([-v_approach, 0.0])
    gal2 = make_spiral_galaxy(
        N_PER_GAL,
        center=gal2_center,
        com_velocity=gal2_vel,
        radius=1000,
        rot_dir=-1,
    )

    stars = gal1 + gal2
    N_TOTAL = len(stars)

    # Simulation parameters
    dt = 0.05
    n_steps = 80000
    theta = 0.2

    print(f"Simulating {N_TOTAL} stars for {n_steps} steps...")

    # Evolve system
    frames = []
    FRAME_STRIDE = 10   # save every 10th step

    for t in range(n_steps):
        step_leapfrog(stars, dt=dt, theta=theta)

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

    

