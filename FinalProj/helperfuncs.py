from matplotlib import pyplot as plt
import numpy as np

def plotInstance(stars, title=None, save=None):
    plt.clf()                             # clear previous frame
    xs = [s.pos[0] for s in stars]
    ys = [s.pos[1] for s in stars]

    plt.scatter(xs, ys, s=10, color="white")
    plt.gca().set_facecolor("black")
    plt.gca().set_aspect("equal", adjustable="box")

    # Auto scale with margin
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    margin = 0.05 * max(xmax - xmin, ymax - ymin, 1e-6)

    plt.xlim(xmin - margin, xmax + margin)
    plt.ylim(ymin - margin, ymax + margin)

    if title:
        plt.title(title, color="white")

    plt.xlabel("x", color="white")
    plt.ylabel("y", color="white")
    plt.tick_params(colors="white")

    # Optional save
    if save is not None:
        plt.savefig(save, dpi=120, bbox_inches="tight")

    plt.pause(0.1)   # allows animation updates
