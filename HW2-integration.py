from math import exp, sin,cos,tan
import argparse
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, Polygon
from scipy.interpolate import lagrange

parser = argparse.ArgumentParser(description="Gives Integral of exp(-x**2)")
parser.add_argument("--func",type=str,default='exp(-x**2)',help="Desired Integrand as a function of x.")
parser.add_argument("--bounds",type=float,nargs=2,default=[0,3],help="Desired bounds of Integration as a list. Ex: [0,3].")
parser.add_argument("--steps",type=int,default=30,help="Desired number of steps. Ex: 30 steps from 0 to 3 is 0.1 steps size.")
args=parser.parse_args()

def f(x):
    return eval(args.func)

def trap(a,b,n: int):
    z = (b-a)/n
    integral = z*(0.5*f(a)+0.5*f(b)+sum([f(a+k*z) for k in range(n)]))
    n2 = n*2
    z2 = (b-a)/n2
    integral2 = z2*(0.5*f(a)+0.5*f(b)+sum([f(a+k*z2) for k in range(n2)]))
    return integral, abs(integral2-integral)/15

def simp(a,b,n: int):
    z = (b-a)/n
    integral = (z/3)*(f(a) + f(b) + 4*sum([f(a+(2*k-1)*z) for k in range((n//2)+1)]) + 2*sum([f(a + 2*k*z) for k in range(n//2)]))
    n2 = n*2
    z2 = (b-a)/n2
    integral2 = (z2/3)*(f(a) + f(b) + 4*sum([f(a+(2*k-1)*z2) for k in range((n2//2)+1)]) + 2*sum([f(a + 2*k*z2) for k in range(n2//2)]))
    return integral, abs(integral2-integral)/15

print(f"Trapezoidal Riemannian Sum: {trap(args.bounds[0],args.bounds[1],args.steps)[0]:.2f}")
print(f"Integral w/ Simpson's Rule: {simp(args.bounds[0],args.bounds[1],args.steps)[0]:.2f}")

fig, axs = plt.subplots(ncols=2,nrows=1)
fig.suptitle(f"Trapezoidal Integral: {trap(args.bounds[0],args.bounds[1],args.steps)[0]:.2f}. Simpson: {simp(args.bounds[0],args.bounds[1],args.steps)[0]:.2f}")

x = np.linspace(args.bounds[0],args.bounds[1],args.steps)
axs[0].set_title(f"Trapezoidal Riemannian Sum of {args.func}")
axs[0].scatter(x,[f(i) for i in x])

stepSize = (args.bounds[1]-args.bounds[0])/args.steps
for i in range(len(x)-1):
    axs[0].add_patch(Polygon([(x[i],0),(x[i],f(x[i])),(x[i+1],f(x[i+1])),(x[i+1],0)],edgecolor='black',facecolor='g'))


axs[1].set_title(f"{args.func}")
axs[1].scatter(x,[f(i) for i in x])

#for i in range(len(x)-1):
plt.show()