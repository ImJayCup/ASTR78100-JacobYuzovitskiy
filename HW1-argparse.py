import sys
import argparse
print(sys.argv)
h = float(sys.argv[1])

def hitTheGround(h,g=9.81):
    t = (2*g*h)**0.5
    return t

if len(sys.argv) == 3:
    g = float(sys.argv[2])
    print(hitTheGround(h,g=g))
else:
    print(hitTheGround(h))