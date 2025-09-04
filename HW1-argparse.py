"""
This is a short program to calculate the time it takes a ball to hit the ground when thrown from an input height 'h' and acted upon
by a force due to gravitational acceleration. Both values are input by the user by running the program from the command line. For help:
'python HW1-argparse --help'
"""

import sys
import argparse
parser = argparse.ArgumentParser(description="Calculates the time it takes for a ball to hit the ground with a given height and acceleration due to gravity.")
parser.add_argument("h",type=float,help="Height of the Ball off the ground")
parser.add_argument("g",type=float,default=9.81,help="Strength of Gravity on target planet")
args=parser.parse_args()
def hitTheGround(h,g=9.81):
    t = (2*g/h)**0.5
    return t
print(f"The ball hit the ground in {hitTheGround(args.h,g=args.g):.2f}s.")
