#! /usr/bin/env python2
# -*- encoding:utf-8 -*-

from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.visualize import view
import numpy as np
from scipy.optimize import fmin
import matplotlib.pyplot as plt
from pprint import pprint

sigma = 3.4
epsilon = 1.04e-2

N = 10

# Calculates the Euclidean distance between two N-dimensional vectors
def euc_dist(r1, r2 = [0,0,0]):
    a = np.sqrt( sum( [ (i-j)**2 for i,j in zip(r1, r2) ] ) )
    return a

# Takes several types of input and calculates the Lennard-Jones potential
def lj(r1, r2 = None):
    # The actual potential equation
    pot = lambda r: 4*epsilon*((sigma/r)**12 - (sigma/r)**6)

    # If we get two arguments we assume it is two vectors
    if r2 is not None:
        return pot(euc_dist(r1,r2))
    # Handle lists like this
    elif (type(r1) in [list, np.ndarray, np.array] 
          and len(r1) > 1):
        return [ pot(r) for r in r1 ]
    # Handle a simple distance 
    elif type(r1) in [float, int] or len(r1) == 1:
        return pot(r1)
    # When in doubt, complain on the user
    else:
        print "Unknown input type ", type(r1)
        return None

# Takes an array of cartesian coordinates and calculates the Lennard-Jones 
# potential
def cluster_lj(atoms):
    potential = 0
    N = np.size(atoms, 0)
    for i in range(N-1):
        for j in range(i+1,N):
            potential += lj(atoms[i], atoms[j])
    return potential

         
# Finds min using x0 as initial guess and 1D gradient descent with fixed step 
# size, super slow but it works.
def find_min(f, x0, gamma = 0.01, h = 0.0001, prec = 1e-10):
    # Could calculate the analytical derivative, but I'm too lazy
    while True:
        df = (f(x0 + h) - f(x0))/h
        x_prev = x0
        x0 += -gamma * df
        if abs(x0 - x_prev) < prec:
            break
    return x0

# Make data for a nice graph
r = np.linspace(3,15,1000)
E = lj(r)

# Find equilibrium distance
my_r_min = find_min(lj, 5)
r_min = fmin(lj, 5, disp=0)[0]

# Read positions from file
pos = []
with open('coordinates.txt','r') as f:
    for line in f:
        strings = line.strip().split(',')
        col = [float(s) for s in strings]
        pos.append(col)

# Create an array with atom positions for my own LJ calculation
my_atoms = np.array(pos)

# Setup ASE
ase_atoms = Atoms('ArArArAr', positions=my_atoms)
calc = LennardJones(epsilon=epsilon, sigma=sigma, rc=100)
ase_atoms.set_calculator(calc)

'''
#Double check that we have the the same positions
print "My positions:"
pprint(my_atoms)
print "ASE positions: "
pprint(ase_atoms.get_positions())
'''

# Calculate potential energy
my_cluster_E = cluster_lj(my_atoms)
ase_cluster_E = ase_atoms.get_potential_energy()

# Print stuff
print "----------------------------------------------------"
print "R_0 potential using own search: ", my_r_min
print "R_0 potential using scipy fmin: ", r_min
print "R_0 difference between methods: ", my_r_min - r_min
print "----------------------------------------------------"
print "My calculated cluster energy:   ", my_cluster_E
print "Ase calculated cluster energy:  ", ase_cluster_E
print "Cluster energy difference:      ", my_cluster_E - ase_cluster_E
print "----------------------------------------------------"

# Visual stuff
#view(ase_atoms)

#plt.plot(r,E)
#plt.show()