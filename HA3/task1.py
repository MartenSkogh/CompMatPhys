#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eig
from pprint import pprint

alpha = np.array([0.297104, 1.236745, 5.749982, 38.216677])

def generate_S(a):
    S = np.empty((4,4))
    for p in range(4):
        for q in range(4):
            S[p][q] = (np.pi/(a[p] + a[q]))**(3/2)

    return S

def generate_h(a):
    h = np.empty((4,4))
    for p in range(4):
        for q in range(4):
            h[p][q] = (3*a[p]*a[q]*np.pi**(3/2)/(a[p]+a[q])**(5/2) 
                      - 4*np.pi/(a[p]+a[q]))

    return h

def generate_Q(a):
    Q = np.empty((4,4,4,4))

    def calc_element(p, q ,r, s):
        const = 2*(np.pi**(5/2))
        nom = ((a[p] + a[q])*(a[r] + a[s])
               *np.sqrt(a[p] + a[q] + a[r] + a[s]))
        return const/nom

    # Check if this is possible to flatten in a nice way
    for p in range(4):
        for q in range(4):
            for r in range(4):
                for s in range(4):
                    Q[p][q][r][s] = calc_element(p,q,r,s)
    
    return Q

def norm_C(C, S):
    # Sum of all possibilities must equal 1
    norm_factor = 0
    for p in range(4):
        for q in range(4):
            norm_factor += C[p]*C[q]*S[p][q]

    return C/np.sqrt(norm_factor)

def calc_E(S,C,Q,h):
    E = 0
    F = np.empty((4,4))
    for p in range(4):
        for q in range(4):
            F[p][q] = h[p][q]
            for r in range(4):
                for s in range(4):
                    F[p][q] += C[r]*C[s]*Q[p][q][r][s]

    eig_val, vr = eig(F, S)

    C = norm_C(vr[-1], S)

    
    for p in range(4):
        for q in range(4):
            E += 2*C[p]*C[q]*h[p][q]
            for r in range(4):
                for s in range(4):
                    E += C[p]*C[q]*C[r]*C[s]*Q[p][q][r][s]

    return E, C

# Starting values
S = generate_S(alpha)
C = norm_C([1,1,1,1], S)
Q = generate_Q(alpha)
h = generate_h(alpha)

# Simulation settings
iterations = 0
max_iter = 1e3
resolution = 1e-8
small_enough = False

# Array with all energies from iterations
energies = []

# Make a firs energy calculation
E, C = calc_E(S, C, Q, h)
energies.append(E)

# Run loop, 
while not small_enough:
    iterations += 1
    E, C = calc_E(S, C, Q, h)
    energies.append(E)
    if (abs(energies[-1] - energies[-2])) < resolution:
        small_enough = True
        print("E: {} eV".format(E))
        print("Converged in {} iterations".format(iterations))
    if iterations > max_iter:
        print("E: {} eV".format(E))
        print("Failed to converge after {} iterations, increase max_iter or"         " re-check simulation input.".format(iterations))
        break



plt.plot(energies, 'ro')
plt.plot(energies)
plt.show()