#!/usr/bin/env python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from scipy.linalg import eig
from pprint import pprint

# Constants
PI = np.pi
R_MAX = 10
R_MIN = 0.01
N = 1000
h = (R_MAX-R_MIN)/N

def phi(r):
    return 1/np.sqrt(PI)*np.exp(-r)

def n(r):
    return 2*np.abs(phi(r))**2

def u(r):
    return np.sqrt(4*PI)*r*phi(r)

def V_H(r):
    if r == 0:
        return 1
    else:
        return 1/r - (1 + 1/r)*np.exp(-2*r)

def V(r):
    A = np.diag([-2]*N)
    A = A + np.diag([1]*(N-1), 1) + np.diag([1]*(N-1), -1)
    A = A/h**2

    pprint(A)

    b = -2*PI*n(r)*r
    b[-1] -= 1/h**2

    U = np.linalg.solve(A, b)
    V = U/r

    return V

def KS(r, Z=0, V_H=0, V_X=0, V_C=0):
    A = np.diag(np.array([-2]*N) - Z/r + V_H + V_X + V_C)
    A = A + np.diag([1]*(N-1), 1) + np.diag([1]*(N-1), -1)
    A = A/(2*h**2)

    pprint(A[1,1])

    eig_val, vr = eig(A)
    i_min = np.argmin(eig_val)

    return eig_val[i_min], vr[i_min]

R = np.linspace(0.01,R_MAX,N)
E_pot = V(R)

E_ks, wv = KS(R, 1)
print(E_ks)
#pprint(wv)
n = abs(wv)**2
n = n/np.trapz(n,R)

# Figures
plt.figure(1)
plt.xlabel('Potential Energy')
plt.ylabel('Distance')
plt.plot(R, E_pot, R, [V_H(r) for r in R],'r-.')


plt.figure(2)
plt.plot(R, n)

plt.show()