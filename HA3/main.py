#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, rc
import numpy as np
from scipy.linalg import eig
import collections
from pprint import pprint

class ConvergenceException(Exception):
    def __init__(self, value):
        self.parameter = value
    def __str__(self):
        return repr(self.parameter)

# Constants
PI = np.pi
N = 1000
R_MAX = 10          # a.u.
R_MIN = 0.01        # a.u. 
h = (R_MAX-R_MIN)/N # a.u.
ALPHA = np.array([0.297104, 1.236745, 5.749982, 38.216677])

# Fix TeX for figure text
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
tex_r_str = 'Distance [$a_0$]' 
tex_wf_str = '$\Psi$ [$a_0^{-3/2}$]'
tex_e_str = '$E_h$'

############################### Support Functions #############################

def norm_wavefunc(v,r):
    return v / np.trapz(np.abs(v), r)

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

def calc_E(S, C, Q, h):
    E = 0
    F = np.empty((4,4))

    # Calculate the Fock term
    for p in range(4):
        for q in range(4):
            F[p][q] = h[p][q]
            for r in range(4):
                for s in range(4):
                    F[p][q] += C[r]*C[s]*Q[p][q][r][s]

    # Find the eigenvalues and (right side) eigenvectors
    eig_val, vr = eig(F, S)
    i_min = np.argmin(eig_val)

    # Assign new weights
    C = norm_C(-vr[:,i_min], S)

    # Calculate the energy
    for p in range(4):
        for q in range(4):
            E += 2*C[p]*C[q]*h[p][q]
            for r in range(4):
                for s in range(4):
                    E += C[p]*C[q]*C[r]*C[s]*Q[p][q][r][s]

    return E, C

def find_E(init_C, alpha, return_all_E=False):
    # Starting values
    S = generate_S(ALPHA)
    C = norm_C(init_C, S)
    Q = generate_Q(ALPHA)
    h = generate_h(ALPHA)

    # Simulation settings and variables
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
            print("Converged in {0} iterations".format(iterations))
            if return_all_E:
                return E, C, Q, h, energies
            else:
                return E, C, Q, h
        if iterations > max_iter:
            raise ConverenceException("Failed to converge after {0} iterations,"
                                      " increase max_iter or re-check "
                                      " simulation input.".format(iterations))
            break # Do I need to break after an exception?

def xi(r, a):
    return np.exp(-1*a*r**2)

def phi(r, C, a):
    return sum(C*xi(r, a))

def phi_s(r):
    return 1/np.sqrt(PI)*np.exp(-r)

def electron_density(r):
    return 2*np.abs(phi_s(r))**2

def u(r):
    return np.sqrt(4*PI)*r*phi_s(r)

def V_H(r):
    # We get a list, best guess is that we want to return a list with the
    # Hartree poptential for all elements.
    if isinstance(r, (collections.Sequence, np.ndarray)):
        return [V_H(r_i) for r_i in r]

    elif r == 0:
        return 1
    else:
        return 1/r - (1 + 1/r)*np.exp(-2*r)

def V(r, n=None):
    A = np.diag([-2]*N)
    A = A + np.diag([1]*(N-1), 1) + np.diag([1]*(N-1), -1)
    A = A/h**2

    b = -2*PI*n*r        
    b[-1] -= 1/h**2

    U = np.linalg.solve(A, b)
    V = U/r

    return V

def KS(r, Z=0, V_H=0, V_X=0, V_C=0):
    # Create the diagonal elements corresponding to index N
    A = np.diag(np.array([2]*N))

    # Add the off diagonal elements for index N-1 and N+1, divide with correct
    # factor.
    A = A + np.diag([-1]*(N-1), 1) + np.diag([-1]*(N-1), -1)
    A = A/(2*h**2)

    # Add the terms from the hamiltonian to the diagonal
    A = A + np.diag(-Z/r + V_H + V_X + V_C)

    # Calculate eigenvalues and eigenvectors
    eig_val, vr = eig(A)
    i_min = np.argmin(eig_val)

    # Return the eigenvalue and eigenvector for the lowest energy/eigenvalue
    return eig_val[i_min], vr[:,i_min]

def ex_E(n):
    return -3/4*(3*n/PI)**(1/3)

def ex_pot(n):
    return -1/4*(3*n/PI)**(1/3) 

def corr_E(n):
    A = 0.0311
    B = -0.048
    C = 0.0020
    D = -0.0116
    gamma = -0.1423
    beta1 = 1.0529
    beta2 = 0.03334

    R_s = (3/(4*PI*n))**(1/3)

    energies = []
    
    for r_s in R_s:
        if r_s < 1:
            energies.append(A*np.log(r_s) + B + C*r_s*np.log(r_s) + D*r_s)
        else:
            energies.append(gamma/(1 + beta1*np.sqrt(r_s) + beta2*r_s))

    return np.array(energies)
    
    
def corr_pot(n, C_E):
    A = 0.0311
    B = -0.048
    C = 0.0020
    D = -0.0116
    gamma = -0.1423
    beta1 = 1.0529
    beta2 = 0.03334

    R_s = (3/(4*PI*n))**(1/3)

    energies = []

    pot = []
    
    for r_s, c_e in zip(R_s, C_E):
        if r_s < 1:
            pot.append(A*np.log(r_s) + B - A/3 + 2/3*C*r_s*np.log(r_s) + (2*D-C)*r_s/3)
        else:
            pot.append(c_e*(1+7/6*beta1*np.sqrt(r_s)+beta2*r_s)/(1+beta1*np.sqrt(r_s)+beta2*r_s))

    return np.array(pot)

########################### Solve the tasks ###################################

def solve_task_1():
    print("### TASK 1 ###")

    E, C, Q, h, energies = find_E([1,1,1,1], ALPHA, return_all_E=True)
    print("E: {0} Hartree".format(E))

    R = np.linspace(R_MIN,R_MAX,N)
    wv = [phi(r, C, ALPHA) for r in R]

    plt.figure()
    plt.title('Energy Convergence')
    plt.xlabel('Iterations')
    plt.ylabel(tex_e_str)
    plt.plot(energies, '-o')
    plt.savefig('images/task1_conv.png')    

    plt.figure()
    plt.title('Wave Function')
    plt.xlabel(tex_r_str)
    plt.ylabel(tex_wf_str)
    plt.plot(R, wv)
    plt.savefig('images/task1_wave.png')
    
    '''
    X = np.linspace(-1, 1, 100)
    Y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(X, Y)


    Z = np.empty((100,100))

    i = j = 0
    for x, i in zip(X[0,:], range(100)):
        for y, j in zip(Y[:,0], range(100)):
            #print("x: {}".format(x))
            #print("y: {}".format(y))
            Z[i][j] = phi(np.sqrt(x**2 + y**2), C, ALPHA)

    surf_fig = plt.figure()
    ax = surf_fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap=cm.viridis)
    '''

def solve_task_2():
    # Task 2
    print("### TASK 2 ###")
    R = np.linspace(R_MIN,R_MAX,N)
    n = electron_density(R)
    E_pot = V(R, n)
    print("See figure for result")

    # Figures
    plt.figure()
    plt.title('Potential Energy')
    plt.xlabel('Distance [$r$]')
    plt.ylabel('Potential Energy [eV]')
    plt.plot(R, E_pot, label='Calculated $E_pot$')
    plt.plot(R, V_H(R), 'r-.', label='Hartree potential')
    plt.legend(loc='upper right')
    plt.savefig('images/task2.png')

def solve_task_3():
    # Task 3
    print("### TASK 3 ###")
    R = np.linspace(R_MIN,R_MAX,N)
    E_ks, u = KS(R, 1)
    print("Ground state energy: {0}".format(E_ks))
    norm_u = abs(u)/np.sqrt(np.trapz(np.abs(u)**2,R))

    plt.figure()
    plt.title('Normalized Wave Function')
    plt.xlabel(tex_r_str)
    plt.ylabel(tex_wf_str)
    plt.plot(R, norm_u)
    plt.savefig('images/task3.png')

def solve_task_4(): 
    print("### TASK 4 ###")
    R = np.linspace(R_MIN,R_MAX,N)
    energies = []
    n = np.ones((N,))/N
    pot = V(R, 2*n)
    E_ks, u = KS(R, Z=2, V_H=pot)
    u = u/np.sqrt(np.trapz(u**2))
    E_ground = 2*E_ks - 2*np.trapz(u**2*pot/2, R)

    energies.append(E_ground)

    # Simulation settings and variables
    iterations = 0
    max_iter = 1e1
    resolution = 1e-5
    small_enough = False
    
    while not small_enough:
        iterations += 1
        n = u**2/(4*PI*R**2)
        pot = V(R, 2*n)
        E_ks, u = KS(R, Z=2, V_H=pot)
        u = u/np.sqrt(np.trapz(u**2))
        E_ground = 2*E_ks - 2*np.trapz(u**2*pot/2, R)
        energies.append(E_ground)
        if (abs(energies[-1] - energies[-2])) < resolution:
            small_enough = True
            print("Converged in {0} iterations".format(iterations))
            
        if iterations > max_iter:
            raise ConverenceException("Failed to converge after {0} iterations,"
                                      " increase max_iter or re-check "
                                      " simulation input.".format(iterations)) 
    
    print("Ground state energy: {0}".format(E_ground))

    plt.figure()
    plt.plot(energies, '-o')
    
def solve_task_5():
    print("### TASK 5 ###")
    R = np.linspace(R_MIN,R_MAX,N)
    energies = []
    n = np.ones((N,))/N
    pot = V(R, 2*n)
    x_pot = ex_pot(2*n)
    E_x = ex_E(2*n)
    E_ks, u = KS(R, Z=2, V_H=2*pot, V_X=x_pot)
    u = u/np.sqrt(np.trapz(u**2))
    E_ground = 2*E_ks - 2*np.trapz(u**2*(pot/2 + x_pot - E_x), R)

    energies.append(E_ground)

    # Simulation settings and variables
    iterations = 0
    max_iter = 1e1
    resolution = 1e-5
    small_enough = False
    
    while not small_enough:
        iterations += 1
        
        n = u**2/(4*PI*R**2)
        pot = V(R, 2*n)
        x_pot = ex_pot(2*n)
        E_ks, u = KS(R, Z=2, V_H=2*pot, V_X=x_pot)
        u = u/np.sqrt(np.trapz(u**2))
        E_x = ex_E(2*n)
        E_ground = 2*E_ks - 2*np.trapz(u**2*(pot/2 + x_pot - E_x), R)

        energies.append(E_ground)
        
        if (abs(energies[-1] - energies[-2])) < resolution:
            small_enough = True
            print("Converged in {0} iterations".format(iterations))
            
        if iterations > max_iter:
            raise ConverenceException("Failed to converge after {0} iterations,"
                                      " increase max_iter or re-check "
                                      " simulation input.".format(iterations)) 
    
    print("Ground state energy: {0}".format(E_ground))

    plt.figure()
    plt.plot(energies, '-o')

def solve_task_6():
    print("### TASK 6 ###")
    R = np.linspace(R_MIN,R_MAX,N)
    energies = []
    n = np.ones((N,))/N
    pot = V(R, 2*n)
    x_pot = ex_pot(2*n)
    E_c = corr_E(2*n)
    c_pot = corr_pot(2*n, E_c)
    E_x = ex_E(2*n)
    E_ks, u = KS(R, Z=2, V_H=2*pot, V_X=x_pot, V_C=c_pot)
    u = u/np.sqrt(np.trapz(u**2))
    E_ground = 2*E_ks - 2*np.trapz(u**2*(pot/2
                                         + x_pot + c_pot
                                         - E_x - E_c), R)
    
    energies.append(E_ground)
    
    # Simulation settings and variables
    iterations = 0
    max_iter = 1e1
    resolution = 1e-5
    small_enough = False
    
    while not small_enough:
        iterations += 1
        
        n = u**2/(4*PI*R**2)
        pot = V(R, 2*n)
        E_c = corr_E(2*n)
        x_pot = ex_pot(2*n)
        c_pot = corr_pot(2*n, E_c)
        E_ks, u = KS(R, Z=2, V_H=2*pot, V_X=x_pot, )
        u = u/np.sqrt(np.trapz(u**2))

        E_x = ex_E(2*n)
        E_ground = 2*E_ks - 2*np.trapz(u**2*(pot/2
                                             + x_pot + c_pot
                                             - E_x - E_c), R)

        energies.append(E_ground)

        if (abs(energies[-1] - energies[-2])) < resolution:
            small_enough = True
            print("Converged in {0} iterations".format(iterations))
            
        if iterations > max_iter:
            raise ConverenceException("Failed to converge after {0} iterations,"
                                      " increase max_iter or re-check "
                                      " simulation input.".format(iterations)) 
    
    print("Ground state energy: {0}".format(E_ground))

    plt.figure()
    plt.plot(energies, '-o')
                                       
# Solve all tasks
solve_task_1()
solve_task_2()
solve_task_3()
solve_task_4()
solve_task_5()
solve_task_6()

print("### All tasks finished! ###")

# Display all figures
plt.show()
