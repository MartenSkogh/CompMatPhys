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

def calc_E(S, C, Q, h):
    E = 0
    F = np.empty((4,4))
    for p in range(4):
        for q in range(4):
            F[p][q] = h[p][q]
            for r in range(4):
                for s in range(4):
                    F[p][q] += C[r]*C[s]*Q[p][q][r][s]

    eig_val, vr = eig(F, S)
    i_min = np.argmin(eig_val)

    C = norm_C(-vr[:,i_min], S)

    
    for p in range(4):
        for q in range(4):
            E += 2*C[p]*C[q]*h[p][q]
            for r in range(4):
                for s in range(4):
                    E += C[p]*C[q]*C[r]*C[s]*Q[p][q][r][s]

    return E, C

def xi(r, a):
    return np.exp(-1*a*r**2)

def phi(r, C, a):
    return sum(C*xi(r, a))

def phi_s(r):
    return 1/np.sqrt(PI)*np.exp(-r)

def n(r):
    return 2*np.abs(phi_s(r))**2

def u(r):
    return np.sqrt(4*PI)*r*phi_s(r)

def V_H(r):
    if r == 0:
        return 1
    else:
        return 1/r - (1 + 1/r)*np.exp(-2*r)

def V(r):
    A = np.diag([-2]*N)
    A = A + np.diag([1]*(N-1), 1) + np.diag([1]*(N-1), -1)
    A = A/h**2

    b = -2*PI*n(r)*r
    b[-1] -= 1/h**2

    U = np.linalg.solve(A, b)
    V = U/r

    return V

def KS(r, Z=0, V_H=0, V_X=0, V_C=0):
    A = np.diag(np.array([2]*N))
    A = A + np.diag([-1]*(N-1), 1) + np.diag([-1]*(N-1), -1)
    A = A/(2*h**2)
    A = A + np.diag(-Z/r + V_H + V_X + V_C)

    eig_val, vr = eig(A)
    i_min = np.argmin(eig_val)

    return eig_val[i_min], vr[:,i_min]

def solve_task_1():
    print("### TASK 1 ###")
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


    R = np.linspace(-1,1,100)
    wv = [phi(r, C, alpha) for r in R]

    plt.figure()
    plt.xlabel('Iterations')
    plt.ylabel('Energy')
    plt.plot(energies, 'ro')
    plt.plot(energies)
    plt.savefig('images/task1_conv.png')    

    plt.figure()
    plt.xlabel('Density')
    plt.ylabel('Distance')
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
            Z[i][j] = phi(np.sqrt(x**2 + y**2), C, alpha)

    surf_fig = plt.figure()
    ax = surf_fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap=cm.viridis)
    '''

def solve_task_2():
    # Task 2
    print("### TASK 2 ###")
    R = np.linspace(R_MIN,R_MAX,N)
    E_pot = V(R)
    print("See figure for result")

    # Figures
    plt.figure()
    plt.xlabel('Potential Energy')
    plt.ylabel('Distance')
    plt.plot(R, E_pot, R, [V_H(r) for r in R],'r-.')
    plt.savefig('images/task2.png')

def solve_task_3():
    # Task 3
    print("### TASK 3 ###")
    R = np.linspace(R_MIN,R_MAX,N)
    E_ks, wv = KS(R, 1)
    print("Ground state energy: {}".format(E_ks))
    n = abs(wv)**2
    n = n/np.trapz(n,R)

    plt.figure()
    plt.xlabel('Distance')
    plt.ylabel('Density')
    plt.plot(R, n)
    plt.savefig('images/task3.png')

def solve_task_4():
    pass

solve_task_1()
solve_task_2()
solve_task_3()
#solve_task_4()

plt.show()