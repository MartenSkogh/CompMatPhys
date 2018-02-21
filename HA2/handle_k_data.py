#!/usr/bin/env python
import numpy as np
from ase.io import read
import matplotlib.pyplot as plt
from pprint import pprint

path = '../hebbe_import/kpts/'

ks = []
energies = [] 
diffs = []

for k in range(4, 25):
    ks.append(k)
    bulk = read(path + 'k-{}.txt'.format(k))
    energies.append(bulk.get_potential_energy())

for i, j in zip(range(len(energies) - 1), range(1, len(energies))):
    diffs.append(energies[j]-energies[i])

plt.figure(1)
plt.xlabel('Number of k-points')
plt.ylabel('Cut-off energy [eV]')
plt.plot(ks, energies)
plt.plot(ks, energies,'ro')
plt.grid(True)

plt.savefig('images/k-points_PW_dzt.png')
plt.show()
