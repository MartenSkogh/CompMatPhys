#!/usr/bin/env python
import numpy as np
from ase.io import read
import matplotlib.pyplot as plt
from pprint import pprint

path = '../hebbe_import/PW_cutoff/'

cutoffs = []
energies = [] 
diffs = []

for c in range(300, 900, 50):
    cutoffs.append(c)
    bulk = read(path + 'cutoff-{}.txt'.format(c))
    energies.append(bulk.get_potential_energy())

for i, j in zip(range(len(energies) - 1), range(1, len(energies))):
    diffs.append(energies[j]-energies[i])

plt.figure(1)
plt.xlabel('Cut-off energy [eV]')
plt.ylabel('Potential energy [eV]')
plt.plot(cutoffs, energies)
plt.plot(cutoffs, energies,'ro')
plt.grid(True)
plt.savefig('PW_cutoff.png', bbox_inches='tight')


plt.figure(2)
plt.xlabel('Cut-off energy [eV]')
plt.ylabel('Potential energy diff [eV]')
plt.plot(cutoffs[1:], diffs)
plt.grid(True)

plt.show()
