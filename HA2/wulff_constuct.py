#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from ase.cluster.wulff import wulff_construction
from ase.visualize import view
from ase.units import J, m
from os import listdir
from pprint import pprint

path = '../hebbe_import/surface_energies/'
ads_path = '../hebbe_import/surface_energies_ads/'

E_bulk = -3.73632531761 # eV

N_list = range(3,22, 3)
E_100 = []
E_111 = []

def get_sigma(slab_file, N):
    slab = read(slab_file)
    cell = slab.get_cell()
    area = np.linalg.norm(np.cross(cell[0], cell[1]))
    sigma = 1/(2*area)*(slab.get_potential_energy() - N*E_bulk)
    return sigma



for facet in ['100', '111']:
    with open('surface_energy_{}.txt'.format(facet), 'w') as f:
        for N in N_list:
            slab_filename = path + 'fcc{}_slab-{}.txt'.format(facet, N)
            sigma = get_sigma(slab_filename, N)

            f.write('{} {}\n'.format(N, sigma))

            if facet is '100':
                E_100.append(sigma)
            if facet is '111':
                E_111.append(sigma)

            print('{}, {}: {}'.format(facet, N, sigma))

for i in [10000]:
    atoms = wulff_construction('Al',
                               surfaces=[(1,0,0), (1,1,1)],
                               energies=[E_100[-1], E_111[-1]],
                               size=i,
                               structure='fcc',
                               rounding='below')
    atoms.center(vacuum=10.0)
    view(atoms)

plt.plot(N_list, E_100)
plt.plot(N_list, E_111)
plt.grid(True)
plt.show()
