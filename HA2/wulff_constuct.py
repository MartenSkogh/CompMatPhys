#!/usr/bin/env python
import numpy as np
from ase.io import read
from ase.cluster.wulff import wulff_construction
from ase.visualize import view
from ase.units import J, m
from os import listdir
from pprint import pprint

path = '../hebbe_import/surface_energies'

E_bulk = -3.73632531761 # eV

E_100 = []
E_111 = []

for facet in ['100', '111']:
    with open('surface_energy_{}.txt'.format(facet), 'w') as f:
        for N in range(1,14, 2):
            slab = read('fcc{}_slab-{}.txt'.format(facet, N))
            cell = slab.get_cell()
            area = np.linalg.norm(np.cross(cell[0], cell[1]))
            sigma = (1/2*area)*(slab.get_potential_energy() - N*E_bulk)
            f.write('{} {}\n'.format(N, sigma/(J/m**2)))

            if facet is '100':
                E_100.append(sigma)
            if facet is '111':
                E_111.append(sigma)

            print('{}, {}: {}'.format(facet, N, slab.get_potential_energy()))

for i in [1000, 10000, 100000]:
    atoms = wulff_construction('Al',
                               surfaces=[(1,0,0), (1,1,1)],
                               energies=[E_100[-1], E_111[-1]],
                               size=i,
                               structure='fcc',
                               rounding='below')
    atoms.center(vacuum=0)
    view(atoms)

