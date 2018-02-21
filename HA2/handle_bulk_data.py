#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from pprint import pprint
from ase.eos import EquationOfState

path = '../hebbe_import/bulk/'

atoms = read(path + 'bulk.txt@:')

volumes = [atom.get_volume() for atom in atoms]
energies = [atom.get_potential_energy() for atom in atoms]

eos = EquationOfState(volumes, energies)

v0, e0, B = eos.fit()
a = (4*v0)**(1/3) # there are 4 atoms per unit cell in an fcc

print(a)

eos.plot()
eos.plot('images/Al-eos.png')
