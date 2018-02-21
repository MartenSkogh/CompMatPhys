#!/usr/bin/env python
from ase import Atoms
from ase.build import fcc100, fcc111, add_adsorbate
from ase.constraints import FixAtoms
from ase.visualize import view

slab = fcc111('Al',size=(2,2,2))
carb_monox = Atoms('CO', positions=[(0,0,0),(0,0,1.13)])
add_adsorbate(slab, carb_monox, 2, 'fcc')

slab.center(vacuum=7.5, axis=2)

view(slab)
