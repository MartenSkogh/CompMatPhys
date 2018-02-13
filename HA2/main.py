#!/usr/bin/env python
from gpaw import GPAW
from ase.build import bulk

al = bulk('Al', 'fcc', a=4.05, cubic=False)

calc = GPAW(mode='lcao',
            basis='dzp',
            h=0.18,
            xc='PBE',
            kpts=(12,12,12),
            txt='out.txt')

al.set_calculator(calc)
al.get_potential_energy()